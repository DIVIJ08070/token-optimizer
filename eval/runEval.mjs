// runEval.mjs  --  node >= 18 (uses global fetch)
// -----------------------------------------------------------------------------
// Scores your /api/chat against the gold questions in queries.jsonl.
// Measures three things SEPARATELY:
//   * retrieval recall  -- of the sources that hold a relevant value, how many
//                          did the pipeline actually surface?
//   * contradiction     -- did the answer reflect >=2 distinct value-groups?
//   * coverage          -- raw count surfaced vs expected
//
// BEST RESULTS: expose the retrieved source ids in your API response (see
// `parseResp` below). Without that, it falls back to scanning the answer text,
// which only measures what the LLM chose to print, not what was retrieved.
//
//   RAG_URL=http://localhost:3000/api/chat node runEval.mjs
// -----------------------------------------------------------------------------
import fs from "node:fs";

const ENDPOINT = process.env.RAG_URL ?? "http://localhost:3000/api/chat";

// ADAPT: shape the request body to match your /api/chat handler.
const buildBody = (q) => ({ question: q, debug: true });

// ADAPT: pull the answer string and (ideally) retrieved source ids out of the
// response. Return retrieved=null to fall back to text scanning.
const parseResp = (j) => ({
  answer: j.answer ?? j.response ?? j.text ?? "",
  retrieved: (j.retrievedSources ?? j.sources ?? j.debug?.sources)?.map(s => String(s).replace(/\.pdf$/i, '')) ?? null,
});

const here = (p) => new URL(p, import.meta.url);
const ALL_IDS = JSON.parse(fs.readFileSync(here("./all_source_ids.json")));
const QUERIES = fs
  .readFileSync(here("./queries.jsonl"), "utf8")
  .trim()
  .split("\n")
  .map((l) => JSON.parse(l));

// match a source id in free text (tolerant of "ICC T20I" vs "ICC-T20I")
function idsInText(text) {
  const t = text.toLowerCase();
  return ALL_IDS.filter((id) => {
    if (t.includes(id.toLowerCase())) return true;
    const loose = id.replace(/-/g, "[\\s-]?");
    return new RegExp(`\\b${loose}\\b`, "i").test(text);
  });
}

let sumRecall = 0,
  contradictionHits = 0;

console.log(`Endpoint: ${ENDPOINT}\n`);
for (const q of QUERIES) {
  let answer = "",
    retrieved = null;
  try {
    const res = await fetch(ENDPOINT, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(buildBody(q.question)),
    });
    ({ answer, retrieved } = parseResp(await res.json()));
  } catch (e) {
    console.log(`${q.id}  REQUEST FAILED: ${e.message}`);
    continue;
  }

  const expected = q.expected_all_sources;
  const surfaced = retrieved ?? idsInText(answer); // prefer true retrieval set
  const hit = expected.filter((id) => surfaced.includes(id));
  const recall = hit.length / expected.length;

  // contradiction detection: did >=2 distinct value-groups get represented?
  const inAns = idsInText(answer);
  const groupsHit = Object.values(q.expected).filter((ids) =>
    ids.some((id) => inAns.includes(id)),
  ).length;
  const contradiction = groupsHit >= 2;

  sumRecall += recall;
  contradictionHits += contradiction ? 1 : 0;

  const miss = expected.length - hit.length;
  console.log(
    `${q.id.padEnd(4)} recall=${(recall * 100).toFixed(0).padStart(3)}%  ` +
      `surfaced=${hit.length}/${expected.length}  missed=${miss}  ` +
      `value-groups=${groupsHit}/${Object.keys(q.expected).length}  ` +
      `contradiction=${contradiction ? "Y" : "N"}   [${q.label}]`,
  );
}

const n = QUERIES.length;
console.log(
  `\nMEAN retrieval recall : ${((100 * sumRecall) / n).toFixed(1)}%` +
    `\nContradiction-detect  : ${((100 * contradictionHits) / n).toFixed(0)}% of questions`,
);
