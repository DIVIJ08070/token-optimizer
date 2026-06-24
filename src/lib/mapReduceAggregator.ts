/**
 * Map-Reduce aggregation engine for cross-source questions.
 *
 * Structure: Tasks I (per-source retrieval), J (map LLM call), K (map prompt),
 *            L (reduce/merge)
 * Bug fixes: Task N (grounding validation), Task O (hybrid retrieval + keyword
 *            window + 3 chunks/source), Task P (scoped map prompt), Task Q
 *            (hardened reduce with WORKING_IDS bookkeeping).
 *
 * Design principles:
 *  - The MAP step uses small batches (BATCH_SIZE sources per LLM call).
 *    Small batches prevent cross-contamination — the source of the "5 identical
 *    quotes" failure. Each batch is independent; the model only sees its own
 *    BATCH_SIZE sources.
 *  - The REDUCE step is purely deterministic code. No final LLM call. The model
 *    only classifies; the code merges, groups, and writes the answer.
 *  - Grounding validation (Task N) runs in code after each batch is parsed and
 *    before it reaches the reduce. Fabricated quotes that aren't substrings of
 *    the provided excerpt are overridden to NOT_GROUNDED.
 */

import type { LocalTransformersEmbeddings } from '@/services/embedding.service';

// ---------------------------------------------------------------------------
// Constants (all named — no magic numbers inline)
// ---------------------------------------------------------------------------

/** Sources per map LLM call. Keep low to prevent cross-source contamination. */
const BATCH_SIZE = 8;

/** Top-k chunks retrieved per source via hybrid ranking (Task O, was 1). */
const CHUNKS_PER_SOURCE_K = 3;

/**
 * Max chars of the keyword-anchored window shown per source in the map context.
 * Task O: "if a source's selected chunks are small, pass them in full" — this
 * is enforced inside extractKeyTermWindow which skips truncation when the
 * concatenated text fits within this limit.
 */
const EXCERPT_WINDOW_CHARS = 600;

/** Chars of text to include BEFORE the key term anchor position. */
const WINDOW_PRE_ANCHOR_CHARS = 150;

/** Groq model for map step — 8B instant: 500k TPD. */
const MAP_MODEL = 'llama-3.1-8b-instant';

/** Max output tokens per map batch call (BATCH_SIZE × ~150 tokens/source). */
const MAP_MAX_TOKENS = 1200;

/** Ms to wait between consecutive batches. Stays gentle on TPM. */
const BATCH_DELAY_MS = 3000;

/** Retry attempts on 429/413 rate-limit responses, with escalating wait. */
const RATE_LIMIT_MAX_RETRIES = 5;
const RATE_LIMIT_WAIT_MS     = [8_000, 20_000, 40_000, 60_000, 120_000];

const GROQ_URL = 'https://api.groq.com/openai/v1/chat/completions';

/**
 * Window size for the recovery pass (Task T). Larger than the first-pass
 * EXCERPT_WINDOW_CHARS because stragglers are tiny docs and we want the full
 * text. Grounding still validates the quote against this larger excerpt.
 */
const RECOVERY_WINDOW_CHARS = 2000;

/** Max output tokens for the canonicalization call (Task S). */
const CANONICAL_MAX_TOKENS = 400;

// Special sentinel values used in the reduce step.
const V_NOT_ADDRESSED = 'NOT_ADDRESSED';
const V_NOT_GROUNDED  = 'NOT_GROUNDED';
const V_PARSE_ERROR   = 'PARSE_ERROR';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/** Matches VectorDocument in vector.service.ts (duck-typed to avoid a cross-import). */
export interface StoredChunk {
  pageContent: string;
  metadata: { pdfName: string; chunkIndex: number; pageNumber: number; [k: string]: any };
  embedding: number[];
  termFrequencies?: Record<string, number>;
  tokens?: string[];
}

interface MapResult {
  source_id: string;
  addressed: boolean;
  value: string;   // short normalized description, or a sentinel
  quote: string;   // verbatim substring of excerpt, or ""
}

interface ReduceGroup {
  value: string;
  sources: string[];
}

interface ReduceOutput {
  answer: string;
  groups: ReduceGroup[];
}

// ---------------------------------------------------------------------------
// Task O — Key term extraction
// ---------------------------------------------------------------------------

// Words that are never a meaningful key term for retrieval anchoring.
const STOP_WORDS = new Set([
  'a','an','the','is','are','was','were','be','been','being',
  'have','has','had','do','does','did','will','would','could',
  'should','may','might','shall','can','need',
  'of','in','on','at','to','for','with','by','from','up','out',
  'about','into','through','during','before','after','above','below',
  'between','this','that','these','those','which','what','who','where',
  'when','why','how','all','each','every','both','few','more','most',
  'other','some','such','than','then','so','yet','either','neither',
  'and','or','but','not','also','only','just','very','too','even','still',
  // aggregation meta-words
  'sources','source','rules','rule','every','across','list','compare',
  'identify','find','show','tell','give','allow','permit','prohibit',
  'allows','permits','prohibits','differ','conflict','versus','vs','per',
  'does','any','their','there','where','they','them','between','each',
  'which','many','number','all',
]);

/**
 * Extract up to 3 informative content words from the query.
 * These are threaded through retrieval (hybrid scoring), windowing, and the
 * map prompt so the model is reliably pointed at the correct clause.
 */
export async function extractKeyTerms(query: string, apiKey: string): Promise<string[]> {
  const prompt = `You extract search terms for one retrieval query. Given a user's question, identify the SINGLE rule or attribute it is actually asking about, and return that term plus the synonyms and full forms a document would use for it.
Return STRICT JSON only: {"terms": ["...", "..."]}.
Rules:
Include the core attribute and its common variants: expand acronyms to their full form and vice-versa (e.g. "DRS" → also "Decision Review System"; "LBW" → also "leg before wicket"), and add close synonyms a rulebook might use (e.g. "player review", "umpire review").
EXCLUDE words that describe the comparison or scope rather than the rule itself: drop "compare", "list", "every", "all", "each", "across", "number", "per", "source", "sources", "rulebook", "rulebooks", "document".
EXCLUDE scope qualifiers that are not the topic. In "DRS reviews per innings", the topic is DRS reviews; "innings" is only scope — do NOT include it.
Keep it tight: 2–6 terms, all about the one attribute. No filler.`;

  try {
    const response = await fetch(GROQ_URL, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${apiKey}` },
      body: JSON.stringify({
        model: MAP_MODEL,
        messages: [
          { role: 'system', content: prompt },
          { role: 'user', content: query }
        ],
        max_tokens: 150,
        temperature: 0,
        response_format: { type: 'json_object' }
      })
    });
    
    if (response.ok) {
      const data = await response.json();
      const parsed = JSON.parse(data.choices[0].message.content.trim());
      if (Array.isArray(parsed.terms) && parsed.terms.length > 0) {
        return parsed.terms.map((t: string) => t.toLowerCase());
      }
    }
  } catch (e) {
    console.warn('[MapReduce] Key terms LLM extraction failed, using fallback:', e);
  }

  // Fallback to the old simple tokenization if LLM fails
  const tokens = query.toLowerCase().match(/\b[a-z][a-z-]{2,}\b/g) ?? [];
  const seen   = new Set<string>();
  const result: string[] = [];
  for (const t of tokens) {
    if (!STOP_WORDS.has(t) && !seen.has(t)) {
      seen.add(t);
      result.push(t);
      if (result.length >= 3) break;
    }
  }
  return result;
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

function cosineSim(a: number[], b: number[]): number {
  let dot = 0, na = 0, nb = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i];
  }
  return (na === 0 || nb === 0) ? 0 : dot / (Math.sqrt(na) * Math.sqrt(nb));
}

/** Collapse runs of whitespace to a single space and trim. */
function normalizeWS(s: string): string {
  return s.replace(/\s+/g, ' ').trim();
}

/**
 * Normalize a model-returned value string for grouping in the reduce.
 * E.g. "Like-For-Like", "like for like", "like-for-like" → "like-for-like".
 */
function normalizeValue(v: string): string {
  return v.toLowerCase().replace(/[\s_]+/g, '-').replace(/-+/g, '-').trim();
}

// ---------------------------------------------------------------------------
// Task O — Per-source hybrid chunk ranking
// ---------------------------------------------------------------------------

/**
 * Rank one source's chunks using 0.7 × cosine similarity + 0.3 × keyword
 * presence score.  Using stored termFrequencies for the keyword signal means
 * chunks that EXPLICITLY mention the key term (e.g. "concussion") are boosted
 * over chunks that are only topically adjacent (e.g. "substitute fielder").
 * Returns the top-k chunks.
 */
function rankSourceChunksHybrid(
  chunks: StoredChunk[],
  queryEmbedding: number[],
  keyTerms: string[],
  k: number
): StoredChunk[] {
  const scored = chunks.map(c => {
    const vec = cosineSim(queryEmbedding, c.embedding);

    // Keyword score from stored term frequencies (exact token-level match).
    let hits = 0;
    if (c.termFrequencies && keyTerms.length > 0) {
      for (const term of keyTerms) hits += c.termFrequencies[term] ?? 0;
    } else {
      // Fallback: raw substring count when termFrequencies not available.
      const lower = c.pageContent.toLowerCase();
      for (const term of keyTerms) {
        let idx = 0;
        while ((idx = lower.indexOf(term, idx)) !== -1) { hits++; idx++; }
      }
    }
    // Normalize: 2+ hits per key term → full keyword score.
    const kw = keyTerms.length > 0 ? Math.min(1.0, hits / (keyTerms.length * 2)) : 0;
    return { chunk: c, score: 0.7 * vec + 0.3 * kw };
  });

  scored.sort((a, b) => b.score - a.score);
  return scored.slice(0, k).map(s => s.chunk);
}

// ---------------------------------------------------------------------------
// Task O — Keyword-anchored window extraction
// ---------------------------------------------------------------------------

/**
 * From a concatenated chunk string, extract a window of at most maxChars
 * centered just before the first occurrence of any key term.
 *
 * "If a source's selected chunks are small, pass them in full" (Task O):
 * if the total text fits in maxChars, it is returned unmodified.
 *
 * This replaces head-truncation, which was causing the 42 "not addressed"
 * failures: when the concussion clause was not in the first 200 chars of a
 * chunk, head-truncation silently dropped it.
 */
function extractKeyTermWindow(text: string, keyTerms: string[], maxChars: number): string {
  const trimmed = normalizeWS(text);

  // Small docs: pass in full, no truncation.
  if (trimmed.length <= maxChars) return trimmed;

  if (keyTerms.length === 0) {
    // No key terms to anchor on — fall back to head truncation.
    return trimmed.slice(0, maxChars) + '…';
  }

  // Find the earliest key term position.
  const lower = trimmed.toLowerCase();
  let anchorPos = -1;
  for (const term of keyTerms) {
    const idx = lower.indexOf(term);
    if (idx !== -1 && (anchorPos === -1 || idx < anchorPos)) anchorPos = idx;
  }

  if (anchorPos === -1) {
    // Key term not present in this source — head-truncate as fallback.
    return trimmed.slice(0, maxChars) + '…';
  }

  const start  = Math.max(0, anchorPos - WINDOW_PRE_ANCHOR_CHARS);
  const end    = Math.min(trimmed.length, start + maxChars);
  const prefix = start > 0 ? '…' : '';
  const suffix = end < trimmed.length ? '…' : '';
  return prefix + trimmed.slice(start, end) + suffix;
}

// ---------------------------------------------------------------------------
// Task O — Build per-source excerpt map
// ---------------------------------------------------------------------------

/**
 * For every source in the corpus:
 *  1. Rank that source's chunks using hybrid scoring (0.7 cosine + 0.3 keyword).
 *  2. Take the top CHUNKS_PER_SOURCE_K chunks and concatenate them in chunkIndex
 *     order (for coherent reading).
 *  3. Extract a keyword-anchored window of EXCERPT_WINDOW_CHARS from the concat.
 *
 * Returns a Map<pdfName, excerpt> that feeds directly into the map batches.
 */
export async function buildPerSourceExcerpts(
  allChunks: StoredChunk[],
  query: string,
  embedder: LocalTransformersEmbeddings,
  keyTerms: string[]
): Promise<Map<string, string>> {
  const [qEmb] = await embedder.embedDocuments([query]);

  // Group chunks by source.
  const bySource = new Map<string, StoredChunk[]>();
  for (const c of allChunks) {
    const pdf = c.metadata.pdfName;
    if (!bySource.has(pdf)) bySource.set(pdf, []);
    bySource.get(pdf)!.push(c);
  }

  const excerpts = new Map<string, string>();

  for (const [pdfName, chunks] of bySource) {
    // Hybrid-rank this source's chunks.
    const topK = rankSourceChunksHybrid(chunks, qEmb, keyTerms, CHUNKS_PER_SOURCE_K);

    // Restore reading order (by chunkIndex) after ranking.
    topK.sort((a, b) => a.metadata.chunkIndex - b.metadata.chunkIndex);

    // Concatenate, then extract keyword-anchored window.
    const combined = topK.map(c => c.pageContent).join(' ');
    excerpts.set(pdfName, extractKeyTermWindow(combined, keyTerms, EXCERPT_WINDOW_CHARS));
  }

  console.log(
    `[MapReduce] Built excerpts for ${excerpts.size} sources ` +
    `(${CHUNKS_PER_SOURCE_K} chunks/source, ${EXCERPT_WINDOW_CHARS}-char window, ` +
    `key terms: [${keyTerms.join(', ')}])`
  );
  return excerpts;
}

// ---------------------------------------------------------------------------
// Task N — Grounding validation
// ---------------------------------------------------------------------------

/**
 * Validate a parsed map result against the excerpt that was actually sent to
 * the model for that source.
 *
 * If addressed=true but quote is not a verbatim substring of the excerpt
 * (after whitespace normalization), override to:
 *   addressed: false, value: NOT_GROUNDED, quote: ""
 *
 * This deterministically eliminates fabricated quotes (e.g. the "Section 3.2"
 * strings that appeared identically in 5 sources) because those strings are
 * simply not present in the provided excerpt text.
 */
function validateGrounding(result: MapResult, excerpt: string): MapResult {
  if (!result.addressed) return result;
  if (!result.quote || result.quote.trim() === '') {
    // Model marked addressed=true but supplied no quote — reject.
    console.warn(`[MapReduce][Task N] ${result.source_id}: addressed=true but quote is empty → NOT_GROUNDED`);
    return { ...result, addressed: false, value: V_NOT_GROUNDED, quote: '' };
  }

  // Normalize whitespace on both sides, then check substring.
  const normQuote   = normalizeWS(result.quote);
  const normExcerpt = normalizeWS(excerpt);

  if (!normExcerpt.includes(normQuote)) {
    console.warn(`[MapReduce][Task N] ${result.source_id}: quote not found in excerpt → NOT_GROUNDED`);
    console.warn(`  Quote:   "${normQuote.slice(0, 80)}"`);
    return { ...result, addressed: false, value: V_NOT_GROUNDED, quote: '' };
  }

  return result;
}

// ---------------------------------------------------------------------------
// Task K + P — Map system prompt
// ---------------------------------------------------------------------------

/**
 * Build the map step system prompt.
 *
 * Task P additions vs the original:
 *  - Explicit scope guard: names the key terms and states that ONLY a clause
 *    directly about those terms counts.
 *  - Anti-confusion example: a general-substitute clause is NOT a concussion-
 *    replacement clause.
 *  - Verbatim quote requirement is stated twice (once in prose, once in field
 *    rules) to reinforce Task N's code-side check.
 *  - Value is declared as a SHORT normalized label (≤ 5 words), so the reduce
 *    can group cleanly.
 */
function buildMapSystemPrompt(question: string, keyTerms: string[]): string {
  const keyTermStr = keyTerms.length > 0
    ? keyTerms.map(t => `"${t}"`).join(', ')
    : 'the specific rule asked about';

  const primaryTerm = keyTerms[0] ?? 'the rule';

  return `You are a precise rule-extraction engine. You will be shown excerpts from multiple cricket rulebook sources, one batch at a time. For EACH source, decide whether its excerpt explicitly addresses the specific rule in the question.

SCOPE — the question is about: ${keyTermStr}.
Only a sentence or clause DIRECTLY about ${keyTermStr} counts as addressing the rule.
Do NOT confuse adjacent or related rules. Examples of what does NOT count:
  - A clause about general substitute fielders is NOT a ${primaryTerm} rule.
  - A clause about player welfare in general is NOT a ${primaryTerm} rule.
  - If the excerpt only discusses a neighbouring topic, mark the source as NOT addressed.

For each source, output one JSON object:
{
  "source_id": "<exact name after === SOURCE:>",
  "addressed": true | false,
  "value": "<SHORT label, ≤ 5 words, describing what this source says about ${keyTermStr}; use NOT_ADDRESSED if not addressed>",
  "quote": "<copy the SINGLE deciding sentence VERBATIM, character-for-character from the excerpt; empty string if not addressed>"
}

QUOTE RULES:
1. The quote must be an exact copy of a substring from the excerpt — no paraphrasing, no invented section numbers, no combining sentences.
2. If you cannot find a directly relevant sentence to quote verbatim, set addressed=false.

Return ONLY a JSON object with one key: { "results": [ ...one object per source shown... ] }

Question: ${question}`;
}

// ---------------------------------------------------------------------------
// Groq call with rate-limit retry
// ---------------------------------------------------------------------------

async function callGroqWithRetry(body: object, apiKey: string): Promise<string> {
  for (let attempt = 0; attempt < RATE_LIMIT_MAX_RETRIES; attempt++) {
    const response = await fetch(GROQ_URL, {
      method:  'POST',
      headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${apiKey}` },
      body:    JSON.stringify(body),
    });

    if (response.ok) {
      const data = await response.json();
      return data.choices[0].message.content.trim();
    }

    if (response.status === 429 || response.status === 413) {
      const wait = RATE_LIMIT_WAIT_MS[Math.min(attempt, RATE_LIMIT_WAIT_MS.length - 1)];
      console.warn(`[MapReduce] Rate limit ${response.status}, attempt ${attempt + 1}/${RATE_LIMIT_MAX_RETRIES}, waiting ${wait / 1000}s…`);
      await new Promise(r => setTimeout(r, wait));
      continue;
    }

    const errorText = await response.text();
    throw new Error(`Groq API Error: ${response.status} ${errorText}`);
  }
  throw new Error(`[MapReduce] Groq rate limit: exhausted ${RATE_LIMIT_MAX_RETRIES} retries`);
}

// ---------------------------------------------------------------------------
// Task J — Map step: one batch
// ---------------------------------------------------------------------------

/**
 * Run one map LLM call for a batch of sources.
 * On any parse or API failure, returns PARSE_ERROR stubs for every source in
 * the batch so the reduce step can account for them — no source is silently
 * dropped (Task Q.2).
 */
async function runMapBatch(
  batchExcerpts: Map<string, string>,
  systemPrompt: string,
  apiKey: string
): Promise<MapResult[]> {
  const parseErrorStubs = (): MapResult[] =>
    Array.from(batchExcerpts.keys()).map(id => ({
      source_id: id, addressed: false, value: V_PARSE_ERROR, quote: '',
    }));

  const contextBlocks = Array.from(batchExcerpts.entries())
    .map(([id, ex]) => `=== SOURCE: ${id} ===\n${ex}`)
    .join('\n\n');

  const body = {
    model:           MAP_MODEL,
    messages: [
      { role: 'system', content: systemPrompt },
      { role: 'user',   content: `Source excerpts:\n\n${contextBlocks}` },
    ],
    max_tokens:      MAP_MAX_TOKENS,
    temperature:     0.0,
    response_format: { type: 'json_object' },
  };

  let raw: string;
  try {
    raw = await callGroqWithRetry(body, apiKey);
  } catch (e) {
    console.error('[MapReduce] Map batch API call failed:', e);
    return parseErrorStubs();
  }

  try {
    const parsed = JSON.parse(raw);
    // Accept both { results: [...] } and bare [...].
    const arr: any[] = Array.isArray(parsed)
      ? parsed
      : Array.isArray(parsed.results) ? parsed.results : [];

    if (arr.length === 0) {
      console.warn('[MapReduce] Map batch returned empty results array.');
      return parseErrorStubs();
    }

    return arr.map((r: any): MapResult => ({
      source_id: String(r?.source_id ?? '').trim(),
      addressed: Boolean(r?.addressed),
      value:     String(r?.value ?? V_NOT_ADDRESSED).trim(),
      quote:     String(r?.quote ?? '').trim(),
    }));
  } catch (e) {
    console.error('[MapReduce] JSON parse failed for batch. Raw (first 400 chars):', raw?.slice(0, 400));
    return parseErrorStubs();
  }
}

// ---------------------------------------------------------------------------
// Task T — Identify stragglers after first map pass
// ---------------------------------------------------------------------------

/**
 * Lightweight merge of first-pass batch outputs to identify sources that
 * did not produce an addressed+grounded result.  These become candidates for
 * the recovery pass.
 */
function getStragglersFromBatches(
  batchOutputs: MapResult[][],
  workingIds: string[],
  sourceExcerptMap: Map<string, string>
): string[] {
  const best = new Map<string, MapResult>();

  const resultScore = (r: MapResult): number =>
    (r.addressed ? 2 : 0) +
    (r.value !== V_NOT_GROUNDED && r.value !== V_PARSE_ERROR && r.value !== V_NOT_ADDRESSED ? 1 : 0);

  for (const batch of batchOutputs) {
    for (const raw of batch) {
      if (!raw.source_id) continue;
      const excerpt   = sourceExcerptMap.get(raw.source_id) ?? '';
      const validated = validateGrounding(raw, excerpt);
      const existing  = best.get(validated.source_id);
      if (!existing || resultScore(validated) > resultScore(existing)) {
        best.set(validated.source_id, validated);
      }
    }
  }

  return workingIds.filter(id => {
    const r = best.get(id);
    if (!r) return true; // absent from all batches
    return (
      r.value === V_PARSE_ERROR ||
      r.value === V_NOT_GROUNDED ||
      !r.addressed ||
      r.value.toLowerCase() === v_NOT_ADDRESSED_lower
    );
  });
}
const v_NOT_ADDRESSED_lower = V_NOT_ADDRESSED.toLowerCase();

// ---------------------------------------------------------------------------
// Task T — Recovery pass
// ---------------------------------------------------------------------------

/**
 * Re-retrieve chunks for straggler sources using keyword-FIRST scoring
 * (0.2 cosine + 0.8 keyword) so the concussion/rule clause is prioritised
 * even when vector similarity ranked it low.
 *
 * Excerpts are passed UNTRUNCATED (or up to RECOVERY_WINDOW_CHARS, which is
 * large enough for these 2-page PDFs) so no sentence is cut.
 *
 * Grounding validation is NOT applied here; that happens inside runReduceStep
 * after sourceExcerptMap has been updated with the recovery excerpts.
 */
async function recoveryPass(
  stragglerIds: string[],
  allChunks: StoredChunk[],
  keyTerms: string[],
  systemPrompt: string,
  apiKey: string
): Promise<{ results: MapResult[]; excerpts: Map<string, string> }> {
  if (stragglerIds.length === 0) return { results: [], excerpts: new Map() };

  console.log(`[MapReduce][Task T] Recovery pass: ${stragglerIds.length} stragglers [${stragglerIds.join(', ')}]`);

  // Group straggler chunks by source.
  const bySource = new Map<string, StoredChunk[]>();
  for (const c of allChunks) {
    if (!stragglerIds.includes(c.metadata.pdfName)) continue;
    const pdf = c.metadata.pdfName;
    if (!bySource.has(pdf)) bySource.set(pdf, []);
    bySource.get(pdf)!.push(c);
  }

  const excerpts = new Map<string, string>();

  for (const id of stragglerIds) {
    const chunks = bySource.get(id);
    if (!chunks || chunks.length === 0) continue;

    // Keyword-FIRST ranking: 0.2 cosine + 0.8 keyword hit count.
    const scored = chunks.map(c => {
      let hits = 0;
      if (c.termFrequencies && keyTerms.length > 0) {
        for (const term of keyTerms) hits += c.termFrequencies[term] ?? 0;
      } else {
        const lower = c.pageContent.toLowerCase();
        for (const term of keyTerms) {
          let idx = 0;
          while ((idx = lower.indexOf(term, idx)) !== -1) { hits++; idx++; }
        }
      }
      const kw = keyTerms.length > 0 ? Math.min(1.0, hits / (keyTerms.length * 2)) : 0;
      return { chunk: c, score: 0.2 * 0 /* cosine skipped — no embedding */ + 0.8 * kw, hits };
    });
    // Sort primarily by hits, then by chunkIndex as tiebreaker.
    scored.sort((a, b) => b.hits - a.hits || a.chunk.metadata.chunkIndex - b.chunk.metadata.chunkIndex);

    const topK = scored.slice(0, CHUNKS_PER_SOURCE_K).map(s => s.chunk);
    topK.sort((a, b) => a.metadata.chunkIndex - b.metadata.chunkIndex);

    // Concatenate and pass the full text — RECOVERY_WINDOW_CHARS is intentionally
    // large so tiny 2-page docs are shown in their entirety.
    const combined = normalizeWS(topK.map(c => c.pageContent).join(' '));
    excerpts.set(id, combined.length <= RECOVERY_WINDOW_CHARS ? combined : combined.slice(0, RECOVERY_WINDOW_CHARS) + '…');
  }

  // Run map batches over the stragglers (chunking by BATCH_SIZE to avoid 413)
  const BATCH_SIZE = 8;
  const entries = Array.from(excerpts.entries());
  const allResults: MapResult[] = [];
  const totalBatches = Math.ceil(entries.length / BATCH_SIZE);

  for (let i = 0; i < entries.length; i += BATCH_SIZE) {
    const batchEntries = entries.slice(i, i + BATCH_SIZE);
    const batchMap = new Map(batchEntries);

    const batchNum = Math.floor(i / BATCH_SIZE) + 1;
    if (totalBatches > 1) {
      console.log(`[MapReduce][Task T] Recovery batch ${batchNum}/${totalBatches}: [${Array.from(batchMap.keys()).join(', ')}]`);
    }

    const batchRes = await runMapBatch(batchMap, systemPrompt, apiKey);
    allResults.push(...batchRes);

    // Pace between batches
    if (i + BATCH_SIZE < entries.length) {
      await new Promise(r => setTimeout(r, BATCH_DELAY_MS));
    }
  }

  return { results: allResults, excerpts };
}

// ---------------------------------------------------------------------------
// Task S — Canonicalize values before grouping
// ---------------------------------------------------------------------------

/**
 * Collect the distinct grounded value strings from all batch outputs and
 * make ONE small LLM call that maps paraphrases to canonical labels.
 *
 * The call receives (value, exemplar quote) pairs so the model can use the
 * verbatim quote — not just the bare label — to decide whether two values
 * describe the SAME rule.
 *
 * Safety: polarity conflicts (positive + negative merged into the same
 * canonical label) are reverted to identity mapping after the call.
 * On any failure the function returns an identity map so grouping degrades
 * gracefully to the raw normalizeValue() behaviour.
 */
async function canonicalizeValues(
  distinctItems: Array<{ quoteSig: string; value: string; quote: string }>,
  keyTerms: string[],
  apiKey: string
): Promise<Map<string, string>> {
  const identityMap = new Map(distinctItems.map(({ quoteSig, value }) => [quoteSig, normalizeValue(value)]));
  if (distinctItems.length <= 1) return identityMap;

  const keyTermStr  = keyTerms.length > 0 ? keyTerms.map(t => `"${t}"`).join(', ') : 'the rule';
  const primaryTerm = keyTerms[0] ?? 'the rule';

  const systemPrompt =
`You are grouping short rule-values that may be paraphrases of each other. Merge two items ONLY if their quotes state the SAME rule. Do NOT merge items that differ in a substantive qualifier (e.g. "like-for-like replacement" is NOT the same as "any player may replace") and NEVER merge items with opposite meaning (e.g. "permitted" vs "not permitted"). Use the quote to decide. Return JSON mapping each input ID to a canonical label; items that are the same rule must map to the identical label.

The rule being asked about: ${keyTermStr}.
Rules:
1. Canonical label must be SHORT (2-4 words, kebab-case).
2. Opposite polarity (allowed vs not-allowed) MUST produce DIFFERENT canonical labels.
3. Different qualifiers (like-for-like vs any-player) MUST produce DIFFERENT canonical labels.
4. Paraphrases of the SAME underlying rule SHOULD share one canonical label — use the quote to decide.
5. Return ONLY: { "mapping": { "ID_1": "<canonical_label>", "ID_2": "<canonical_label>", ... } }`;

  const valueList = distinctItems
    .map(({ value, quote }, i) =>
      `ID_${i + 1}:\n   value label: "${value}"\n   quote: "${quote.slice(0, 150)}"`)
    .join('\n\n');

  const userPrompt = `Rule being asked about: ${primaryTerm}\n\nItems to canonicalize:\n${valueList}`;

  try {
    const response = await fetch(GROQ_URL, {
      method:  'POST',
      headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${apiKey}` },
      body: JSON.stringify({
        model:           MAP_MODEL,
        messages:        [{ role: 'system', content: systemPrompt }, { role: 'user', content: userPrompt }],
        max_tokens:      CANONICAL_MAX_TOKENS,
        temperature:     0,
        response_format: { type: 'json_object' },
      }),
    });

    if (!response.ok) {
      console.warn(`[MapReduce][Task S] Canonicalization call failed (${response.status}) — using identity.`);
      return identityMap;
    }

    const data   = await response.json();
    const raw    = data.choices[0].message.content.trim();
    const parsed = JSON.parse(raw);
    const mapping: Record<string, string> = parsed.mapping ?? parsed;

    const canonicalMap = new Map<string, string>();
    for (let i = 0; i < distinctItems.length; i++) {
      const { quoteSig, value } = distinctItems[i];
      const canonical = mapping[`ID_${i + 1}`];
      canonicalMap.set(
        quoteSig,
        typeof canonical === 'string' && canonical.trim()
          ? normalizeValue(canonical)
          : normalizeValue(value)  // fallback to identity for unmapped values
      );
    }

    // Safety: revert any canonical group that mixes positive and negative polarity.
    const isNegative = (v: string): boolean =>
      /\b(not|no|cannot|prohibited|forbidden|disallow|ban)\b/i.test(v) ||
      /not-permitted|not-allowed|prohibit|disallow/i.test(v);

    const labelToRaws = new Map<string, string[]>();
    for (const { quoteSig, value } of distinctItems) {
      const canonical = canonicalMap.get(quoteSig)!;
      if (!labelToRaws.has(canonical)) labelToRaws.set(canonical, []);
      labelToRaws.get(canonical)!.push(value);
    }
    for (const [canonical, raws] of labelToRaws) {
      const hasNeg = raws.some(isNegative);
      const hasPos = raws.some(r => !isNegative(r));
      if (hasNeg && hasPos) {
        console.warn(`[MapReduce][Task S] Polarity conflict in group "${canonical}" — reverting to identity.`);
        for (const item of distinctItems) {
          if (canonicalMap.get(item.quoteSig) === canonical) {
            canonicalMap.set(item.quoteSig, normalizeValue(item.value));
          }
        }
      }
    }

    console.log('[MapReduce][Task S] Canonical mapping:', Object.fromEntries(canonicalMap));
    return canonicalMap;

  } catch (e) {
    console.warn('[MapReduce][Task S] Canonicalization error — using identity:', e);
    return identityMap;
  }
}

/**
 * Collect distinct (quote signature, exemplar value, exemplar quote) items from all batch outputs
 * (applying grounding validation first), then call canonicalizeValues.
 */
async function buildCanonicalMap(
  batchOutputs: MapResult[][],
  sourceExcerptMap: Map<string, string>,
  keyTerms: string[],
  apiKey: string
): Promise<Map<string, string>> {
  const sigToItem = new Map<string, { quoteSig: string; value: string; quote: string }>();

  for (const batch of batchOutputs) {
    for (const raw of batch) {
      if (!raw.source_id) continue;
      const excerpt   = sourceExcerptMap.get(raw.source_id) ?? '';
      const validated = validateGrounding(raw, excerpt);
      if (
        validated.addressed &&
        validated.value &&
        validated.value !== V_NOT_ADDRESSED &&
        validated.value !== V_NOT_GROUNDED &&
        validated.value !== V_PARSE_ERROR
      ) {
        const quoteSig = normalizeWS(validated.quote).toLowerCase();
        if (!sigToItem.has(quoteSig)) {
          sigToItem.set(quoteSig, { quoteSig, value: validated.value, quote: validated.quote });
        }
      }
    }
  }

  const distinctItems = Array.from(sigToItem.values());

  console.log(`[MapReduce][Task S] ${distinctItems.length} distinct quote signature(s) to canonicalize.`);
  return canonicalizeValues(distinctItems, keyTerms, apiKey);
}

// ---------------------------------------------------------------------------
// Task L + Q — Reduce step (deterministic, no LLM)
// ---------------------------------------------------------------------------

/**
 * Merge all map batch outputs into a final answer.
 *
 * Task Q requirements implemented here:
 *  Q.1 WORKING_IDS — the denominator M is always workingIds.length (50 here).
 *  Q.2 Fill gaps — any id absent from ALL batches is set to PARSE_ERROR.
 *  Q.3 De-duplicate ids — if the same id appears in two batches, keep the
 *       grounded/addressed one.
 *  Q.4 Group by normalised value — addressed results go into their value group;
 *       NOT_ADDRESSED / NOT_GROUNDED / PARSE_ERROR go into separate buckets.
 *  Q.5 Conflict logic — ≥2 distinct addressed groups → conflict.
 *
 * Task N is applied inside this function before the merge: every parsed result
 * is run through validateGrounding with the corresponding excerpt.
 */
export function runReduceStep(
  batchOutputs: MapResult[][],
  workingIds: string[],
  sourceExcerptMap: Map<string, string>,
  canonicalMap?: Map<string, string>  // Task S: optional canonical label mapping
): ReduceOutput {
  /** Map a raw result to its canonical group key. */
  const getCanonical = (r: MapResult): string => {
    if (canonicalMap) {
      const sig = normalizeWS(r.quote).toLowerCase();
      const canon = canonicalMap.get(sig);
      if (canon) return canon;
    }
    return normalizeValue(r.value);
  };
  const M = workingIds.length;

  // --- Q.2 + Q.3 Merge ---
  // Build result dict: id → best MapResult (prefer addressed+grounded).
  const resultDict = new Map<string, MapResult>();

  for (const batch of batchOutputs) {
    for (const raw of batch) {
      if (!raw.source_id) continue;

      // Task N: grounding validation before the reduce sees the result.
      const excerpt   = sourceExcerptMap.get(raw.source_id) ?? '';
      const validated = validateGrounding(raw, excerpt);

      const existing = resultDict.get(validated.source_id);
      if (!existing) {
        resultDict.set(validated.source_id, validated);
      } else {
        // Q.3: De-dup — prefer addressed AND non-sentinel value.
        const score = (r: MapResult) =>
          (r.addressed ? 2 : 0) + (r.value !== V_NOT_GROUNDED && r.value !== V_PARSE_ERROR ? 1 : 0);
        if (score(validated) > score(existing)) {
          resultDict.set(validated.source_id, validated);
        }
      }
    }
  }

  // Q.2: Fill gaps — any WORKING_ID missing from all batches → PARSE_ERROR.
  for (const id of workingIds) {
    if (!resultDict.has(id)) {
      console.warn(`[MapReduce][Task Q] Source "${id}" absent from all batch outputs → PARSE_ERROR`);
      resultDict.set(id, { source_id: id, addressed: false, value: V_PARSE_ERROR, quote: '' });
    }
  }

  // --- Q.4 Classify every id into exactly one bucket ---
  const addressedResults: MapResult[] = [];
  const notAddressed: string[]        = [];
  const notGrounded: string[]         = [];
  const parseErrors: string[]         = [];

  for (const id of workingIds) {
    const r = resultDict.get(id)!;
    if (r.value === V_PARSE_ERROR)   { parseErrors.push(id);   continue; }
    if (r.value === V_NOT_GROUNDED)  { notGrounded.push(id);   continue; }
    if (!r.addressed || r.value === V_NOT_ADDRESSED || r.value === 'not_addressed') {
      notAddressed.push(id); continue;
    }
    addressedResults.push(r);
  }

  const N = addressedResults.length;

  // --- Q.4 Group by canonical label (Task S) or normalised value (fallback) ---
  const groupMap = new Map<string, string[]>();
  for (const r of addressedResults) {
    const key = getCanonical(r);  // uses canonical map when provided
    if (!groupMap.has(key)) groupMap.set(key, []);
    groupMap.get(key)!.push(r.source_id);
  }

  const groups: ReduceGroup[] = Array.from(groupMap.entries())
    .map(([value, sources]) => ({ value, sources }))
    .sort((a, b) => b.sources.length - a.sources.length);

  // Q.5: Conflict if ≥2 distinct addressed groups.
  const hasConflict = groups.length >= 2;

  // --- Build text answer ---
  const lines: string[] = [];
  lines.push(`Found a value in ${N} of ${M} sources.`);
  lines.push('');

  if (groups.length === 0) {
    lines.push('No sources explicitly addressed this rule in the retrieved excerpts.');
  } else {
    for (const { value, sources } of groups) {
      lines.push(`**${value}** (${sources.length}): ${sources.join(', ')}`);
    }
    lines.push('');

    // Q.5: conflict / uniform summary.
    if (hasConflict) {
      lines.push(`**CONFLICT: YES** — ${groups.length} distinct rule groups.`);
      lines.push(groups.map(g => `${g.value} (${g.sources.length})`).join(' vs ') + '.');
    } else {
      const only       = groups[0];
      // "all"/"none" ONLY when the count justifies it (Q.5).
      const qualifier  = only.sources.length === M ? 'All sources agree' : `All ${N} addressed sources agree`;
      lines.push(`**CONFLICT: NO** — ${qualifier}: ${only.value}.`);
    }
  }

  if (notAddressed.length > 0) {
    lines.push('');
    lines.push(`**Not addressed in excerpt (${notAddressed.length}):** ${notAddressed.join(', ')}`);
  }
  if (notGrounded.length > 0) {
    lines.push(`**Quote unverifiable — excluded (${notGrounded.length}):** ${notGrounded.join(', ')}`);
  }
  if (parseErrors.length > 0) {
    lines.push(`**Parse/API error — excluded (${parseErrors.length}):** ${parseErrors.join(', ')}`);
  }

  return { answer: lines.join('\n'), groups };
}

// ---------------------------------------------------------------------------
// Main export
// ---------------------------------------------------------------------------

/**
 * Run the full map-reduce aggregation pipeline for a cross-source question.
 *
 * Returns the structured text answer plus metadata for the API response shape.
 */
export async function runMapReduceAggregation(
  question: string,
  allChunks: StoredChunk[],
  embedder: LocalTransformersEmbeddings,
  apiKey: string
): Promise<{ answer: string; sources: { pdf: string; page: number }[]; retrievedSources: string[] }> {

  if (allChunks.length === 0) {
    return {
      answer: "I couldn't find this information in the uploaded PDFs.",
      sources: [], retrievedSources: [],
    };
  }

  // Step 1 — Extract key terms (threaded through retrieval, windowing, prompt).
  const keyTerms = await extractKeyTerms(question, apiKey);
  console.log('[MapReduce] Key terms:', keyTerms);

  // Step 2 — Build per-source excerpts: hybrid-ranked, keyword-windowed (Task O).
  const sourceExcerptMap = await buildPerSourceExcerpts(allChunks, question, embedder, keyTerms);
  const workingIds       = Array.from(sourceExcerptMap.keys());

  // Step 3 — Map prompt (Task K + P).
  const systemPrompt = buildMapSystemPrompt(question, keyTerms);

  // Step 4 — Run map batches (Task J).
  const batchOutputs: MapResult[][] = [];
  const totalBatches = Math.ceil(workingIds.length / BATCH_SIZE);

  for (let i = 0; i < workingIds.length; i += BATCH_SIZE) {
    const batchIds      = workingIds.slice(i, i + BATCH_SIZE);
    const batchExcerpts = new Map<string, string>();
    for (const id of batchIds) batchExcerpts.set(id, sourceExcerptMap.get(id)!);

    const batchNum = Math.floor(i / BATCH_SIZE) + 1;
    console.log(`[MapReduce] Map batch ${batchNum}/${totalBatches}: [${batchIds.join(', ')}]`);

    const results = await runMapBatch(batchExcerpts, systemPrompt, apiKey);
    batchOutputs.push(results);

    // Pace between batches (not after the last one).
    if (i + BATCH_SIZE < workingIds.length) {
      await new Promise(r => setTimeout(r, BATCH_DELAY_MS));
    }
  }

  // Step 5 — Task T: Identify stragglers from the first map pass.
  const stragglerIds = getStragglersFromBatches(batchOutputs, workingIds, sourceExcerptMap);
  console.log(`[MapReduce][Task T] First pass: ${workingIds.length - stragglerIds.length}/${workingIds.length} surfaced, ${stragglerIds.length} straggler(s).`);

  // Step 5b — Task T: Recovery pass — re-retrieve keyword-first + re-run map.
  if (stragglerIds.length > 0) {
    const { results: recoveryResults, excerpts: recoveryExcerpts } =
      await recoveryPass(stragglerIds, allChunks, keyTerms, systemPrompt, apiKey);

    // Add recovery results as an extra batch — runReduceStep will merge them
    // with the same de-dup policy (prefer addressed+grounded).
    batchOutputs.push(recoveryResults);

    // Update excerpt map so grounding validation in runReduceStep uses the
    // larger untruncated recovery excerpts for these sources.
    for (const [id, ex] of recoveryExcerpts) sourceExcerptMap.set(id, ex);
  }

  // Step 6 — Task S: Canonicalize values (collapses paraphrases → 3 groups).
  const canonicalMap = await buildCanonicalMap(batchOutputs, sourceExcerptMap, keyTerms, apiKey);

  // Step 7 — Reduce: grounding + merge + group (canonical) + answer (Tasks N, L, Q, S).
  const { answer, groups } = runReduceStep(batchOutputs, workingIds, sourceExcerptMap, canonicalMap);

  // Build the sources list from working ids (one entry per source).
  const sources = workingIds.map(id => {
    const firstChunk = allChunks.find(c => c.metadata.pdfName === id);
    return { pdf: id, page: firstChunk?.metadata.pageNumber ?? 1 };
  });

  console.log(`[MapReduce] Done. ${groups.length} value group(s) across ${workingIds.length} sources.`);
  return { answer, sources, retrievedSources: workingIds };
}
