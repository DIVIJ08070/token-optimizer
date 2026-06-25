#!/usr/bin/env node
/**
 * eval/runEval.mjs
 *
 * Threshold calibration script for the FAQ chatbot.
 *
 * Usage:
 *   node eval/runEval.mjs
 *   node eval/runEval.mjs --threshold 0.006   (test a specific value)
 *
 * Requires:
 *   - data/faq-store.json to exist with approved + indexed pairs
 *   - eval/eval-questions.json to have labelled questions
 *
 * Reports:
 *   - Correct answers (in-scope, right pair matched)
 *   - Correctly refused (out-of-scope, score below threshold)
 *   - False positives (matched but wrong pair, or matched when out-of-scope)
 *   - False negatives (in-scope question refused)
 *   - Threshold sweep table from 0.003 to 0.015
 */

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT      = path.join(__dirname, '..');

// ---------------------------------------------------------------------------
// Load store from disk
// ---------------------------------------------------------------------------

const STORE_FILE = path.join(ROOT, 'data', 'faq-store.json');
const EVAL_FILE  = path.join(ROOT, 'eval', 'eval-questions.json');

if (!fs.existsSync(STORE_FILE)) {
  console.error('❌  data/faq-store.json not found. Upload PDFs and index pairs first.');
  process.exit(1);
}

if (!fs.existsSync(EVAL_FILE)) {
  console.error('❌  eval/eval-questions.json not found. Create it using eval-questions.template.json as a guide.');
  process.exit(1);
}

const storeData = JSON.parse(fs.readFileSync(STORE_FILE, 'utf-8'));
const evalData  = JSON.parse(fs.readFileSync(EVAL_FILE, 'utf-8'));

const pairs = (storeData.pairs ?? []).filter(p => p.status === 'approved' && p.question_embedding);
const evalQs = Array.isArray(evalData) ? evalData : evalData.questions ?? [];

if (pairs.length === 0) {
  console.error('❌  No approved+indexed pairs found. Run "Index Approved Pairs" in the UI first.');
  process.exit(1);
}

if (evalQs.length === 0) {
  console.error('❌  No eval questions found in eval-questions.json.');
  process.exit(1);
}

console.log(`\n📊  FAQ Eval — ${pairs.length} indexed pairs, ${evalQs.length} eval questions\n`);

// ---------------------------------------------------------------------------
// Local cosine + BM25 (mirrors faq-store.service.ts — no imports needed)
// ---------------------------------------------------------------------------

function tokenize(text) {
  return text.toLowerCase().match(/\b\w+\b/g) || [];
}

function cosine(a, b) {
  let dot = 0, na = 0, nb = 0;
  for (let i = 0; i < a.length; i++) { dot += a[i]*b[i]; na += a[i]*a[i]; nb += b[i]*b[i]; }
  return (na === 0 || nb === 0) ? 0 : dot / (Math.sqrt(na) * Math.sqrt(nb));
}

// Build in-memory search docs from stored pairs
const docs = [];
for (const pair of pairs) {
  if (!pair.question_embedding) continue;
  const qToks = tokenize(pair.question);
  docs.push({ id: pair.id, embedding: pair.question_embedding, tokens: qToks, text: pair.question });

  if (pair.rephrasing_embeddings) {
    for (let i = 0; i < pair.rephrasings.length; i++) {
      const rToks = tokenize(pair.rephrasings[i]);
      docs.push({
        id: pair.id,
        embedding: pair.rephrasing_embeddings[i] ?? pair.question_embedding,
        tokens: rToks,
        text: pair.rephrasings[i],
      });
    }
  }
}

// BM25 stats
const N = docs.length;
let totalLen = 0;
const df = {};
for (const d of docs) {
  totalLen += d.tokens.length;
  const unique = [...new Set(d.tokens)];
  for (const t of unique) df[t] = (df[t] ?? 0) + 1;
}
const avgdl = N > 0 ? totalLen / N : 1;
const idf = {};
for (const t in df) {
  const n = df[t];
  idf[t] = Math.log((N - n + 0.5) / (n + 0.5) + 1);
}

function bm25Score(queryTokens, doc) {
  const k1 = 1.5, b = 0.75;
  const dl = doc.tokens.length;
  const tf = {};
  for (const t of doc.tokens) tf[t] = (tf[t] ?? 0) + 1;
  let score = 0;
  for (const q of queryTokens) {
    const f = tf[q] ?? 0;
    if (f === 0) continue;
    const idfVal = idf[q] ?? 0;
    score += idfVal * (f * (k1 + 1)) / (f + k1 * (1 - b + b * (dl / avgdl)));
  }
  return score;
}

const WEIGHT_VEC  = 0.7;
const WEIGHT_BM25 = 0.3;
const RRF_K       = 60;

// ---------------------------------------------------------------------------
// Local bge-small embedding (using @xenova/transformers)
// ---------------------------------------------------------------------------

let extractor = null;

async function embedText(text) {
  if (!extractor) {
    const { pipeline, env } = await import('@xenova/transformers');
    env.allowLocalModels = false;
    extractor = await pipeline('feature-extraction', 'Xenova/bge-small-en-v1.5');
    console.log('  [Embedder] bge-small loaded (local, no network)');
  }
  const out = await extractor(text, { pooling: 'mean', normalize: true });
  return Array.from(out.data);
}

// ---------------------------------------------------------------------------
// Search
// ---------------------------------------------------------------------------

async function search(query) {
  const qEmb    = await embedText(query);
  const qTokens = tokenize(query);

  const vecScores  = docs.map((d, i) => ({ i, score: cosine(qEmb, d.embedding) }));
  const bm25Scores = docs.map((d, i) => ({ i, score: bm25Score(qTokens, d)    }));

  vecScores .sort((a, b) => b.score - a.score);
  bm25Scores.sort((a, b) => b.score - a.score);

  const rrfMap = new Array(docs.length).fill(0);
  vecScores .forEach(({ i }, rank) => { rrfMap[i] += WEIGHT_VEC  * (1 / (RRF_K + rank + 1)); });
  bm25Scores.forEach(({ i }, rank) => { rrfMap[i] += WEIGHT_BM25 * (1 / (RRF_K + rank + 1)); });

  const indexed = rrfMap.map((score, i) => ({ i, score }));
  indexed.sort((a, b) => b.score - a.score);

  // Deduplicate by pair id
  const seen = new Set();
  const results = [];
  for (const { i, score } of indexed) {
    const d = docs[i];
    if (seen.has(d.id)) continue;
    seen.add(d.id);
    const pair = pairs.find(p => p.id === d.id);
    results.push({ pair, score, matchedText: d.text });
    if (results.length >= 5) break;
  }
  return results;
}

// ---------------------------------------------------------------------------
// Evaluate at a specific threshold
// ---------------------------------------------------------------------------

function evaluate(results, threshold) {
  let correct = 0, refused = 0, falsePos = 0, falseNeg = 0;
  const details = [];

  for (const item of results) {
    const { evalQ, top } = item;
    const score = top?.score ?? 0;
    const answered = score >= threshold;

    if (evalQ.outOfScope) {
      if (answered) {
        falsePos++;
        details.push({ q: evalQ.question, result: '❌ FALSE POS', score, pairId: top?.pair?.id });
      } else {
        refused++;
        details.push({ q: evalQ.question, result: '✅ REFUSED', score });
      }
    } else {
      if (answered) {
        const matchOk = evalQ.expectedPairId == null || top?.pair?.id === evalQ.expectedPairId;
        if (matchOk) {
          correct++;
          details.push({ q: evalQ.question, result: '✅ CORRECT', score, pairId: top?.pair?.id });
        } else {
          falsePos++;
          details.push({ q: evalQ.question, result: '❌ WRONG PAIR', score, expected: evalQ.expectedPairId, got: top?.pair?.id });
        }
      } else {
        falseNeg++;
        details.push({ q: evalQ.question, result: '❌ MISSED', score });
      }
    }
  }

  return { correct, refused, falsePos, falseNeg, details };
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

async function main() {
  console.log('Running searches… (this loads bge-small locally, takes a moment)\n');

  // Pre-compute search results for all eval questions
  const allResults = [];
  for (const evalQ of evalQs) {
    process.stdout.write(`  Searching: "${evalQ.question.slice(0, 60)}"…\r`);
    const searchResults = await search(evalQ.question);
    allResults.push({ evalQ, top: searchResults[0] });
  }
  process.stdout.write(' '.repeat(80) + '\r');

  // ---------------------------------------------------------------------------
  // Threshold sweep
  // ---------------------------------------------------------------------------
  // The RRF scores are very small numbers (~ 0.003 – 0.015 range)
  const thresholds = [];
  for (let t = 0.0010; t <= 0.0200; t += 0.0005) {
    thresholds.push(parseFloat(t.toFixed(4)));
  }

  console.log('\n📈  Threshold sweep:\n');
  console.log('  THRESHOLD │ CORRECT │ REFUSED │ FALSE+ │ FALSE- │ ACCURACY');
  console.log('  ──────────┼─────────┼─────────┼────────┼────────┼─────────');

  let bestThreshold = thresholds[0];
  let bestScore = -1;

  for (const t of thresholds) {
    const { correct, refused, falsePos, falseNeg } = evaluate(allResults, t);
    const inScope    = evalQs.filter(q => !q.outOfScope).length;
    const outOfScope = evalQs.filter(q => q.outOfScope).length;
    const accuracy   = inScope + outOfScope > 0
      ? ((correct + refused) / (inScope + outOfScope) * 100).toFixed(1)
      : '0.0';
    const combined = correct + refused - falsePos - falseNeg;

    if (combined > bestScore) { bestScore = combined; bestThreshold = t; }

    const marker = (combined === bestScore && t === bestThreshold) ? ' ◀ BEST' : '';
    console.log(
      `  ${t.toFixed(4).padStart(9)} │ ${String(correct).padStart(7)} │ ${String(refused).padStart(7)} │ ${String(falsePos).padStart(6)} │ ${String(falseNeg).padStart(6)} │ ${accuracy.padStart(7)}%${marker}`
    );
  }

  // ---------------------------------------------------------------------------
  // Detailed results at best threshold
  // ---------------------------------------------------------------------------
  console.log(`\n📋  Detailed results at threshold=${bestThreshold}:\n`);
  const { correct, refused, falsePos, falseNeg, details } = evaluate(allResults, bestThreshold);

  for (const d of details) {
    console.log(`  ${d.result.padEnd(14)} [${d.score?.toFixed(5) ?? '—'}] "${d.q.slice(0, 70)}"`);
  }

  console.log(`\n📊  Summary at threshold=${bestThreshold}:`);
  console.log(`    ✅ Correct answers:     ${correct}`);
  console.log(`    ✅ Correctly refused:   ${refused}`);
  console.log(`    ❌ False positives:     ${falsePos}`);
  console.log(`    ❌ False negatives:     ${falseNeg}`);
  console.log(`    Total eval questions:   ${evalQs.length}`);

  console.log(`\n💡  Recommended: set FAQ_THRESHOLD=${bestThreshold} in .env.local`);
  console.log(`    Current .env.local value: ${process.env.FAQ_THRESHOLD ?? '(not set, default=0.0055)'}\n`);
}

main().catch(e => { console.error('Eval failed:', e); process.exit(1); });
