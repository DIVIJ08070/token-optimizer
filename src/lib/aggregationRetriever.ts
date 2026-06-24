/**
 * Tasks D, E, F — Aggregation retrieval, context builder, and system prompt.
 *
 * Used only on the aggregation path (isAggregationQuery === true).
 * The normal path continues to use the existing vector service methods.
 */

import type { LocalTransformersEmbeddings } from '@/services/embedding.service';

// ---------------------------------------------------------------------------
// Types (must match VectorDocument in vector.service.ts)
// ---------------------------------------------------------------------------

export interface AggChunk {
  pageContent: string;
  metadata: {
    pdfName: string;
    chunkIndex: number;
    pageNumber: number;
    [key: string]: any;
  };
  embedding: number[];
  [key: string]: any;
}

export interface AggChunkWithScore extends AggChunk {
  score: number;
  viaFloor: boolean;
}

// ---------------------------------------------------------------------------
// Parameters (Task D / Section 5)
// ---------------------------------------------------------------------------

export const AGG_POOL_SIZE = 200;         // global candidate pool safety cap
export const AGG_MAX_PER_PDF = 4;         // max chunks one source may contribute
export const AGG_FLOOR_PER_PDF = 1;       // min chunks guaranteed per source
export const AGG_FLOOR_MIN_SCORE = 0;     // gate floor inclusion by similarity (0 = off)

/**
 * Maximum characters taken from each chunk when building the aggregation context.
 * Keeps the total prompt under the 6k TPM limit even with 50 sources.
 * 50 sources × 1 chunk × 200 chars ≈ 2,500 tokens context.
 */
export const AGG_MAX_CHARS_PER_CHUNK = 200;

/**
 * Maximum chunks shown per source in the aggregation context.
 * Set to 1 so each source contributes exactly one excerpt to the LLM.
 * The retriever may pull more (up to AGG_MAX_PER_PDF) for quality,
 * but only the highest-ranked one per source reaches the model.
 */
export const AGG_MAX_CHUNKS_PER_SOURCE = 1;

// ---------------------------------------------------------------------------
// Internal helper
// ---------------------------------------------------------------------------

function cosineSimilarity(a: number[], b: number[]): number {
  let dot = 0, normA = 0, normB = 0;
  for (let i = 0; i < a.length; i++) {
    dot   += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  if (normA === 0 || normB === 0) return 0;
  return dot / (Math.sqrt(normA) * Math.sqrt(normB));
}

// ---------------------------------------------------------------------------
// Task D — Balanced per-source retriever
// ---------------------------------------------------------------------------

/**
 * Retrieve chunks for an aggregation question, guaranteeing at least one chunk
 * per source and preventing any single source from monopolising the pool.
 *
 * @param query       The raw user query (used to embed and score chunks).
 * @param allChunks   Every chunk currently stored in the in-memory vector store.
 * @param embedQuery  The embedder (the store's own embeddings object).
 * @param opts        Optional tuning parameters — all have defaults.
 */
export async function retrieveAggregationBalanced(
  query: string,
  allChunks: AggChunk[],
  embedQuery: LocalTransformersEmbeddings,
  opts: {
    poolSize?: number;
    maxPerPdf?: number;
    floorPerPdf?: number;
    floorMinScore?: number;
  } = {}
): Promise<AggChunkWithScore[]> {
  const poolSize      = opts.poolSize      ?? AGG_POOL_SIZE;
  const maxPerPdf     = opts.maxPerPdf     ?? AGG_MAX_PER_PDF;
  const floorPerPdf   = opts.floorPerPdf   ?? AGG_FLOOR_PER_PDF;
  const floorMinScore = opts.floorMinScore ?? AGG_FLOOR_MIN_SCORE;

  if (allChunks.length === 0) return [];

  // Step 1 — Score every chunk by cosine similarity to the query embedding.
  const [queryEmbedding] = await embedQuery.embedDocuments([query]);

  const scored: AggChunkWithScore[] = allChunks.map(chunk => ({
    ...chunk,
    score: cosineSimilarity(queryEmbedding, chunk.embedding),
    viaFloor: false,
  }));

  // Sort descending by score.
  scored.sort((a, b) => b.score - a.score);

  // Step 2 — Build the pool with a per-source cap.
  const pool: AggChunkWithScore[] = [];
  const poolCountsByPdf: Record<string, number> = {};

  for (const chunk of scored) {
    if (pool.length >= poolSize) break;

    const pdf = chunk.metadata.pdfName;
    const count = poolCountsByPdf[pdf] ?? 0;

    if (count < maxPerPdf) {
      pool.push(chunk);
      poolCountsByPdf[pdf] = count + 1;
    }
  }

  // Step 3 — Per-source floor: guarantee every source has at least floorPerPdf
  //           chunk(s) in the result, subject to score >= floorMinScore.
  const pooledPdfs = new Set(pool.map(c => c.metadata.pdfName));
  const allPdfs    = new Set(allChunks.map(c => c.metadata.pdfName));

  for (const pdf of allPdfs) {
    if (pooledPdfs.has(pdf)) continue;

    // Find the best floorPerPdf chunks for this missing source.
    const sourceBest = scored
      .filter(c => c.metadata.pdfName === pdf && c.score >= floorMinScore)
      .slice(0, floorPerPdf);

    for (const chunk of sourceBest) {
      pool.push({ ...chunk, viaFloor: true });
    }
  }

  // Final sort: pool chunks by score descending (floor chunks land at the end
  // naturally since they scored low — but we keep them visible for the model).
  pool.sort((a, b) => b.score - a.score);

  console.log(
    `[Aggregation] Balanced retrieval: pool=${pool.length} across ${
      new Set(pool.map(c => c.metadata.pdfName)).size
    } sources (floorSources=${pool.filter(c => c.viaFloor).length > 0 ? 'yes' : 'no'})`
  );

  return pool;
}

// ---------------------------------------------------------------------------
// Task E — Aggregation context builder
// ---------------------------------------------------------------------------

/**
 * Renders retrieved chunks grouped by source under explicit delimiters,
 * ordered by chunkIndex within each source.
 *
 * Format per source:
 *   === SOURCE: <pdfName> ===
 *   <chunk text(s) ordered by chunkIndex>
 */
export function buildAggregationContext(
  chunks: AggChunk[],
  maxCharsPerChunk: number = AGG_MAX_CHARS_PER_CHUNK,
  maxChunksPerSource: number = AGG_MAX_CHUNKS_PER_SOURCE
): string {
  // Group by pdfName, preserving the order sources first appear in the list
  // (the list is already sorted by relevance score, so the first chunk per
  // source is already the highest-scoring one).
  const grouped = new Map<string, AggChunk[]>();

  for (const chunk of chunks) {
    const pdf = chunk.metadata.pdfName;
    if (!grouped.has(pdf)) grouped.set(pdf, []);
    grouped.get(pdf)!.push(chunk);
  }

  const blocks: string[] = [];

  for (const [pdfName, sourceChunks] of grouped) {
    // Order within source by chunkIndex so the text reads sequentially,
    // then cap to maxChunksPerSource.
    const ordered = [...sourceChunks]
      .sort((a, b) => a.metadata.chunkIndex - b.metadata.chunkIndex)
      .slice(0, maxChunksPerSource);

    // Truncate each chunk to maxCharsPerChunk to keep total tokens manageable.
    const text = ordered
      .map(c => {
        const content = c.pageContent.trim();
        return content.length > maxCharsPerChunk
          ? content.slice(0, maxCharsPerChunk) + '…'
          : content;
      })
      .join('\n\n');

    blocks.push(`=== SOURCE: ${pdfName} ===\n${text}`);
  }

  return blocks.join('\n\n');
}

// ---------------------------------------------------------------------------
// Task F — Aggregation system prompt
// ---------------------------------------------------------------------------

export const AGGREGATION_SYSTEM_PROMPT = `You are answering a question that compares a rule ACROSS MULTIPLE SOURCES.
You are given the single most relevant excerpt from EACH source, delimited by
\`=== SOURCE: <id> ===\` followed by the excerpt.

Follow these rules exactly:
1. Produce one entry for EVERY source shown above. Never omit a source.
2. For each source, state the value it gives for the rule asked about and cite
   its id. Quote the deciding phrase where useful (keep quotes short).
3. If a source's excerpt is present but does NOT address the rule, write
   "not addressed in excerpt" for that source. Do not guess, and do not drop it.
4. After the per-source list, GROUP sources by value and state explicitly where
   they CONFLICT (e.g. "X allows … whereas Y prohibits …").
5. Begin with a coverage line: "Found a value in N of M sources."
6. NEVER write "all sources" or "none of the sources" unless every source shown
   confirms it. If you only have some, say "of the sources shown". A single
   counter-example makes a universal claim false — check before you write it.`;
