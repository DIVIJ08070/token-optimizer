import { NextRequest, NextResponse } from 'next/server';
import {
  queryVectorStore,
  apply3LayerDedup,
  expandWithNeighbors,
  applyUpstreamDedup,
  getVectorStore,
} from '@/services/vector.service';
import { generateAnswer, rerankChunks, expandQuery, verifyEvidence } from '@/services/chat.service';
import { isAggregationQuery } from '@/lib/queryClassifier';
import { runMapReduceAggregation } from '@/lib/mapReduceAggregator';

// ---------------------------------------------------------------------------
// Parameters (Section 5)
// ---------------------------------------------------------------------------

/** Upstream per-source cap on fused-100 before the cross-encoder (Task B). */
const UPSTREAM_MAX_PER_PDF = 3;

/** Final dedup per-source cap — fixes the Infinity bug (Task A). */
const FINAL_MAX_PER_PDF = 3;

// ---------------------------------------------------------------------------
// POST handler
// ---------------------------------------------------------------------------

export async function POST(req: NextRequest) {
  try {
    const body = await req.json();
    let { question, debug } = body;

    if (Array.isArray(question)) question = question[0];
    if (typeof question !== 'string') question = String(question || '');

    if (!question.trim()) {
      return NextResponse.json({ error: 'Question is required' }, { status: 400 });
    }

    const isDebug = Boolean(debug);

    if (isAggregationQuery(question)) {
      return handleAggregationPath(question, isDebug);
    }

    return handleNormalPath(question, isDebug);

  } catch (error: any) {
    console.error('[Backend] Chat Route Error:', error);
    return NextResponse.json({ error: error.message }, { status: 500 });
  }
}

// ---------------------------------------------------------------------------
// Aggregation path — Tasks G, I-L, N-Q
// ---------------------------------------------------------------------------

async function handleAggregationPath(question: string, isDebug: boolean) {
  console.log('[Backend] *** Map-Reduce Aggregation path ***');

  const store   = await getVectorStore();
  const chunks  = store.getAllChunks();
  const embedder = store.embeddings;
  const apiKey  = process.env.GROQ_API_KEY ?? '';

  if (chunks.length === 0) {
    return NextResponse.json({
      answer: "I couldn't find this information in the uploaded PDFs.",
      sources: [],
    });
  }

  // Run the full map-reduce pipeline (Tasks I-L + N-Q).
  // Handles its own rate-limit retries internally.
  const result = await runMapReduceAggregation(question, chunks, embedder, apiKey);

  const responsePayload: any = {
    answer:  result.answer,
    sources: result.sources,
  };

  // Task H — expose retrieved source ids in debug mode.
  if (isDebug) {
    responsePayload.retrievedSources = result.retrievedSources;
    responsePayload.pathUsed = 'map-reduce-aggregation';
  }

  return NextResponse.json(responsePayload);
}

// ---------------------------------------------------------------------------
// Normal path — Tasks A + B
// ---------------------------------------------------------------------------

async function handleNormalPath(question: string, isDebug: boolean) {
  // 1. Query Expansion (LLM-based).
  const expandedQueries = await expandQuery(question);

  // 2. Hybrid retrieve, top 100 — wide net for recall.
  const chunks = await queryVectorStore(expandedQueries, 100);

  if (!chunks || chunks.length === 0) {
    return NextResponse.json({
      answer: "I couldn't find this information in the uploaded PDFs.",
      sources: [],
    });
  }

  // 3. Task B — Upstream content-dedup + per-source cap before the reranker cut.
  //    Prevents majority clusters from monopolising the top-30 rerank slots.
  const dedupedChunks = applyUpstreamDedup(chunks, UPSTREAM_MAX_PER_PDF);
  console.log(`[Backend] After upstream dedup: ${chunks.length} → ${dedupedChunks.length} chunks`);

  // 4. Cross-Encoder Reranking → Top 30.
  const topReranked = await rerankChunks(question, dedupedChunks, 30);
  console.log('[Backend] After Reranker (Top 30):', topReranked.map((c: any) => `Page ${c.metadata.pageNumber}`).join(', '));

  // 5. Neighbour Expansion on Top 30.
  const expandedChunks = await expandWithNeighbors(topReranked);
  console.log('[Backend] After Expansion:', expandedChunks.map((c: any) => `Page ${c.metadata.pageNumber}`).join(', '));

  // 6. Task A — 3-Layer Deduplication with fixed per-source cap (was Infinity).
  const finalChunks = apply3LayerDedup(expandedChunks, 30, FINAL_MAX_PER_PDF);
  console.log('[Backend] After Dedup:', finalChunks.map((c: any) => `Page ${c.metadata.pageNumber}`).join(', '));

  // 7. LLM answer generation (70B, JSON mode).
  const jsonResult  = await generateAnswer(finalChunks, question);

  // 8. Deterministic evidence verification.
  const finalResult = verifyEvidence(jsonResult, finalChunks);

  // 9. Raw context for client display.
  const rawAnswer = finalChunks.map((c: any) => c.pageContent).join('\n\n---\n\n');

  const responsePayload: any = {
    answer:    finalResult.answer,
    sources:   finalResult.sources,
    rawAnswer,
  };

  // Task H — expose retrieved source ids in debug mode.
  if (isDebug) {
    responsePayload.retrievedSources = [...new Set(finalChunks.map((c: any) => c.metadata.pdfName))];
    responsePayload.pathUsed = 'normal';
  }

  return NextResponse.json(responsePayload);
}
