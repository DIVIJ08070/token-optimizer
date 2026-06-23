import { NextRequest, NextResponse } from 'next/server';
import { queryVectorStore } from '@/services/vector.service';
import { generateAnswer, expandQuery, rerankChunks } from '@/services/chat.service';

export async function POST(req: NextRequest) {
  try {
    let { question } = await req.json();

    if (Array.isArray(question)) question = question[0];
    if (typeof question !== 'string') question = String(question || "");

    if (!question.trim()) {
      return NextResponse.json({ error: 'Question is required' }, { status: 400 });
    }

    // 1. Query Expansion (LLM + Cleaning)
    const expandedQueries = await expandQuery(question);

    // 2. Retrieve top 50 chunks to establish a broad recall base for Hybrid RRF
    const chunks = await queryVectorStore(expandedQueries, 50);

    if (!chunks || chunks.length === 0) {
      return NextResponse.json({
        answer: "I couldn't find this information in the uploaded PDFs.",
        sources: []
      });
    }

    // 3. Cross-Encoder Reranking
    const topChunks = await rerankChunks(question, chunks, 4);

    // 4. Get the LLM's intelligent response
    const { answer: llmAnswer, sources } = await generateAnswer(topChunks, question);

    // 5. Get the raw text chunks found by semantic search
    const rawAnswer = topChunks.map(c => c.pageContent).join('\n\n---\n\n');

    // Combine both so the user can compare
    const combinedAnswer = `**🤖 LLM Answer:**\n${llmAnswer}\n\n---\n\n**🔍 Raw Deep Reranked Search Results:**\n${rawAnswer}`;

    return NextResponse.json({
      answer: combinedAnswer,
      sources
    });
  } catch (error: any) {
    console.error('Chat Error:', error);
    return NextResponse.json({ error: error.message }, { status: 500 });
  }
}
