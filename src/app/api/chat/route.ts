import { NextRequest, NextResponse } from 'next/server';
import { queryVectorStore } from '@/services/vector.service';
import { generateAnswer } from '@/services/chat.service';

export async function POST(req: NextRequest) {
  try {
    const { question } = await req.json();

    if (!question) {
      return NextResponse.json({ error: 'Question is required' }, { status: 400 });
    }

    // Retrieve top 5 chunks to establish a broad recall base for Hybrid RRF
    const chunks = await queryVectorStore(question, 5);

    if (!chunks || chunks.length === 0) {
      return NextResponse.json({
        answer: "I couldn't find this information in the uploaded PDFs.",
        sources: []
      });
    }

    // Prioritize precision for rulebook lookups: only send top 2 reranked chunks to the LLM
    const topChunks = chunks.slice(0, 2);

    // Send strictly filtered chunks + question to local free LLM
    const { answer, sources } = await generateAnswer(topChunks, question);

    return NextResponse.json({ answer, sources });
  } catch (error: any) {
    console.error('Chat Error:', error);
    return NextResponse.json({ error: error.message }, { status: 500 });
  }
}
