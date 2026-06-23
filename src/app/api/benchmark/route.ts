import { NextRequest, NextResponse } from 'next/server';
import { queryVectorStore } from '@/services/vector.service';
import { expandQuery, rerankChunks } from '@/services/chat.service';

const TEST_QUERIES = [
  "How many players are in a team?",
  "How many members are allowed in a playing side?",
  "Team size?",
  "Number of cricketers in a side?",
  "How many people can play?"
];

// In this simple benchmark, we assume the chunk containing the exact answer "11 players"
// or "eleven" is the golden chunk.
function isGoldenChunk(text: string) {
  const lower = text.toLowerCase();
  return lower.includes('11 players') || lower.includes('eleven players') || lower.includes('eleven (11) players');
}

export async function GET(req: NextRequest) {
  try {
    const results = [];

    for (const query of TEST_QUERIES) {
      console.log(`[Benchmark] Testing query: "${query}"`);
      
      const expandedQueries = await expandQuery(query);
      const chunks = await queryVectorStore(expandedQueries, 50);
      
      const rerankedChunks = await rerankChunks(query, chunks, 50); // Get all 50 sorted by reranker

      let hybridRank = -1;
      let finalRerank = -1;

      for (let i = 0; i < chunks.length; i++) {
        if (isGoldenChunk(chunks[i].pageContent)) {
          hybridRank = i + 1; // 1-indexed
          break;
        }
      }

      for (let i = 0; i < rerankedChunks.length; i++) {
        if (isGoldenChunk(rerankedChunks[i].pageContent)) {
          finalRerank = i + 1;
          break;
        }
      }

      results.push({
        query,
        expandedQueries,
        hybridRank: hybridRank === -1 ? 'Not in Top 50' : hybridRank,
        finalRerank: finalRerank === -1 ? 'Not in Top 50' : finalRerank,
        success: finalRerank > 0 && finalRerank <= 5
      });
    }

    return NextResponse.json({
      benchmarkResults: results,
      summary: `${results.filter(r => r.success).length} / ${results.length} queries successfully ranked the golden chunk in the Top 5.`
    });
  } catch (error: any) {
    console.error('Benchmark Error:', error);
    return NextResponse.json({ error: error.message }, { status: 500 });
  }
}
