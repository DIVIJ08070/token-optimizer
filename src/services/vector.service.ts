import { LocalTransformersEmbeddings } from './embedding.service';

interface VectorDocument {
  id: number;
  pageContent: string;
  metadata: any;
  embedding: number[];
  tokens: string[];
  termFrequencies: Record<string, number>;
}

// Tokenizer function: lowercases and splits by word boundaries
function tokenize(text: string): string[] {
  return text.toLowerCase().match(/\b\w+\b/g) || [];
}

export class SimpleMemoryVectorStore {
  documents: VectorDocument[] = [];
  embeddings: LocalTransformersEmbeddings;

  // BM25 properties
  private k1 = 1.5;
  private b = 0.75;
  private avgdl = 0;
  private idf: Record<string, number> = {};

  constructor(embeddings: LocalTransformersEmbeddings) {
    this.embeddings = embeddings;
  }

  async addDocuments(chunks: any[]) {
    console.log(`[Backend] Generating embeddings for ${chunks.length} chunks...`);
    const texts = chunks.map(c => c.pageContent);
    const vectors = await this.embeddings.embedDocuments(texts);
    
    for (let i = 0; i < chunks.length; i++) {
      const tokens = tokenize(chunks[i].pageContent);
      const termFrequencies: Record<string, number> = {};
      
      for (const token of tokens) {
        termFrequencies[token] = (termFrequencies[token] || 0) + 1;
      }

      this.documents.push({
        id: this.documents.length,
        pageContent: chunks[i].pageContent,
        metadata: chunks[i].metadata,
        embedding: vectors[i],
        tokens,
        termFrequencies
      });
    }

    this.recalculateBM25();
    console.log(`[Backend] Success! Stored ${chunks.length} chunks. Total chunks in memory: ${this.documents.length}`);
  }

  private recalculateBM25() {
    const N = this.documents.length;
    let totalLength = 0;
    const documentFrequencies: Record<string, number> = {};

    for (const doc of this.documents) {
      totalLength += doc.tokens.length;
      const uniqueTokens = Object.keys(doc.termFrequencies);
      for (const token of uniqueTokens) {
        documentFrequencies[token] = (documentFrequencies[token] || 0) + 1;
      }
    }

    this.avgdl = N > 0 ? totalLength / N : 0;
    
    // Calculate IDF for each term: log( (N - n(q) + 0.5) / (n(q) + 0.5) + 1 )
    this.idf = {};
    for (const token in documentFrequencies) {
      const n = documentFrequencies[token];
      this.idf[token] = Math.log((N - n + 0.5) / (n + 0.5) + 1);
    }
  }

  private getKeywordScore(queryTokens: string[], doc: VectorDocument): number {
    let score = 0;
    const dl = doc.tokens.length;

    for (const q of queryTokens) {
      const tf = doc.termFrequencies[q] || 0;
      if (tf === 0) continue;
      
      const idf = this.idf[q] || 0;
      const numerator = tf * (this.k1 + 1);
      const denominator = tf + this.k1 * (1 - this.b + this.b * (dl / this.avgdl));
      
      score += idf * (numerator / denominator);
    }
    return score;
  }

  async similaritySearch(query: string, topK: number = 5) {
    console.log(`[Backend] Performing Hybrid Search (Semantic + BM25) for query: "${query}"...`);
    
    // 1. Vector Search
    const queryVector = await this.embeddings.embedQuery(query);
    const vectorScores = this.documents.map(doc => ({
      id: doc.id,
      score: cosineSimilarity(queryVector, doc.embedding)
    }));
    vectorScores.sort((a, b) => b.score - a.score);

    // 2. Keyword Search (BM25)
    const queryTokens = tokenize(query);
    const keywordScores = this.documents.map(doc => ({
      id: doc.id,
      score: this.getKeywordScore(queryTokens, doc)
    }));
    keywordScores.sort((a, b) => b.score - a.score);

    // 3. Reciprocal Rank Fusion (RRF)
    const RRF_K = 60;
    const rrfScores = new Map<number, { doc: VectorDocument, score: number }>();

    for (const doc of this.documents) {
      rrfScores.set(doc.id, { doc, score: 0 });
    }

    vectorScores.forEach((v, rank) => {
      const current = rrfScores.get(v.id)!;
      current.score += 1.0 / (RRF_K + rank + 1); // 1-indexed rank
    });

    keywordScores.forEach((k, rank) => {
      const current = rrfScores.get(k.id)!;
      current.score += 1.0 / (RRF_K + rank + 1);
    });

    // 4. Sort and return top results
    const combinedResults = Array.from(rrfScores.values());
    combinedResults.sort((a, b) => b.score - a.score);

    const topResults = combinedResults.slice(0, topK);
    console.log(`[Backend] Found top ${topResults.length} matches with Hybrid RRF scores:`, topResults.map(r => r.score.toFixed(4)));
    
    return topResults.map(r => r.doc);
  }
}

function cosineSimilarity(vecA: number[], vecB: number[]) {
  let dotProduct = 0;
  let normA = 0;
  let normB = 0;
  for (let i = 0; i < vecA.length; i++) {
    dotProduct += vecA[i] * vecB[i];
    normA += vecA[i] * vecA[i];
    normB += vecB[i] * vecB[i];
  }
  if (normA === 0 || normB === 0) return 0;
  return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
}

// Global instance for memory persistence during the Node.js process lifetime
const globalAny = globalThis as any;
let vectorStore: SimpleMemoryVectorStore | null = globalAny.__vectorStore || null;

export async function getVectorStore(): Promise<SimpleMemoryVectorStore> {
  if (!vectorStore) {
    console.log("[Backend] Initializing new Hybrid Vector Store memory...");
    const embeddings = new LocalTransformersEmbeddings();
    vectorStore = new SimpleMemoryVectorStore(embeddings);
    globalAny.__vectorStore = vectorStore;
  }
  return vectorStore;
}

export async function addChunksToStore(chunks: any[]) {
  const store = await getVectorStore();
  await store.addDocuments(chunks);
}

export async function queryVectorStore(query: string, topK: number = 5) {
  const store = await getVectorStore();
  return store.similaritySearch(query, topK);
}
