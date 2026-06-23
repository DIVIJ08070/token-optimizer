import { Embeddings, EmbeddingsParams } from '@langchain/core/embeddings';
import { pipeline, env } from '@xenova/transformers';

// Configure transformers.js to download from HuggingFace Hub instead of local filesystem
env.allowLocalModels = false;

export class LocalTransformersEmbeddings extends Embeddings {
  private extractor: any = null;

  constructor(fields?: EmbeddingsParams) {
    super(fields ?? {});
  }

  async init() {
    if (!this.extractor) {
      // Using a small and very fast model for local embeddings
      this.extractor = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2');
    }
  }

  async embedQuery(document: string): Promise<number[]> {
    await this.init();
    const output = await this.extractor(document, { pooling: 'mean', normalize: true });
    return Array.from(output.data as Float32Array) as number[];
  }

  async embedDocuments(documents: string[]): Promise<number[][]> {
    await this.init();
    const results: number[][] = [];
    for (const text of documents) {
      const output = await this.extractor(text, { pooling: 'mean', normalize: true });
      results.push(Array.from(output.data as Float32Array) as number[]);
    }
    return results;
  }
}
