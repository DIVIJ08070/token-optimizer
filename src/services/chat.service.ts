import { pipeline, env } from '@xenova/transformers';

// Configure transformers.js
env.allowLocalModels = false;

async function callGroq(messages: any[], model = 'llama-3.1-8b-instant', max_tokens = 350, temperature = 0.1) {
  const apiKey = process.env.GROQ_API_KEY ;
  const response = await fetch('https://api.groq.com/openai/v1/chat/completions', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${apiKey}`
    },
    body: JSON.stringify({
      model,
      messages,
      max_tokens,
      temperature
    })
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`Groq API Error: ${response.status} ${errorText}`);
  }

  const data = await response.json();
  return data.choices[0].message.content.trim();
}

export async function expandQuery(question: string): Promise<string[]> {
  const prompt = `You are a query expansion engine. Generate exactly 3 semantic search variants (synonyms, related phrases) for the user's question. Do not answer the question. Separate each variant with a comma. Do not include quotes or bullet points.`;
  
  let generated = "";
  try {
    generated = await callGroq([
      { role: 'system', content: prompt },
      { role: 'user', content: question }
    ], 'llama-3.1-8b-instant', 50, 0.3);
  } catch (e) {
    console.error("[Backend] Query Expansion Failed:", e);
  }

  const variants = generated ? generated.split(',').map((v: string) => v.trim()).filter((v: string) => v.length > 0) : [];
  
  const cleaned = question
    .toLowerCase()
    .replace(/[^\w\s]/g, ' ')
    .replace(/\s+/g, ' ')
    .trim();

  const finalQueries = Array.from(new Set([question, cleaned, ...variants]));
  console.log("[Backend] LLM Expanded Queries:", finalQueries);
  return finalQueries;
}

let rerankerTokenizer: any = null;
let rerankerModel: any = null;

export async function rerankChunks(question: string, chunks: any[], topK: number = 4) {
  if (!chunks || chunks.length === 0) return [];
  
  if (!rerankerTokenizer || !rerankerModel) {
    console.log("[Backend] Loading Cross-Encoder Reranker...");
    const transformers = await import('@xenova/transformers');
    rerankerTokenizer = await transformers.AutoTokenizer.from_pretrained('Xenova/bge-reranker-base');
    rerankerModel = await transformers.AutoModelForSequenceClassification.from_pretrained('Xenova/bge-reranker-base');
  }

  console.log(`[Backend] Reranking ${chunks.length} chunks...`);

  // We process sequentially to avoid OOM on local memory, but cross-encoders are fast
  const scoredChunks = [];
  for (const chunk of chunks) {
    const inputs = rerankerTokenizer(question, { text_pair: chunk.pageContent, padding: true, truncation: true });
    const { logits } = await rerankerModel(inputs);
    const score = logits.data[0];
    scoredChunks.push({ chunk, score });
  }

  // Sort descending by cross-encoder logit score
  scoredChunks.sort((a, b) => b.score - a.score);
  
  // Return the reranked chunks sliced to topK
  return scoredChunks.slice(0, topK).map(s => s.chunk);
}

export async function generateAnswer(contextChunks: any[], question: string) {
  // Combine chunk texts, separated by double newlines for clear boundaries
  const context = contextChunks.map((c: any) => c.pageContent).join('\n\n');

  const systemPrompt = `You are an expert assistant. Answer ONLY using the provided context. If the answer is not explicitly stated, say:\n"I couldn't find this information in the documents."\nDo not infer or summarize unrelated sections.`;
  const userPrompt = `Context:\n${context}\n\nQuestion: ${question}`;

  console.log("[Backend] ---------- LLM CONTEXT DUMP ----------");
  console.log(context);
  console.log("[Backend] --------------------------------------");
  console.log(`[Backend] Question asked: "${question}"`);

  let answer = "";
  try {
    answer = await callGroq([
      { role: 'system', content: systemPrompt },
      { role: 'user', content: userPrompt }
    ], 'llama-3.3-70b-versatile', 350, 0.1);
    
    console.log("[Backend] Groq Generated Answer:", answer);
  } catch (error) {
    console.error("[Backend] Groq API Error:", error);
    answer = "The AI model encountered an error or API failure, but here is the exact text found in your PDFs:\n\n" + context;
  }

  // Extract sources
  const sources = contextChunks.map((c: any) => ({
    pdf: c.metadata.pdfName,
    page: c.metadata.pageNumber,
    chunk: c.metadata.chunkIndex
  }));

  // Filter unique sources
  const uniqueSources = Array.from(new Set(sources.map(s => JSON.stringify(s)))).map(s => JSON.parse(s as string));

  return {
    answer,
    sources: uniqueSources
  };
}
