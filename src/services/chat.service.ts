import { pipeline, env } from '@xenova/transformers';

// Configure transformers.js
env.allowLocalModels = false;

async function callGroq(messages: any[], model = 'llama-3.3-70b-versatile', max_tokens = 350, temperature = 0.1, response_format?: any) {
  const apiKey = process.env.GROQ_API_KEY ;
  
  const body: any = {
    model,
    messages,
    max_tokens,
    temperature
  };
  if (response_format) body.response_format = response_format;

  const response = await fetch('https://api.groq.com/openai/v1/chat/completions', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${apiKey}`
    },
    body: JSON.stringify(body)
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`Groq API Error: ${response.status} ${errorText}`);
  }

  const data = await response.json();
  return data.choices[0].message.content.trim();
}

/**
 * Translate a user's message into English for retrieval. The FAQ store is in
 * English, and the local embedder handles English far better than romanized
 * Gujarati/Hindi — so searching with the English version of the query is what
 * makes multilingual matching accurate. Uses the cheapest/fastest Groq model.
 * Falls back to the original text on any failure.
 */
export async function translateToEnglish(text: string): Promise<string> {
  const system = `You translate a customer's message to a business into literal English. The input may be Hindi, Gujarati, Hinglish, or romanized (Latin-script) Indian language.
STRICT rules:
- Translate ONLY what is written. Do NOT answer the question, add context, or guess.
- Do NOT invent, add, or substitute any company/product names. Keep names that ARE present (e.g. "Wonder App") exactly; if no name is present, add none.
- If you cannot translate a word, keep it as-is rather than guessing.
- Output ONLY the literal English translation — one short sentence, no quotes, no notes.`;
  try {
    // 70B — the 8B model hallucinates/mistranslates romanized Gujarati, which
    // wrecks retrieval accuracy. This is the accuracy-critical step, so it's
    // worth the better model (~$0.001/query, only for non-English questions).
    const out = await callGroq(
      [{ role: 'system', content: system }, { role: 'user', content: text }],
      'llama-3.3-70b-versatile',
      120,
      0,
    );
    return out?.trim() || text;
  } catch (e) {
    console.error('[Translate→EN] Failed, using original:', e);
    return text;
  }
}

/**
 * Translate a finished answer into the user's language (Hindi/Gujarati),
 * matching the casual WhatsApp tone. Uses the cheapest/fastest Groq model.
 * Callers cache the result so each answer is translated at most once per
 * language. Falls back to the original English answer on any failure.
 */
export async function translateAnswer(
  answer: string,
  langName: 'Hindi' | 'Gujarati',
  userQuestion: string,
  script: 'roman' | 'native' = 'roman',
): Promise<string> {
  const scriptRule = script === 'roman'
    ? `- CRITICAL: Write ONLY in romanized ${langName} using Latin/English letters (e.g. "tamne madad kari shaku"). Do NOT use Devanagari or Gujarati script at all.`
    : `- Write in the native ${langName} script.`;

  const system = `You translate a friendly WhatsApp business chatbot's reply into ${langName}.
Rules:
${scriptRule}
- Keep it warm, short, and natural — like a real person texting, not a textbook.
- Preserve ALL facts, numbers, names, and emojis exactly. Do not add or remove information.
- Output ONLY the translated reply. No quotes, no notes, no preamble.`;
  const user = `User's message: "${userQuestion}"\n\nReply to translate into ${langName}:\n${answer}`;

  try {
    // 70B for translation quality (esp. Gujarati). This runs at most ONCE per
    // (answer, language) — the result is cached on the pair — so the better
    // model costs nothing on repeats.
    const out = await callGroq(
      [{ role: 'system', content: system }, { role: 'user', content: user }],
      'llama-3.3-70b-versatile',
      400,
      0.2,
    );
    return out?.trim() || answer;
  } catch (e) {
    console.error('[Translate] Failed, serving original:', e);
    return answer;
  }
}

export async function expandQuery(question: string): Promise<string[]> {
  const prompt = `You are a query expansion engine. Generate exactly 3 alternative search queries for the user's question.

Rules:
- Preserve the original intent.
- Do NOT change the task.
- Do NOT ask a different question.
- Keep all key entities.
- Separate each variant with a comma.
- Do not include quotes or bullet points.`;
  
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

export function verifyEvidence(jsonResult: any, chunks: any[]) {
  const validChunkIds = new Set(chunks.map((c: any) => c.metadata.chunkIndex));
  const validCitations: any[] = [];
  
  for (const ev of jsonResult.evidence || []) {
    if (ev.chunkId !== undefined && validChunkIds.has(ev.chunkId)) {
      const sourceChunk = chunks.find((c: any) => c.metadata.chunkIndex === ev.chunkId);
      if (sourceChunk) {
        validCitations.push({
          pdf: sourceChunk.metadata.pdfName,
          page: sourceChunk.metadata.pageNumber,
          chunkId: ev.chunkId
        });
      }
    } else {
      console.warn(`[Backend] Deterministic Verification FAILED for chunkId ${ev.chunkId}. Stripped from citations.`);
    }
  }
  
  const uniqueCitations = Array.from(new Set(validCitations.map(s => JSON.stringify(s)))).map(s => JSON.parse(s as string));
  
  return {
    answer: jsonResult.answer || "No answer generated.",
    sources: uniqueCitations
  };
}

export async function generateAnswer(contextChunks: any[], question: string) {
  const context = contextChunks.map((c: any) => `[Chunk ID: ${c.metadata.chunkIndex}, Source: ${c.metadata.pdfName}, Page ${c.metadata.pageNumber}]\n${c.pageContent}`).join('\n\n');

  const systemPrompt = `You are a precise, evidence-based assistant. Answer the user's question using ONLY the provided context chunks.
Rules:
- Answer concisely and directly. Do not pad or repeat yourself.
- Only use information from the context. If the context does not contain enough to answer, say so honestly.
- Never invent facts, names, numbers, or details not present in the context.
- Cite only the Chunk IDs that directly support your answer.
- You must generate EXACTLY 4 rephrasings/synonyms of the user's question to capture different ways users might ask the same thing. One must be a natural English synonym, one in conversational Hindi (romanized/Hinglish), one in Gujarati (romanized), and one in highly colloquial Hinglish.
You must output valid JSON exactly matching this structure:
{
  "answer": "Your answer here, written as a clear human-readable response.",
  "grounded_quote": "Exact phrase copied character-for-character from the context that supports the answer",
  "rephrasings": [
    "English synonym question",
    "Hindi romanized question",
    "Gujarati romanized question",
    "Hinglish colloquial question"
  ],
  "evidence": [
    {"chunkId": 17},
    {"chunkId": 23}
  ]
}
Only cite Chunk IDs that actively support your answer. Rely ONLY on the provided context.`;
  
  const maxTokens = 3500;
  const userPrompt = `Context:\n${context}\n\nQuestion: ${question}`;

  console.log("[Backend] Generating Answer in JSON Mode...");

  let raw = "";
  try {
    raw = await callGroq([
      { role: 'system', content: systemPrompt },
      { role: 'user', content: userPrompt }
    ], 'llama-3.3-70b-versatile', maxTokens, 0.1, { type: "json_object" });
    
  } catch (error) {
    console.error("[Backend] Groq API Error:", error);
    return { answer: "Error generating response.", evidence: [] };
  }

  try {
    const parsed = JSON.parse(raw);
    return {
      answer: parsed.answer || "No answer generated.",
      grounded_quote: parsed.grounded_quote || "",
      rephrasings: Array.isArray(parsed.rephrasings) ? parsed.rephrasings : [],
      evidence: Array.isArray(parsed.evidence) ? parsed.evidence : []
    };
  } catch(e) {
    console.error("[Backend] JSON Parse Failed:", e);
    return { answer: raw, grounded_quote: "", evidence: [] };
  }
}
