/**
 * openai.service.ts
 *
 * OpenAI wrapper for FAQ Q&A generation.
 * Used ONLY at upload time — never at chat time.
 * NEVER embeds with OpenAI. Embeddings are always local (bge-small).
 */

import OpenAI from 'openai';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface RawPair {
  question: string;
  rephrasings: string[];
  answer: string;
  grounded_quote: string;
}

// ---------------------------------------------------------------------------
// Client (lazy singleton)
// ---------------------------------------------------------------------------

let _client: OpenAI | null = null;

function getClient(): OpenAI {
  if (!_client) {
    const apiKey = process.env.GROQ_API_KEY;
    if (!apiKey) {
      throw new Error(
        '[Groq] GROQ_API_KEY is not set in .env.local. ' +
        'Add GROQ_API_KEY=gsk_... to your .env.local file.'
      );
    }
    _client = new OpenAI({ 
      apiKey,
      baseURL: 'https://api.groq.com/openai/v1'
    });
  }
  return _client;
}

// ---------------------------------------------------------------------------
// FAQ generation prompts — WhatsApp bot training data from document content
// ---------------------------------------------------------------------------

export interface GenerationConfig {
  tone?: string;
  language?: string;
}

const DEFAULT_CONFIG: GenerationConfig = {
  tone: 'short, warm sentences with one light emoji. Sounds like a helpful WhatsApp bot',
  language: 'English, Hindi (script or roman), and Hinglish (formal and casual slang)',
};

function getChunkFaqPrompt(config: GenerationConfig = DEFAULT_CONFIG): string {
  const tone = config.tone ?? DEFAULT_CONFIG.tone;
  const language = config.language ?? DEFAULT_CONFIG.language;
  
  return `You are an expert conversational AI trainer. Turn the document passage below into production-grade bot training data.

Read ONLY the passage provided. Produce every distinct inquiry or question a user might reasonably ask that this passage answers.

For each inquiry:
1. Provide a primary "question" — specific to THIS content, never generic.
2. Provide a clear, self-contained "answer" written ONLY from the passage:
   - Tone/Style: ${tone}.
   - NEVER invent facts, prices, clients, numbers, or capabilities not in the passage.
   - NEVER use generic filler ("we offer various services", "we have many products").
3. Provide exactly 8 to 12 "rephrasings". This is critical. You must generate variations in:
   - English (synonyms, different ways to ask)
   - Hindi (romanized / Hinglish, formal and casual slang)
   - Gujarati (romanized, casual conversational)
   - You MUST include at least 2 rephrasings in EACH of these languages to ensure broad semantic coverage.
   At least half must have realistic typos and missing punctuation — they should sound like real people texting fast, not textbook examples.
4. Provide a "grounded_quote". The grounded_quote must be copied character-for-character from the source — do NOT reword it, even though the answer itself is reworded. The answer and the quote are separate: answer = warm/rephrased, quote = exact.

Quality rules:
- Every rephrasing must sound like a real person, not a textbook.
- Specific price/stock/availability/discount → NEVER answer, skip those questions entirely.
- Use no knowledge beyond the passage.
- If the passage contains nothing a user would ask about, return an empty pairs list.

Return STRICT JSON only — no commentary:
{"pairs":[{"question":"...","rephrasings":["...","..."],"answer":"...","grounded_quote":"<verbatim snippet from THIS passage>"}]}`;
}

const DOC_OVERVIEW_SYSTEM_PROMPT = `You are creating high-level FAQ entries for a document. Based on the provided summary of the document's content, generate 2-4 broad overview questions such as "What is this document about?", "What are the main topics covered?", or "What are the key points?". For each question give a clear, self-contained answer and 2-3 natural rephrasings. Use no knowledge beyond what is provided.
Return STRICT JSON only:
{"pairs":[{"question":"...","rephrasings":["...","..."],"answer":"...","grounded_quote":"<verbatim phrase from the summary that supports the answer>"}]}`;

// Vision path: used when text extraction yields < PAGE_TEXT_THRESHOLD chars
function getPageVisionPrompt(config: GenerationConfig = DEFAULT_CONFIG): string {
  const tone = config.tone ?? DEFAULT_CONFIG.tone;
  const language = config.language ?? DEFAULT_CONFIG.language;

  return `You are an expert conversational AI trainer. Turn the scanned or image-based document page below into production-grade bot training data.

Read ONLY what is visible in the image. Produce every distinct inquiry or question a user might reasonably ask that this page answers.

For each inquiry:
1. Provide a primary "question" — specific to THIS content, never generic.
2. Provide a clear, self-contained "answer" written ONLY from what is visible:
   - Tone/Style: ${tone}.
   - NEVER invent facts, prices, clients, numbers, or capabilities not visible in the image.
   - NEVER use generic filler.
3. Provide exactly 8 to 12 "rephrasings". This is critical. You must generate variations in:
   - English (synonyms, different ways to ask)
   - Hindi (romanized / Hinglish, formal and casual slang)
   - Gujarati (romanized, casual conversational)
   - You MUST include at least 2 rephrasings in EACH of these languages to ensure broad semantic coverage.
   At least half must have realistic typos and missing punctuation — they should sound like real people texting fast, not textbook examples.
4. Provide a "grounded_quote" — ideally an exact transcription of text visible in the image that supports the answer. If no text supports it, leave blank.

Quality rules:
- Every rephrasing must sound like a real person, not a textbook.
- Specific price/stock/availability/discount → NEVER answer, skip those questions entirely.
- If the page contains no readable content (blank, decorative, or purely graphical with no informational text), return an empty pairs list.

Return STRICT JSON only — no commentary:
{"pairs":[{"question":"...","rephrasings":["...","..."],"answer":"...","grounded_quote":"<verbatim phrase from visible text>"}]}`;
}

// ---------------------------------------------------------------------------
// Helper: call OpenAI with retry
// ---------------------------------------------------------------------------

async function callOpenAI(
  systemPrompt: string,
  userContent: string,
  maxRetries = 2
): Promise<string> {
  const client = getClient();
  let lastError: Error | null = null;

  for (let attempt = 0; attempt <= maxRetries; attempt++) {
    try {
      const completion = await client.chat.completions.create({
        model: 'llama-3.3-70b-versatile',
        messages: [
          { role: 'system', content: systemPrompt },
          { role: 'user',   content: userContent },
        ],
        temperature: 0.2,
        response_format: { type: 'json_object' },
      });

      return completion.choices[0].message.content?.trim() ?? '{"pairs":[]}';
    } catch (e: any) {
      lastError = e;
      const isRateLimit = e?.status === 429;
      if (isRateLimit && attempt < maxRetries) {
        const wait = (attempt + 1) * 8000;
        console.warn(`[Groq] Rate limit, retrying in ${wait / 1000}s... (attempt ${attempt + 1}/${maxRetries})`);
        await new Promise(r => setTimeout(r, wait));
        continue;
      }
      throw e;
    }
  }

  throw lastError ?? new Error('[Groq] Unknown error after retries');
}

// ---------------------------------------------------------------------------
// Helper: call OpenAI vision (image + system prompt)
// ---------------------------------------------------------------------------

async function callOpenAIVision(
  systemPrompt: string,
  imageBuffer: Buffer,
  maxRetries = 2,
): Promise<string> {
  const client = getClient();
  let lastError: Error | null = null;

  const base64 = imageBuffer.toString('base64');
  const dataUrl = `data:image/png;base64,${base64}`;

  for (let attempt = 0; attempt <= maxRetries; attempt++) {
    try {
      const completion = await client.chat.completions.create({
        model: 'llama-3.2-90b-vision-preview',
        messages: [
          { role: 'system', content: systemPrompt },
          {
            role: 'user',
            content: [
              {
                type: 'image_url',
                image_url: { url: dataUrl, detail: 'high' },
              },
            ],
          },
        ],
        temperature: 0.2,
        response_format: { type: 'json_object' },
      });

      return completion.choices[0].message.content?.trim() ?? '{"pairs":[]}';
    } catch (e: any) {
      lastError = e;
      const isRateLimit = e?.status === 429;
      if (isRateLimit && attempt < maxRetries) {
        const wait = (attempt + 1) * 8000;
        console.warn(`[Groq/Vision] Rate limit, retrying in ${wait / 1000}s... (attempt ${attempt + 1}/${maxRetries})`);
        await new Promise(r => setTimeout(r, wait));
        continue;
      }
      throw e;
    }
  }

  throw lastError ?? new Error('[Groq/Vision] Unknown error after retries');
}

// ---------------------------------------------------------------------------
// Parse OpenAI JSON response
// ---------------------------------------------------------------------------

function dedupeRephrasings(rephrasings: any[]): string[] {
  if (!Array.isArray(rephrasings)) return [];
  const seen = new Set<string>();
  const unique: string[] = [];
  for (const r of rephrasings) {
    const s = String(r).trim();
    const norm = s.toLowerCase();
    if (norm && !seen.has(norm)) {
      seen.add(norm);
      unique.push(s);
    }
  }
  return unique;
}

function parsePairs(raw: string): RawPair[] {
  try {
    const parsed = JSON.parse(raw);
    const arr = Array.isArray(parsed?.pairs) ? parsed.pairs : [];
    return arr
      .filter((p: any) =>
        typeof p?.question === 'string' &&
        typeof p?.answer   === 'string' &&
        p.question.trim() && p.answer.trim()
      )
      .map((p: any): RawPair => ({
        question:       p.question.trim(),
        rephrasings:    dedupeRephrasings(p.rephrasings),
        answer:         p.answer.trim(),
        grounded_quote: typeof p.grounded_quote === 'string' ? p.grounded_quote.trim() : '',
      }));
  } catch (e) {
    console.error('[Groq] Failed to parse pairs JSON:', e);
    return [];
  }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/**
 * Generate Q&A pairs for a single chunk.
 * Called at upload time for EACH chunk.
 * Makes one OpenAI API call (gpt-4o). NOT called at chat time.
 */
export async function generateFaqPairs(
  chunkText: string,
  config?: GenerationConfig
): Promise<RawPair[]> {
  if (!chunkText || chunkText.trim().length < 50) return [];

  try {
    const prompt = getChunkFaqPrompt(config);
    const raw = await callOpenAI(prompt, chunkText);
    const pairs = parsePairs(raw);
    console.log(`[Groq] Chunk → ${pairs.length} pairs generated.`);
    return pairs;
  } catch (e: any) {
    console.error('[Groq] generateFaqPairs failed:', e?.message);
    return [];
  }
}

/**
 * Generate Q&A pairs from a rendered page image (vision path).
 * Called at upload time for image-only pages where text extraction failed.
 * NOT called at chat time.
 */
export async function generateFaqPairsFromImage(
  imageBuffer: Buffer,
  pageNumber: number,
  config?: GenerationConfig
): Promise<RawPair[]> {
  try {
    const prompt = getPageVisionPrompt(config);
    const raw = await callOpenAIVision(prompt, imageBuffer);
    const pairs = parsePairs(raw);
    console.log(`[Groq/Vision] Page ${pageNumber} → ${pairs.length} pairs generated.`);
    return pairs;
  } catch (e: any) {
    console.error(`[Groq/Vision] generateFaqPairsFromImage failed for page ${pageNumber}:`, e?.message);
    return [];
  }
}

/**
 * Generate 2–4 document-level overview Q&A pairs per PDF.
 * Called once per document after all chunks are processed.
 * NOT called at chat time.
 */
export async function generateDocumentOverviewPairs(
  allChunkTexts: string[],
  pdfName: string
): Promise<RawPair[]> {
  // Build a compact summary: first 200 chars of each chunk, joined
  const summary = allChunkTexts
    .map((t, i) => `[Chunk ${i + 1}]: ${t.slice(0, 200)}`)
    .join('\n\n')
    .slice(0, 4000); // stay within token budget

  const userContent = `Document name: "${pdfName}"\n\nContent summary:\n${summary}`;

  try {
    const raw = await callOpenAI(DOC_OVERVIEW_SYSTEM_PROMPT, userContent);
    const pairs = parsePairs(raw);
    console.log(`[OpenAI] Document overview → ${pairs.length} overview pairs generated for "${pdfName}".`);
    return pairs;
  } catch (e: any) {
    console.error('[OpenAI] generateDocumentOverviewPairs failed:', e?.message);
    return [];
  }
}
