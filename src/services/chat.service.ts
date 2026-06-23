import { pipeline, env } from '@xenova/transformers';

// Configure transformers.js
env.allowLocalModels = false;

let generator: any = null;

export async function generateAnswer(contextChunks: any[], question: string) {
  if (!generator) {
    console.log("[Backend] Loading Local LLM Model (this happens once)...");
    // Upgrading to a modern Decoder-only chat model (0.5 Billion parameters).
    // This model actually understands logic, reasoning, and bullet point formatting!
    generator = await pipeline('text-generation', 'Xenova/Qwen1.5-0.5B-Chat');
  }

  // Combine chunk texts, separated by double newlines for clear boundaries
  const context = contextChunks.map((c: any) => c.pageContent).join('\n\n');

  // Qwen uses ChatML formatting for highly accurate instruction following
  const prompt = `<|im_start|>system\nYou are an expert assistant. Answer ONLY using the provided context. If the answer is not explicitly stated, say:\n"I couldn't find this information in the documents."\nDo not infer, summarize unrelated sections, or combine information from different laws.\nCite the law number used.<|im_end|>\n<|im_start|>user\nContext:\n${context}\n\nQuestion: ${question}<|im_end|>\n<|im_start|>assistant\n`;

  console.log("[Backend] ---------- LLM CONTEXT DUMP ----------");
  console.log(context);
  console.log("[Backend] --------------------------------------");
  console.log(`[Backend] Question asked: "${question}"`);

  const result = await generator(prompt, {
    max_new_tokens: 350,
    temperature: 0.1,
    return_full_text: false, // This ensures it only returns the newly generated answer!
    callback_function: (output: any) => {
      // Print a dot to the console for every token generated so the user knows it's actively working!
      process.stdout.write(".");
    }
  });

  process.stdout.write("\n"); // New line after dots finish

  let answer = result[0].generated_text.trim();
  console.log("[Backend] LLM Generated Answer:", answer);

  // Failsafe: if the tiny local model fails to understand the prompt and outputs nothing, 
  // we will just show the user the raw extracted text so they still get their answer!
  if (!answer || answer.length < 5) {
    answer = "The local AI model had trouble summarizing this, but here is the exact text found in your PDFs:\n\n" + context;
  }

  // Extract sources
  const sources = contextChunks.map(c => ({
    pdf: c.metadata.pdfName,
    law: c.metadata.law,
    page: c.metadata.pageNumber
  }));

  // Filter unique sources
  const uniqueSources = Array.from(new Set(sources.map(s => JSON.stringify(s)))).map(s => JSON.parse(s as string));

  return {
    answer,
    sources: uniqueSources
  };
}
