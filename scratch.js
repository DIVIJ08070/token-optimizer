const { pipeline, env } = require('@xenova/transformers');
env.allowLocalModels = false;

async function run() {
  console.log("Loading model...");
  const generator = await pipeline('text-generation', 'Xenova/Qwen1.5-0.5B-Chat');
  const prompt = `<|im_start|>system\nYou are a helpful AI assistant.<|im_end|>\n<|im_start|>user\nContext: test\nQuestion: test<|im_end|>\n<|im_start|>assistant\n`;
  console.log("Generating...");
  const result = await generator(prompt, { max_new_tokens: 10, temperature: 0.1, return_full_text: false });
  console.log("Result:", JSON.stringify(result));
}
run();
