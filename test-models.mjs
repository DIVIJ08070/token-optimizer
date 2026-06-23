import { AutoTokenizer, AutoModelForSequenceClassification, env } from '@xenova/transformers';

async function test() {
  env.allowLocalModels = false;
  try {
    const tokenizer = await AutoTokenizer.from_pretrained('Xenova/bge-reranker-base');
    const model = await AutoModelForSequenceClassification.from_pretrained('Xenova/bge-reranker-base');
    
    const inputs1 = tokenizer('How many players in a team?', { text_pair: 'A side consists of 11 players.', padding: true, truncation: true });
    const { logits: logits1 } = await model(inputs1);
    console.log('logits1', logits1.data);

    const inputs2 = tokenizer('How many players in a team?', { text_pair: 'The umpire must check the stumps.', padding: true, truncation: true });
    const { logits: logits2 } = await model(inputs2);
    console.log('logits2', logits2.data);
  } catch (e) {
    console.error('Failed:', e);
  }
}

test();
