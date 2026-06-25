import fs from 'fs';
import path from 'path';

const faqStorePath = path.join(process.cwd(), 'data', 'faq-store.json');
const vectorStorePath = path.join(process.cwd(), 'data', 'vector-store.json');

console.log('Clearing embeddings...');

// 1. Clear vector store
if (fs.existsSync(vectorStorePath)) {
  fs.unlinkSync(vectorStorePath);
  console.log(`Deleted ${vectorStorePath}`);
} else {
  console.log(`${vectorStorePath} not found, skipping.`);
}

// 2. Clear FAQ store embeddings
if (fs.existsSync(faqStorePath)) {
  try {
    const data = JSON.parse(fs.readFileSync(faqStorePath, 'utf-8'));
    if (data && data.pairs && Array.isArray(data.pairs)) {
      let clearedCount = 0;
      data.pairs.forEach((pair: any) => {
        if (pair.question_embedding || pair.rephrasing_embeddings) {
          clearedCount++;
        }
        delete pair.question_embedding;
        delete pair.rephrasing_embeddings;
        delete pair.question_tokens;
        delete pair.rephrasing_tokens;
      });
      fs.writeFileSync(faqStorePath, JSON.stringify(data, null, 2), 'utf-8');
      console.log(`Cleared embeddings for ${clearedCount} pairs in ${faqStorePath}.`);
    } else {
      console.log('Invalid format in faq-store.json.');
    }
  } catch (err) {
    console.error('Error modifying faq-store.json:', err);
  }
} else {
  console.log(`${faqStorePath} not found, skipping.`);
}

console.log('Done! Now click "Index Approved Pairs" in the UI to re-embed with the new model.');
