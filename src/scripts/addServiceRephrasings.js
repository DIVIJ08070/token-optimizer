const fs = require('fs');
const path = require('path');

const storePath = path.join(process.cwd(), 'data', 'faq-store.json');
if (!fs.existsSync(storePath)) {
  console.log('Store not found.');
  process.exit(1);
}

const data = JSON.parse(fs.readFileSync(storePath, 'utf8'));

let found = false;
for (const pair of data.pairs) {
  if (pair.question.toLowerCase().includes('what services does palm infotech provide') || 
      pair.question.toLowerCase().includes('what are the main services') ||
      pair.answer.toLowerCase().includes('mobile app development') && pair.answer.toLowerCase().includes('web application')) {
    
    console.log(`Found pair: ${pair.question}`);
    
    const newRephrasings = [
      "which services do you provide",
      "what do you do",
      "tme kai sevao apo cho",
      "tamari pase kai suvidhao uplabdh ase",
      "what kind of services",
      "service list",
      "what are your offerings"
    ];

    for (const rep of newRephrasings) {
      if (!pair.rephrasings.includes(rep)) {
        pair.rephrasings.push(rep);
      }
    }
    
    // Clear embeddings so it gets re-indexed
    delete pair.question_embedding;
    delete pair.rephrasing_embeddings;
    delete pair.question_tokens;
    delete pair.rephrasing_tokens;
    pair.indexed = false; // Add indexed flag reset if they have one

    console.log(`Added rephrasings. Total now: ${pair.rephrasings.length}`);
    found = true;
    break; // only do the main one
  }
}

if (!found) {
  console.log('Could not find the services pair.');
} else {
  fs.writeFileSync(storePath, JSON.stringify(data, null, 2));
  console.log('Saved to faq-store.json');
}
