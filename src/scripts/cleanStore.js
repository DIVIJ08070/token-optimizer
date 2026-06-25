const fs = require('fs');
const path = require('path');

const storePath = path.join(process.cwd(), 'data', 'faq-store.json');
const data = JSON.parse(fs.readFileSync(storePath, 'utf8'));

// 1. Deduplicate 120+ apps
const appsPairs = data.pairs.filter(p => p.answer.includes('120') && p.question !== 'What are the key points of the Palm Infotech Overview document?');
if (appsPairs.length > 0) {
  const mainPair = appsPairs[0];
  console.log(`Main 120+ apps pair: ${mainPair.question}`);
  
  for (let i = 1; i < appsPairs.length; i++) {
    const dup = appsPairs[i];
    console.log(`Merging dup: ${dup.question}`);
    if (!mainPair.rephrasings.includes(dup.question)) {
      mainPair.rephrasings.push(dup.question);
    }
    for (const rep of dup.rephrasings) {
      if (!mainPair.rephrasings.includes(rep)) {
        mainPair.rephrasings.push(rep);
      }
    }
    // Remove the dup from data.pairs
    data.pairs = data.pairs.filter(p => p.id !== dup.id);
  }
  // Clear embeddings to force re-indexing
  delete mainPair.question_embedding;
  delete mainPair.rephrasing_embeddings;
  delete mainPair.question_tokens;
  delete mainPair.rephrasing_tokens;
  mainPair.indexed = false;
}

// 2. Add services rephrasings
const servicesPair = data.pairs.find(p => p.question === 'What does Palm Infotech do?' && p.answer.includes('listening to ideas'));
if (servicesPair) {
  console.log(`Found services pair: ${servicesPair.question}`);
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
    if (!servicesPair.rephrasings.includes(rep)) {
      servicesPair.rephrasings.push(rep);
    }
  }
  delete servicesPair.question_embedding;
  delete servicesPair.rephrasing_embeddings;
  delete servicesPair.question_tokens;
  delete servicesPair.rephrasing_tokens;
  servicesPair.indexed = false;
}

fs.writeFileSync(storePath, JSON.stringify(data, null, 2));
console.log('Cleaned and saved to faq-store.json');
