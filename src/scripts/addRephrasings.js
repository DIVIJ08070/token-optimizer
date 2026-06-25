const fs = require('fs');

const storePath = './data/faq-store.json';
const data = JSON.parse(fs.readFileSync(storePath, 'utf8'));

// 1. How big is the team?
const teamPair = data.pairs.find(p => p.question === 'How big is the team?');
if (teamPair) {
  teamPair.rephrasings.push("kitne log kaam karte hai team me");
}

// 2. What is the company's slogan?
const sloganPair = data.pairs.find(p => p.question === "What is the company's slogan?");
if (sloganPair) {
  sloganPair.rephrasings.push("slogan kya hai");
}

// Clear embeddings to force re-index
data.pairs.forEach(p => {
  delete p.question_embedding;
  delete p.rephrasing_embeddings;
  delete p.question_tokens;
  delete p.rephrasing_tokens;
});

fs.writeFileSync(storePath, JSON.stringify(data, null, 2), 'utf8');

const vectorStorePath = './data/vector-store.json';
if (fs.existsSync(vectorStorePath)) {
  fs.unlinkSync(vectorStorePath);
}

console.log('Added targeted rephrasings and cleared embeddings.');
