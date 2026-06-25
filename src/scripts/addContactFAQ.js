const fs = require('fs');
const crypto = require('crypto');

const storePath = './data/faq-store.json';
const data = JSON.parse(fs.readFileSync(storePath, 'utf8'));

const newPair = {
  id: crypto.randomUUID(),
  question: "How do I contact Palm Infotech?",
  answer: "You can contact us via our website www.palminfotech.com, email contact@palminfotech.com, or call +91 8320118036.",
  rephrasings: [
    "What is your contact number?",
    "contact details",
    "phone number",
    // Multilingual
    "contact number su che",
    "tamarro number aapo",
    "aapka phone number kya hai",
    "mujhe aapse baat karni hai contact details dijiye",
    "contact karva mate number su che",
    "call karna hai",
    "number moklo"
  ],
  source: "Palm_Infotech_Overview_Plain_Text.pdf",
  chunk_ref: "0",
  grounded_quote: "www.palminfotech.com | contact@palminfotech.com | +91 8320118036",
  status: "approved"
};

data.pairs.push(newPair);

// clear embeddings so it re-indexes
data.pairs.forEach(p => {
  delete p.question_embedding;
  delete p.rephrasing_embeddings;
  delete p.question_tokens;
  delete p.rephrasing_tokens;
});

fs.writeFileSync(storePath, JSON.stringify(data, null, 2), 'utf8');

// Also delete vector store
const vectorStorePath = './data/vector-store.json';
if (fs.existsSync(vectorStorePath)) {
  fs.unlinkSync(vectorStorePath);
}

console.log('Added contact FAQ and cleared embeddings to force re-index.');
