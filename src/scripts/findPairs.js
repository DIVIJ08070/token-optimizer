const fs = require('fs');
const path = require('path');

const storePath = path.join(process.cwd(), 'data', 'faq-store.json');
const data = JSON.parse(fs.readFileSync(storePath, 'utf8'));

data.pairs.filter(p => p.question.toLowerCase().includes('help clients with') || p.question.toLowerCase().includes('services')).forEach(p => {
  console.log('ID:', p.id);
  console.log('Q:', p.question);
  console.log('A:', p.answer);
  console.log('Rephrasings:', p.rephrasings.join(' | '));
  console.log('---');
});
