import { indexApprovedPairs, loadStore } from './src/services/faq-store.service';

async function main() {
  console.log('Loading store...');
  loadStore();
  console.log('Indexing approved pairs...');
  const count = await indexApprovedPairs();
  console.log(`Successfully indexed ${count} pairs.`);
}

main().catch(console.error);
