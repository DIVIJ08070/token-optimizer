import { NextResponse } from 'next/server';
import { getAllPairs, saveStore, indexApprovedPairs } from '@/services/faq-store.service';

export async function GET() {
  const pairs = getAllPairs();
  let modified = 0;

  // 1. Remove duplicate "120 apps" pairs
  const appPairs = pairs.filter(p => p.answer.includes('120') && !p.question.includes('Overview document'));
  if (appPairs.length > 1) {
    const main = appPairs[0];
    for (let i = 1; i < appPairs.length; i++) {
      const dup = appPairs[i];
      if (!main.rephrasings.includes(dup.question)) main.rephrasings.push(dup.question);
      dup.rephrasings.forEach(r => { if (!main.rephrasings.includes(r)) main.rephrasings.push(r); });
      // Remove dup
      const idx = pairs.findIndex(p => p.id === dup.id);
      if (idx !== -1) {
        pairs.splice(idx, 1);
        modified++;
      }
    }
    delete main.question_embedding;
    modified++;
  }

  // 2. Fix the sales team pair stealing services
  const salesPair = pairs.find(p => p.question.toLowerCase().includes('help clients with'));
  if (salesPair) {
    salesPair.question = "What do you do for sales teams?";
    salesPair.rephrasings = salesPair.rephrasings.filter(r => 
      !r.toLowerCase().includes('what do you do') && 
      !r.toLowerCase().includes('services')
    );
    delete salesPair.question_embedding;
    modified++;
  }

  // 3. Strengthen main services pair
  const servicesPair = pairs.find(p => p.question === 'What does Palm Infotech do?' && p.answer.toLowerCase().includes('listening'));
  if (servicesPair) {
    const strongRephs = [
      "which services do you provide",
      "what do you offer",
      "what can you do for me",
      "what services are available",
      "what do you do",
      "tme kai sevao apo cho",
      "tamari pase kai suvidhao uplabdh ase",
      "what kind of services",
      "service list",
      "what are your offerings"
    ];
    strongRephs.forEach(r => {
      if (!servicesPair.rephrasings.includes(r)) servicesPair.rephrasings.push(r);
    });
    delete servicesPair.question_embedding;
    modified++;
  }

  if (modified > 0) {
    saveStore();
    await indexApprovedPairs();
  }

  return NextResponse.json({ success: true, modified, pairs: pairs.length });
}
