import fs from 'fs';
import path from 'path';
import { pipeline, env } from '@xenova/transformers';

env.allowLocalModels = false;

const EVAL_FILE = path.join(process.cwd(), 'eval-questions.json');
const STORE_FILE = path.join(process.cwd(), 'data', 'faq-store.json');

// From queryClassifier.ts
const GREETING_EXAMPLES = [
  "hi", "hello", "hey", "good morning", "good evening", "how are you", "what's up",
  "namaste", "kem cho", "kaise ho", "hi there",
];
const LEAD_EXAMPLES = [
  "I want to build an app", "i need a website for my business", "looking for a developer",
  "quote for a project", "can you make something for my business", "mane ek mobile app joiye che",
  "mujhe ek website banwani hai", "how much does it cost to build an app?",
  "app ketla price ma banse", "kitna kharcha aayega app ka", "what's your pricing",
  "kitne paise lagenge ek app me", "mare app banavavi chhe", "website banavava mate shu kharch aavse",
  "mare app book karavu che", "software project mate developer joiye che",
];
const CLARIFY_EXAMPLES = [
  "mane nathi samaj pdti", "I didn't get that", "what do you mean", "could you explain",
];

function cosine(a: number[], b: number[]): number {
  let dot = 0, na = 0, nb = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i];
  }
  return (na === 0 || nb === 0) ? 0 : dot / (Math.sqrt(na) * Math.sqrt(nb));
}

async function main() {
  const storeData = JSON.parse(fs.readFileSync(STORE_FILE, 'utf-8'));
  const evalData = JSON.parse(fs.readFileSync(EVAL_FILE, 'utf-8'));
  const pairs = storeData.pairs.filter((p: any) => p.status === 'approved' && p.question_embedding);
  const questions = evalData.questions;

  console.log(`Loaded ${pairs.length} indexed pairs and ${questions.length} eval questions.`);
  console.log('Loading multilingual model...');
  const extractor = await pipeline('feature-extraction', 'Xenova/paraphrase-multilingual-MiniLM-L12-v2');

  const embed = async (text: string) => {
    const out = await extractor(text, { pooling: 'mean', normalize: true });
    return Array.from(out.data) as number[];
  };

  console.log('Embedding intent examples...');
  const greetingVectors = await Promise.all(GREETING_EXAMPLES.map(embed));
  const leadVectors = await Promise.all(LEAD_EXAMPLES.map(embed));
  const clarifyVectors = await Promise.all(CLARIFY_EXAMPLES.map(embed));

  console.log('Embedding eval questions...');
  const evalRecords = [];
  for (const q of questions) {
    const qVec = await embed(q.question);
    
    // Compute max intent similarities
    const greetSim = Math.max(...greetingVectors.map(v => cosine(qVec, v)));
    const leadSim = Math.max(...leadVectors.map(v => cosine(qVec, v)));
    const clarifySim = Math.max(...clarifyVectors.map(v => cosine(qVec, v)));
    
    // Compute max FAQ similarity
    let bestFaqScore = -1;
    for (const pair of pairs) {
      const qs = cosine(qVec, pair.question_embedding);
      let rs = -1;
      if (pair.rephrasing_embeddings) {
        rs = Math.max(...pair.rephrasing_embeddings.map((v: number[]) => cosine(qVec, v)));
      }
      bestFaqScore = Math.max(bestFaqScore, qs, rs);
    }

    evalRecords.push({
      ...q,
      greetSim, leadSim, clarifySim, bestFaqScore
    });
  }

  console.log('\nRunning sweep...');
  
  let bestScore = -1;
  let bestParams = null;

  // Sweep Intent Threshold
  for (let intentT = 0.65; intentT <= 0.85; intentT += 0.05) {
    // Sweep FAQ High Threshold
    for (let faqHigh = 0.55; faqHigh <= 0.80; faqHigh += 0.05) {
      // Sweep FAQ Low Threshold
      for (let faqLow = 0.40; faqLow <= faqHigh; faqLow += 0.05) {
        
        let correct = 0;
        
        for (const record of evalRecords) {
          let predictedBehavior = '';
          
          if (record.category === 'mid_conversation_lead') {
            predictedBehavior = 'lead_captured';
          } else {
            // Classifier Logic
            const intentScores = {
              greeting: record.greetSim,
              lead: record.leadSim,
              clarify: record.clarifySim
            };
            const maxIntentScore = Math.max(intentScores.greeting, intentScores.lead, intentScores.clarify);
            let intent = 'none';
            if (maxIntentScore >= intentT) {
              if (maxIntentScore === intentScores.greeting) intent = 'greeting';
              else if (maxIntentScore === intentScores.lead) intent = 'lead';
              else if (maxIntentScore === intentScores.clarify) intent = 'clarify';
            }

            if (intent === 'lead') predictedBehavior = 'routed_to_lead';
            else if (intent === 'greeting') predictedBehavior = 'greeting';
            else if (intent === 'clarify') predictedBehavior = 'clarify';
            else {
              // FAQ Logic
              if (record.bestFaqScore >= faqHigh) predictedBehavior = 'answered';
              else if (record.bestFaqScore >= faqLow) predictedBehavior = 'answered'; // Tier 2 generated
              else predictedBehavior = 'refused';
            }
          }
          
          // Match logic
          let isPass = false;
          if (record.category === 'mid_conversation_lead') isPass = (predictedBehavior === 'lead_captured');
          else if (record.expected_route === 'lead') isPass = (predictedBehavior === 'routed_to_lead');
          else if (record.expected_route === 'greeting' || record.category === 'greeting') isPass = (predictedBehavior === 'greeting');
          else if (record.expected_band === 'refuse') isPass = (predictedBehavior === 'refused');
          else if (record.expected_band === 'answer') isPass = (predictedBehavior === 'answered');
          
          if (isPass) correct++;
        }
        
        if (correct > bestScore) {
          bestScore = correct;
          bestParams = { intentT, faqHigh, faqLow };
        }
      }
    }
  }

  console.log(`\n🏆 BEST PARAMS FOUND:`);
  if (!bestParams) {
    console.log(`   No suitable params found.`);
    return;
  }
  
  console.log(`   Intent Threshold : ${bestParams.intentT.toFixed(2)}`);
  console.log(`   FAQ High (Tier 1): ${bestParams.faqHigh.toFixed(2)}`);
  console.log(`   FAQ Low (Tier 2) : ${bestParams.faqLow.toFixed(2)}`);
  console.log(`   Accuracy         : ${bestScore} / ${questions.length} (${((bestScore/questions.length)*100).toFixed(1)}%)`);

  // Print failures at best params
  console.log('\nFailures at best params:');
  let failCount = 0;
  for (const record of evalRecords) {
    let predictedBehavior = '';
    
    if (record.category === 'mid_conversation_lead') {
      predictedBehavior = 'lead_captured';
    } else {
      const intentScores = {
        greeting: record.greetSim,
        lead: record.leadSim,
        clarify: record.clarifySim
      };
      const maxIntentScore = Math.max(intentScores.greeting, intentScores.lead, intentScores.clarify);
      let intent = 'none';
      if (maxIntentScore >= bestParams.intentT) {
        if (maxIntentScore === intentScores.greeting) intent = 'greeting';
        else if (maxIntentScore === intentScores.lead) intent = 'lead';
        else if (maxIntentScore === intentScores.clarify) intent = 'clarify';
      }

      if (intent === 'lead') predictedBehavior = 'routed_to_lead';
      else if (intent === 'greeting') predictedBehavior = 'greeting';
      else if (intent === 'clarify') predictedBehavior = 'clarify';
      else {
        if (record.bestFaqScore >= bestParams.faqHigh) predictedBehavior = 'answered';
        else if (record.bestFaqScore >= bestParams.faqLow) predictedBehavior = 'answered';
        else predictedBehavior = 'refused';
      }
    }
    
    let isPass = false;
    if (record.category === 'mid_conversation_lead') isPass = (predictedBehavior === 'lead_captured');
    else if (record.expected_route === 'lead') isPass = (predictedBehavior === 'routed_to_lead');
    else if (record.expected_route === 'greeting' || record.category === 'greeting') isPass = (predictedBehavior === 'greeting');
    else if (record.expected_band === 'refuse') isPass = (predictedBehavior === 'refused');
    else if (record.expected_band === 'answer') isPass = (predictedBehavior === 'answered');
    
    if (!isPass) {
      failCount++;
      console.log(` ❌ [${record.question}]`);
      console.log(`    Expected: ${record.expected_route || record.expected_band} | Got: ${predictedBehavior}`);
      console.log(`    Scores -> Greet: ${record.greetSim.toFixed(2)}, Lead: ${record.leadSim.toFixed(2)}, FAQ: ${record.bestFaqScore.toFixed(2)}`);
    }
  }
}

main().catch(console.error);
