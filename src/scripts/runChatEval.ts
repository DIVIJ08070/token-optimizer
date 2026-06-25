import fs from 'fs';
import path from 'path';

const EVAL_FILE = path.join(process.cwd(), 'eval-questions.json');

async function main() {
  if (!fs.existsSync(EVAL_FILE)) {
    console.error('Eval file not found at:', EVAL_FILE);
    process.exit(1);
  }

  const data = JSON.parse(fs.readFileSync(EVAL_FILE, 'utf-8'));
  const questions = data.questions;
  
  if (!questions || !Array.isArray(questions)) {
    console.error('No questions array found in eval file.');
    process.exit(1);
  }

  let passed = 0;
  let failed = 0;

  console.log(`\nStarting evaluation of ${questions.length} questions...\n`);
  
  for (const q of questions) {
    // Print with fixed width
    process.stdout.write(`Testing: "${q.question.slice(0, 45).padEnd(45)}" ... `);
    
    let chatState = 'idle';
    if (q.category === 'mid_conversation_lead') {
      chatState = 'awaiting_lead';
    }
    
    try {
      const res = await fetch('http://localhost:3000/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: q.question, chatState, debug: true })
      });
      const json = await res.json();
      
      let isPass = false;
      let actualBehavior = '';
      
      if (q.category === 'mid_conversation_lead') {
        isPass = json.nextState === 'idle';
        actualBehavior = json.nextState === 'idle' ? 'lead_captured' : (json.isFallback ? 'refused' : 'faq_matched');
      } else if (q.expected_route === 'lead') {
        isPass = !!json.isLeadCapture;
        actualBehavior = json.isLeadCapture ? 'routed_to_lead' : (json.isFallback ? 'refused' : 'faq_matched');
      } else if (q.expected_route === 'greeting' || q.category === 'greeting') {
        isPass = !!json.isGreeting;
        actualBehavior = json.isGreeting ? 'greeting' : (json.isFallback ? 'refused' : 'faq_matched');
      } else if (q.expected_band === 'refuse' || q.expected_band === 'meta' || q.expected_band === 'clarify' || q.category === 'confusion' || q.category === 'abuse_or_meta') {
        isPass = !!json.isFallback;
        actualBehavior = json.isFallback ? 'refused' : (json.isLeadCapture ? 'lead' : 'force_matched_faq');
      } else if (q.expected_band === 'answer' || q.expected_route === 'faq_or_band2' || q.expected_band === 'faq_or_band2') {
        isPass = !!json.answer && !json.isFallback && !json.isLeadCapture && !json.isGreeting;
        actualBehavior = isPass ? 'answered' : (json.isFallback ? 'refused' : 'other');
      }
      
      if (isPass) {
        console.log('✅ PASS');
        passed++;
      } else {
        console.log(`❌ FAIL`);
        console.log(`   └─ Expected: ${q.expected_route || q.expected_band}`);
        console.log(`   └─ Actual  : ${actualBehavior}`);
        if (json.score !== undefined) {
          console.log(`   └─ Score   : ${json.score.toFixed(4)}`);
        }
        console.log(`   └─ Response: ${json.answer?.slice(0, 80).replace(/\n/g, ' ')}...`);
        failed++;
      }
    } catch (e: any) {
      console.log(`❌ ERROR: ${e.message}`);
      failed++;
    }
  }
  
  const total = passed + failed;
  const pct = total > 0 ? ((passed / total) * 100).toFixed(1) : '0.0';
  console.log(`\n\n🏆 Evaluation Complete: ${passed}/${total} Passed (${pct}%)`);
}

main().catch(console.error);
