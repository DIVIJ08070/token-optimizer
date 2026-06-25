import { LocalTransformersEmbeddings } from '@/services/embedding.service';

// ---------------------------------------------------------------------------
// Intent Examples (Multilingual)
// ---------------------------------------------------------------------------

const GREETING_EXAMPLES = [
  "hi",
  "hello there",
  "hey",
  "good morning",
  "namaste",
  "kem cho",
  "hola",
  "greetings",
  "suprabhat"
];

const LEAD_EXAMPLES = [
  // English
  "I want to build an app",
  "I need a website created",
  "Looking for a software development team",
  "Can you give me a quote for a new project",
  // Hindi / Hinglish
  "mujhe ek app banwana hai",
  "website banwani hai",
  "humko software develop karwana hai",
  "project ka estimate chahiye",
  // Gujarati
  "mare app banavavi chhe",
  "software project mate developer joiye che",
  "app ni prise su che",
  "app ni prise",
  "app no kharcho",
  "ketla rupiya thase",
  // Pricing/Cost
  "how much does an app cost",
  "price for an app",
  "what is the cost",
  "price ketli thase"
];

const OFF_TOPIC_EXAMPLES = [
  // General Knowledge & Sports
  "aaj match kon jeeta",
  "what's the weather today?",
  "who is the prime minister of India?",
  "recommend me a good restaurant",
  "tell me a joke",
  "who won the world cup",
  "aaj barish hogi kya",
  // Abuse & Meta
  "are you dumb",
  "this service is useless",
  "are you a bot or a human?",
  "tu pagal hai kya",
  "gande kaam",
];

const CLARIFY_EXAMPLES = [
  "mane nathi samaj pdti",
  "I didn't get that",
  "what do you mean",
  "could you explain",
  "can you repeat that",
  "mujhe samajh nahi aaya"
];

// ---------------------------------------------------------------------------
// Cached Embeddings
// ---------------------------------------------------------------------------

let greetingEmbeddings: number[][] = [];
let leadEmbeddings: number[][] = [];
let offTopicEmbeddings: number[][] = [];
let clarifyEmbeddings: number[][] = [];
let isInitialized = false;

// Configurable threshold for semantic intent matching
const INTENT_THRESHOLD = 0.65;

export async function initIntentClassifier(embedder: LocalTransformersEmbeddings) {
  if (isInitialized) return;
  console.log('[Backend] Initializing Semantic Intent Classifier...');
  greetingEmbeddings = await embedder.embedDocuments(GREETING_EXAMPLES);
  leadEmbeddings = await embedder.embedDocuments(LEAD_EXAMPLES);
  offTopicEmbeddings = await embedder.embedDocuments(OFF_TOPIC_EXAMPLES);
  clarifyEmbeddings = await embedder.embedDocuments(CLARIFY_EXAMPLES);
  isInitialized = true;
  console.log('[Backend] Semantic Intent Classifier Ready.');
}

function cosineSimilarity(vecA: number[], vecB: number[]) {
  let dotProduct = 0;
  let normA = 0;
  let normB = 0;
  for (let i = 0; i < vecA.length; i++) {
    dotProduct += vecA[i] * vecB[i];
    normA += vecA[i] * vecA[i];
    normB += vecB[i] * vecB[i];
  }
  if (normA === 0 || normB === 0) return 0;
  return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
}

function getMaxSimilarity(queryVector: number[], intentVectors: number[][]): number {
  let maxSim = -1;
  for (const v of intentVectors) {
    const sim = cosineSimilarity(queryVector, v);
    if (sim > maxSim) maxSim = sim;
  }
  return maxSim;
}

export function classifyIntent(queryVector: number[]): 'greeting' | 'lead' | 'off_topic' | 'clarify' | null {
  if (!isInitialized) {
    console.warn('[Backend] classifyIntent called before initialization!');
    return null;
  }

  const greetingScore = getMaxSimilarity(queryVector, greetingEmbeddings);
  const leadScore = getMaxSimilarity(queryVector, leadEmbeddings);
  const offTopicScore = getMaxSimilarity(queryVector, offTopicEmbeddings);
  const clarifyScore = getMaxSimilarity(queryVector, clarifyEmbeddings);

  const maxScore = Math.max(greetingScore, leadScore, offTopicScore, clarifyScore);

  if (maxScore >= INTENT_THRESHOLD) {
    if (maxScore === offTopicScore) return 'off_topic';
    if (maxScore === clarifyScore) return 'clarify';
    if (maxScore === greetingScore) return 'greeting';
    if (maxScore === leadScore) return 'lead';
  }

  return null;
}

