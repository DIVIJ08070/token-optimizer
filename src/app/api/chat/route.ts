/**
 * api/chat/route.ts  (refactored — Semantic FAQ chatbot)
 *
 * At chat time this route makes ZERO network calls and calls NO model.
 * Flow:
 *  1. Embed user message with local bge-small (in-process WASM, no network)
 *  2. Hybrid search (cosine + BM25 + RRF) over approved FAQ pairs
 *  3. Threshold gate:
 *       score >= THRESHOLD  → return stored approved answer
 *       score <  THRESHOLD  → return honest fallback
 */

import { NextRequest, NextResponse } from 'next/server';
import { hybridSearch, loadStore, addAndIndexPair, getEmbedder, savePairTranslation, type FaqPair } from '@/services/faq-store.service';
import { normalizeQuery } from '@/lib/queryNormalizer';
import { logMissedQuery } from '@/services/miss-logger.service';
import { initIntentClassifier, classifyIntent } from '@/lib/queryClassifier';
import { queryVectorStore } from '@/services/vector.service';
import { generateAnswer, translateAnswer, translateToEnglish } from '@/services/chat.service';
import { detectLang, langName, msg, type Lang } from '@/lib/i18n';
import { v4 as uuidv4 } from 'uuid';

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

const THRESHOLD_HIGH = 0.55; // Strong semantic match (Sweep optimized)
const THRESHOLD_LOW  = 0.35; // Weak semantic match (Lowered safely due to gibberish filter)
// Only offer a mid-tier "did you mean?" when the competing matches are near-
// high. Below this, a weak tie is unhelpful — generate a grounded answer instead.
const THRESHOLD_DISAMBIG = 0.50;

// Friendly, user-facing scope name. Override per deployment via ASSISTANT_SCOPE
// in .env.local. Never show raw PDF filenames to end users.
const ASSISTANT_SCOPE = process.env.ASSISTANT_SCOPE?.trim() || 'Palm Infotech';

// Turn a raw source filename (e.g. "9af3-Palm_Infotech_Overview.pdf") into a
// clean label for display. Falls back to the configured scope name.
function prettifySource(raw?: string): string {
  if (!raw) return ASSISTANT_SCOPE;
  const cleaned = raw
    .replace(/^[0-9a-f]{8}-[0-9a-f-]{27,}-/i, '') // strip leading uuid- prefix
    .replace(/\.[a-z0-9]+$/i, '')                  // strip extension
    .replace(/[_-]+/g, ' ')                          // underscores/dashes → spaces
    .replace(/\s+/g, ' ')
    .trim();
  return cleaned || ASSISTANT_SCOPE;
}

function isGrounded(quote: string, chunkText: string): boolean {
  if (!quote || quote.trim() === '') return false;
  const normalizedQuote = quote.replace(/\s+/g, ' ').trim();
  const normalizedText = chunkText.replace(/\s+/g, ' ');
  return normalizedText.includes(normalizedQuote);
}

function isGibberish(text: string): boolean {
  // 1. Check for 5 or more repeating characters (e.g. "aaaaa")
  if (/(.)\1{4,}/.test(text)) return true;
  
  // 2. Check for mostly symbols / extremely low alphanumeric ratio
  const alphaNumCount = (text.match(/[a-zA-Z0-9]/g) || []).length;
  if (text.length > 3 && alphaNumCount / text.length < 0.3) return true;
  
  // 3. Check for very long words without vowels (10+ chars)
  const words = text.split(/\s+/);
  for (const word of words) {
    if (word.length > 10 && !/[aeiouyAEIOUY]/.test(word)) return true;
  }
  
  return false;
}

// ---------------------------------------------------------------------------
// POST handler
// ---------------------------------------------------------------------------

export async function POST(req: NextRequest) {
  try {
    loadStore();

    const body = await req.json();
    let { question, debug, chatState } = body;

    if (Array.isArray(question)) question = question[0];
    if (typeof question !== 'string') question = String(question ?? '');

    if (!question.trim()) {
      return NextResponse.json({ error: 'Question is required' }, { status: 400 });
    }

    const isDebug = Boolean(debug);

    // Detect the user's language (local, free). All replies are returned in it.
    const lang: Lang = detectLang(question);

    console.log(`[Chat] Raw: "${question.slice(0, 50)}" | Lang: ${lang}`);

    // 0.1 Gibberish Pre-Filter
    if (isGibberish(question)) {
      console.log(`[Chat] Intercepted gibberish input: "${question}"`);
      const payload: any = {
        answer: msg('offTopic', lang, ASSISTANT_SCOPE),
        sources: [],
        suggestions: [],
        isFallback: true
      };
      if (isDebug) payload.pathUsed = 'gibberish-filter';
      return NextResponse.json(payload);
    }

    // 0.2 Conversational State Intercept
    let forceIdleNextState = false;

    if (chatState === 'awaiting_lead') {
      // Check if the input looks like contact info (at least 7 digits OR an @ sign)
      const hasPhone = /\d{7,}/.test(question);
      const hasEmail = /@\S+\.\S+/.test(question);

      if (hasPhone || hasEmail) {
        console.log(`[Chat] Intercepted message as lead details: "${question}"`);
        try {
          const fs = require('fs');
          const path = require('path');
          const leadsFile = path.join(process.cwd(), 'data', 'leads.json');
          const leads = fs.existsSync(leadsFile) ? JSON.parse(fs.readFileSync(leadsFile, 'utf8')) : [];
          leads.push({ details: question, timestamp: new Date().toISOString() });
          fs.writeFileSync(leadsFile, JSON.stringify(leads, null, 2), 'utf8');
        } catch (e) {
          console.error('[Chat] Failed to save lead:', e);
        }
        
        return NextResponse.json({
          answer: msg('leadThanks', lang),
          nextState: 'idle'
        });
      } else {
        console.log(`[Chat] User ignored lead prompt and asked a question. Breaking out of lead state.`);
        forceIdleNextState = true;
      }
    }

    // Helper to ensure nextState is returned if we broke out of lead capture
    const sendJson = (data: any, init?: any) => {
      if (forceIdleNextState && !data.nextState) {
        data.nextState = 'idle';
      }
      return NextResponse.json(data, init);
    };

    // Return a stored answer in the user's language. English → unchanged (free).
    // Otherwise serve a cached translation, or translate once (cheap 8B model)
    // and cache it on the pair so every future hit is free.
    // Match the user's script: romanized input → romanized reply.
    const script: 'roman' | 'native' = /[઀-૿ऀ-ॿ]/.test(question) ? 'native' : 'roman';
    const localizeAnswer = async (answer: string, pair?: FaqPair): Promise<string> => {
      if (lang === 'en') return answer;
      const cacheKey = `${lang}:${script}`;
      const cached = pair?.answer_i18n?.[cacheKey];
      if (cached) return cached;
      const translated = await translateAnswer(answer, langName(lang) as 'Hindi' | 'Gujarati', question, script);
      if (pair) savePairTranslation(pair.id, cacheKey, translated);
      return translated;
    };

    // 1. Initialize embedder + classifier.
    const embedder = getEmbedder();
    await initIntentClassifier(embedder);

    // Intent (greeting/lead/off_topic/clarify) is classified on the ORIGINAL-
    // language query — the classifier's examples are multilingual ("namaste",
    // "kem cho", romanized leads), so translating first would break them.
    const intentEmbedding = await embedder.embedQuery(normalizeQuery(question));

    // Greetings are unambiguous and must short-circuit BEFORE the FAQ search
    // (and before paying for translation). lead / off_topic / clarify are
    // deferred until AFTER search so they can't hijack a real FAQ question.
    if (classifyIntent(intentEmbedding) === 'greeting') {
      return sendJson({ answer: msg('greeting', lang), isGreeting: true });
    }

    // Translate non-English queries to English for RETRIEVAL only. The FAQ store
    // and local embedder are English-first, so embedding a romanized Gujarati/
    // Hindi query directly gives inaccurate matches (it latches onto shared
    // words like "app"). We search in English, then reply in the user's
    // language. English queries skip this and stay 100% free.
    const searchQuestion = lang === 'en' ? question : await translateToEnglish(question);
    if (lang !== 'en') console.log(`[Chat] Translated for search: "${searchQuestion.slice(0, 60)}"`);
    const normalizedQ = normalizeQuery(searchQuestion);
    const queryEmbedding = lang === 'en'
      ? intentEmbedding
      : await embedder.embedQuery(normalizedQ);

    // 2. Hybrid Search FIRST.
    //    A strong FAQ match must always win over the fuzzy intent classifier —
    //    otherwise a real question like "What is the Wonder App about?" gets
    //    hijacked into lead capture simply because it contains the word "app".
    //    Intent (greeting / lead / off_topic / clarify) is only consulted below,
    //    AFTER we know there is no high-confidence stored answer.
    const results = await hybridSearch(normalizedQ, 5, queryEmbedding);

    if (results.length === 0) {
      console.log('[Chat] Store is empty.');
      const payload: any = { answer: msg('noAnswer', lang), sources: [], suggestions: [], isFallback: true };
      if (isDebug) payload.pathUsed = 'faq-semantic-empty';
      return sendJson(payload);
    }

    const top = results[0];

    console.log(
      `[Chat] Top match vecScore=${top.vecScore.toFixed(4)} (Hybrid=${top.score.toFixed(4)}), ` +
      `HIGH=${THRESHOLD_HIGH}, LOW=${THRESHOLD_LOW}, pair="${top.pair.question.slice(0, 40)}..."`
    );

    // ------------------------------------------------------------------
    // Disambiguation Logic
    // ------------------------------------------------------------------
    const TIE_MARGIN = 0.05;
    let topMatches = results.filter(r => r.vecScore >= top.vecScore - TIE_MARGIN);

    // 1. Exact Match Bypass (prevents looping when user clicks a suggestion)
    const isExactMatch = normalizedQ === normalizeQuery(top.pair.question);

    // 2. Answer Deduplication (prevents asking to choose between identical answers)
    const uniqueMatches: typeof topMatches = [];
    const seenAnswers = new Set<string>();
    for (const m of topMatches) {
      if (!seenAnswers.has(m.pair.answer)) {
        seenAnswers.add(m.pair.answer);
        uniqueMatches.push(m);
      }
    }
    topMatches = uniqueMatches;

    // ------------------------------------------------------------------
    // TIER 1: High Confidence
    // ------------------------------------------------------------------
    if (top.vecScore >= THRESHOLD_HIGH) {
      // Check for a true tie among HIGH scorers
      if (!isExactMatch && topMatches.length > 1 && topMatches[1].vecScore >= THRESHOLD_HIGH) {
        return sendJson({
          isDidYouMean: true,
          answer: msg('didYouMean', lang),
          suggestions: topMatches.map(r => r.pair.question),
        });
      }

      const payload: any = {
        answer: await localizeAnswer(top.pair.answer, top.pair),
        sources: [{ pdf: top.pair.source, chunkRef: top.pair.chunk_ref }],
        matchedQuestion: top.pair.question,
      };

      if (isDebug) {
        payload.pathUsed = 'faq-semantic-high';
        payload.score = top.score;
        payload.retrievedSources = results.map(r => ({ question: r.pair.question, score: r.score }));
      }
      return sendJson(payload);
    }

    // ------------------------------------------------------------------
    // Intent classification — ONLY now that we know there is no
    // high-confidence stored answer. This keeps real FAQ questions from
    // being hijacked by the fuzzy lead/off-topic classifier.
    // ------------------------------------------------------------------
    const intent = classifyIntent(intentEmbedding);

    if (intent === 'lead') {
      return sendJson({
        answer: msg('lead', lang),
        isLeadCapture: true,
        nextState: 'awaiting_lead'
      });
    }

    if (intent === 'off_topic') {
      console.log(`[Chat] Intent classified as off_topic.`);
      const payload: any = {
        answer: msg('offTopic', lang, ASSISTANT_SCOPE),
        sources: [],
        suggestions: [],
        isFallback: true
      };
      if (isDebug) payload.pathUsed = 'intent-off-topic';
      return sendJson(payload);
    }

    if (intent === 'clarify') {
      const payload: any = {
        answer: msg('clarify', lang),
        sources: [],
        suggestions: [],
        isFallback: true
      };
      if (isDebug) payload.pathUsed = 'intent-clarify';
      return sendJson(payload);
    }

    // Log the miss since we fell below the HIGH threshold
    logMissedQuery(question, top.pair.question, top.score);

    // ------------------------------------------------------------------
    // TIER 2: Mid Confidence (Fallback Generation)
    // ------------------------------------------------------------------
    if (top.vecScore >= THRESHOLD_LOW) {
      // Only disambiguate on a genuinely strong, near-high tie. A weak tie
      // (e.g. a generic "what services do you provide") is better served by
      // generating a grounded answer than by asking "did you mean?".
      if (
        !isExactMatch &&
        topMatches.length > 1 &&
        top.vecScore >= THRESHOLD_DISAMBIG &&
        topMatches[1].vecScore >= THRESHOLD_DISAMBIG
      ) {
        return sendJson({
          isDidYouMean: true,
          answer: msg('notSure', lang),
          suggestions: topMatches.map(r => r.pair.question),
        });
      }

      console.log(`[Chat] Score in mid-tier → retrieving chunks for fallback generation.`);
      
      const chunks = await queryVectorStore(normalizedQ, 3);
      if (chunks.length > 0) {
        console.log(`[Chat] Calling API to generate answer from ${chunks.length} chunks.`);
        const { answer, grounded_quote, evidence, rephrasings } = await generateAnswer(chunks, searchQuestion);

        const combinedChunkText = chunks.map(c => c.pageContent).join('\n');

        if (isGrounded(grounded_quote, combinedChunkText)) {
          console.log('[Chat] Generated fallback passed grounding check → caching + indexing as a free, reusable answer.');

          const newPair: FaqPair = {
            id: uuidv4(),
            // Store the English question for accurate future matching; keep the
            // user's original phrasing as a rephrasing so it also matches directly.
            question: searchQuestion,
            rephrasings: [question, ...(rephrasings || [])].filter((r, i, a) => a.indexOf(r) === i),
            answer: answer,
            source: chunks[0].metadata.pdfName || 'unknown',
            chunk_ref: chunks[0].metadata.chunkIndex !== undefined ? String(chunks[0].metadata.chunkIndex) : 'fallback',
            grounded_quote: grounded_quote,
            status: 'approved',
            isAutoGenerated: true
          };
          // Self-learning cache: embed locally + make it live NOW, so the next
          // time this (or a similar) question is asked it is answered for free.
          await addAndIndexPair(newPair);

          const payload: any = {
            // Translate into the user's language (and cache it on the pair so
            // future hits are free). English is returned unchanged.
            answer: await localizeAnswer(answer, newPair),
            sources: evidence.map((ev: any) => ({ chunkRef: ev.chunkId })),
            matchedQuestion: question,
            apiCalled: true,
            pairId: newPair.id,
          };
          
          if (isDebug) {
            payload.pathUsed = 'faq-semantic-mid-generated';
            payload.score = top.score;
          }
          return sendJson(payload);
        } else {
          console.warn('[Chat] Generated fallback failed grounding check. Falling through to refusal.');
        }
      } else {
        console.warn('[Chat] No chunks found in vector store for fallback generation.');
      }
    }

    // ------------------------------------------------------------------
    // TIER 3: Low Confidence (Refusal)
    // ------------------------------------------------------------------
    console.log(`[Chat] Score below LOW → fallback refusal.`);
    const docContext = prettifySource(results[0]?.pair.source) || ASSISTANT_SCOPE;
    const suggestions = results.slice(0, 3).map(r => r.pair.question);

    const payload: any = {
      answer: msg('fallbackIntro', lang, docContext),
      sources: [],
      suggestions,
      isFallback: true
    };

    if (isDebug) {
      payload.pathUsed  = 'faq-semantic-fallback';
      payload.score     = top.score;
    }

    return sendJson(payload);

  } catch (error: any) {
    console.error('[Chat] Error:', error);
    // sendJson is scoped inside the try block — use NextResponse directly here.
    return NextResponse.json({ error: error.message }, { status: 500 });
  }
}
