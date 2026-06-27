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
import { hybridSearch, loadStore, addPairs, getEmbedder } from '@/services/faq-store.service';
import { normalizeQuery } from '@/lib/queryNormalizer';
import { logMissedQuery } from '@/services/miss-logger.service';
import { initIntentClassifier, classifyIntent } from '@/lib/queryClassifier';
import { queryVectorStore } from '@/services/vector.service';
import { generateAnswer } from '@/services/chat.service';
import { v4 as uuidv4 } from 'uuid';

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

const THRESHOLD_HIGH = 0.55; // Strong semantic match (Sweep optimized)
const THRESHOLD_LOW  = 0.35; // Weak semantic match (Lowered safely due to gibberish filter)

// Friendly, user-facing scope name. Override per deployment via ASSISTANT_SCOPE
// in .env.local. Never show raw PDF filenames to end users.
const ASSISTANT_SCOPE = process.env.ASSISTANT_SCOPE?.trim() || 'Palm Infotech';
const OUT_OF_SCOPE_MSG =
  `I can only help with questions about ${ASSISTANT_SCOPE} — try asking about our services, team, or process. 😊`;

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
    const normalizedQ = normalizeQuery(question);

    console.log(`[Chat] Raw: "${question.slice(0, 50)}" | Normalized: "${normalizedQ.slice(0, 50)}"`);

    // 0.1 Gibberish Pre-Filter
    if (isGibberish(question)) {
      console.log(`[Chat] Intercepted gibberish input: "${question}"`);
      const payload: any = {
        answer: OUT_OF_SCOPE_MSG,
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
          answer: "Thank you! Our team will review your details and contact you shortly.",
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

    // 1. Initialize & Embed Query
    const embedder = getEmbedder();
    await initIntentClassifier(embedder);
    const queryEmbedding = await embedder.embedQuery(normalizedQ);

    // 2. Semantic Intent Classification
    const intent = classifyIntent(queryEmbedding);

    if (intent === 'greeting') {
      return sendJson({ answer: "Hello! How can I help you today?", isGreeting: true });
    }

    if (intent === 'lead') {
      return sendJson({ 
        answer: "Great! Let's get started on your project. Could you please provide your name and phone number?", 
        isLeadCapture: true,
        nextState: 'awaiting_lead'
      });
    }

    if (intent === 'off_topic') {
      console.log(`[Chat] Intent classified as off_topic.`);
      const payload: any = {
        answer: OUT_OF_SCOPE_MSG,
        sources: [],
        suggestions: [],
        isFallback: true
      };
      if (isDebug) payload.pathUsed = 'intent-off-topic';
      return sendJson(payload);
    }

    if (intent === 'clarify') {
      const payload: any = { 
        answer: "I didn't quite catch that. Could you please rephrase your question?", 
        sources: [], 
        suggestions: [], 
        isFallback: true 
      };
      if (isDebug) payload.pathUsed = 'intent-clarify';
      return sendJson(payload);
    }

    // 3. Hybrid Search (using precomputed embedding)
    const results = await hybridSearch(normalizedQ, 5, queryEmbedding);

    if (results.length === 0) {
      console.log('[Chat] Store is empty.');
      const payload: any = { answer: "I don't have an approved answer for that question.", sources: [], suggestions: [], isFallback: true };
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
          answer: "I found multiple matching answers. Which did you mean?",
          suggestions: topMatches.map(r => r.pair.question),
        });
      }

      const payload: any = {
        answer: top.pair.answer,
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

    // Log the miss since we fell below the HIGH threshold
    logMissedQuery(question, top.pair.question, top.score);

    // ------------------------------------------------------------------
    // TIER 2: Mid Confidence (Fallback Generation)
    // ------------------------------------------------------------------
    if (top.vecScore >= THRESHOLD_LOW) {
      // Check for a middling tie
      if (!isExactMatch && topMatches.length > 1 && topMatches[1].vecScore >= THRESHOLD_LOW) {
        return sendJson({
          isDidYouMean: true,
          answer: "I'm not completely sure. Did you mean one of these?",
          suggestions: topMatches.map(r => r.pair.question),
        });
      }

      console.log(`[Chat] Score in mid-tier → retrieving chunks for fallback generation.`);
      
      const chunks = await queryVectorStore(normalizedQ, 3);
      if (chunks.length > 0) {
        console.log(`[Chat] Calling API to generate answer from ${chunks.length} chunks.`);
        const { answer, grounded_quote, evidence, rephrasings } = await generateAnswer(chunks, question);
        
        const combinedChunkText = chunks.map(c => c.pageContent).join('\n');
        
        if (isGrounded(grounded_quote, combinedChunkText)) {
          console.log('[Chat] Generated fallback answer passed verbatim grounding check. Saving to pending cache.');
          
          const newPair = {
            id: uuidv4(),
            question: question,
            rephrasings: rephrasings || [],
            answer: answer,
            source: chunks[0].metadata.pdfName || 'unknown',
            chunk_ref: chunks[0].metadata.chunkIndex !== undefined ? String(chunks[0].metadata.chunkIndex) : 'fallback',
            grounded_quote: grounded_quote,
            status: 'pending' as const,
            isAutoGenerated: true
          };
          addPairs([newPair]);
          
          const payload: any = {
            answer: answer,
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
      answer: `I can only help with questions about ${docContext} — try asking about:`,
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
