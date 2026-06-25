/**
 * src/lib/queryNormalizer.ts
 *
 * Normalizes incoming user queries before they hit the local embedding model.
 * 1. Strips common conversational filler so the embedding focuses on the meat.
 * 2. Expands known acronyms/jargon into their full forms.
 *
 * Runs 100% locally. Zero API calls.
 */

// A map of abbreviations/shorthand to their full forms.
// We avoid massive synonym swapping here, keeping it focused on chat shorthand.
const SYNONYM_MAP: Record<string, string> = {
  u: 'you',
  ur: 'your',
  pls: 'please',
  plz: 'please',
  k: 'okay',
  ok: 'okay',
  wht: 'what',
  thx: 'thanks',
  // Keep domain jargon if needed
};

// Common conversational filler words that dilute semantic search meaning
const FILLER_WORDS = [
  'can you tell me',
  'what is the',
  'what are the',
  'how do i',
  'could you explain',
  'please explain',
  'i want to know',
  'do you know',
  'is there a',
  'are there any',
];

export function normalizeQuery(query: string): string {
  if (!query) return '';

  let normalized = query.toLowerCase().trim();

  // 1. Strip conversational filler
  for (const filler of FILLER_WORDS) {
    if (normalized.startsWith(filler)) {
      normalized = normalized.replace(filler, '').trim();
    }
  }

  // 1.5. Collapse repeating characters (e.g., "heyyy" -> "hey", "hii" -> "hi")
  // Replace 3 or more repeating characters with just one
  normalized = normalized.replace(/(.)\1{2,}/g, '$1');

  // 2. Expand abbreviations/shorthand
  // We use word boundaries (\b) so we don't accidentally replace parts of words
  for (const [short, full] of Object.entries(SYNONYM_MAP)) {
    const regex = new RegExp(`\\b${short}\\b`, 'g');
    normalized = normalized.replace(regex, full);
  }

  // Fallback if the user typed *only* filler words
  return normalized || query.toLowerCase().trim();
}
