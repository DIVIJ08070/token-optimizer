/**
 * src/lib/i18n.ts
 *
 * Lightweight, 100%-local language detection + a small dictionary of the bot's
 * fixed system messages in English / Hindi / Gujarati.
 *
 * Detection covers BOTH native scripts (Gujarati, Devanagari) and the romanized
 * Hinglish / Gujlish that real users actually text in ("mare app banavu che",
 * "kitna time lagega"). Zero network calls.
 */

export type Lang = 'en' | 'hi' | 'gu';

// Common romanized Gujarati tokens (distinct from English).
const GU_ROMAN = new Set([
  'che', 'chee', 'chho', 'cho', 'chu', 'shu', 'su', 'kayu', 'kayo', 'kaya', 'kayi',
  'kro', 'karo', 'kare', 'karva', 'tme', 'tame', 'tamaru', 'tmaru', 'tamne',
  'tamara', 'tamaro', 'tari', 'taru', 'banavu', 'banavi', 'banavavu', 'banavanu',
  'banavya', 'banavva', 'mare', 'mane', 'amne', 'ame', 'amaru', 'amari', 'amaru',
  'ketla', 'ketli', 'ketlo', 'ketlu', 'kem', 'nathi', 'padse', 'thase', 'thay',
  'thai', 'joiye', 'joie', 'apsi', 'apso', 'aapo', 'apo', 'kri', 'karyu', 'ena',
  'eni', 'mate', 'saru', 'sari', 'kharcho', 'rupiya', 'prise', 'gamtu', 'gamti',
  'vishe', 'janavo', 'janav', 'jano', 'mahiti', 'kaam', 'sevao', 'seva',
  'sudhi', 'hmna', 'hamna', 'atyar', 'kyare', 'kone', 'koni', 'ketlak',
  'malse', 'aapso', 'sathe', 'mateno',
]);

// Common romanized Hindi tokens (distinct from English).
const HI_ROMAN = new Set([
  'hai', 'hain', 'kya', 'kyaa', 'kaise', 'kaisa', 'kitna', 'kitne', 'kitni',
  'mujhe', 'humko', 'humein', 'hume', 'banwana', 'banwani', 'banana', 'banaye',
  'banaya', 'chahiye', 'karna', 'karwana', 'karte', 'karenge', 'nahi', 'nahin',
  'batao', 'bata', 'bataye', 'bataiye', 'hoga', 'hogi', 'kaun', 'kab', 'kyun',
  'kyon', 'mera', 'meri', 'apna', 'apni', 'aapka', 'aapki', 'rupaye', 'paisa',
  'paise', 'lagega', 'lagenge', 'samay', 'jankari', 'wala', 'wali',
  'hota', 'hoti', 'krte', 'krna',
]);

/**
 * Detect the language of a user message. Native scripts are authoritative;
 * otherwise we look for romanized Gujarati/Hindi marker words. Defaults to
 * English when there is no clear signal (so English answers are never
 * needlessly translated).
 */
export function detectLang(text: string): Lang {
  if (!text) return 'en';
  if (/[઀-૿]/.test(text)) return 'gu'; // Gujarati script
  if (/[ऀ-ॿ]/.test(text)) return 'hi'; // Devanagari (Hindi)

  const words = text.toLowerCase().replace(/[^\w\s]/g, ' ').split(/\s+/).filter(Boolean);
  let gu = 0;
  let hi = 0;
  for (const w of words) {
    if (GU_ROMAN.has(w)) gu++;
    if (HI_ROMAN.has(w)) hi++;
  }
  if (gu === 0 && hi === 0) return 'en';
  return gu >= hi ? 'gu' : 'hi';
}

export function langName(lang: Lang): string {
  return lang === 'hi' ? 'Hindi' : lang === 'gu' ? 'Gujarati' : 'English';
}

// ---------------------------------------------------------------------------
// Fixed system messages, pre-translated (free — no LLM call).
// {scope} is interpolated with the assistant's friendly scope name.
// ---------------------------------------------------------------------------

type MsgKey =
  | 'greeting'
  | 'lead'
  | 'offTopic'
  | 'clarify'
  | 'didYouMean'
  | 'notSure'
  | 'fallbackIntro'
  | 'leadThanks'
  | 'noAnswer';

const MESSAGES: Record<MsgKey, Record<Lang, string>> = {
  greeting: {
    en: 'Hello! How can I help you today?',
    hi: 'Namaste! Main aaj aapki kaise madad kar sakta hoon?',
    gu: 'Namaste! Hu aaje tamne kevi rite madad kari shaku?',
  },
  lead: {
    en: "Great! Let's get started on your project. Could you please provide your name and phone number?",
    hi: 'Bahut badhiya! Chaliye aapke project par shuru karte hain. Kripya apna naam aur phone number share karein.',
    gu: 'Saras! Chalo tamara project par sharu kariye. Krupa kari tamaru naam ane phone number aapo.',
  },
  offTopic: {
    en: 'I can only help with questions about {scope} — try asking about our services, team, or process. 😊',
    hi: 'Main sirf {scope} se jude sawalon mein madad kar sakta hoon — humari services, team ya process ke baare mein poochhein. 😊',
    gu: 'Hu fakt {scope} vishe na prashno ma madad kari shaku — amari services, team ke process vishe poochho. 😊',
  },
  clarify: {
    en: "I didn't quite catch that. Could you please rephrase your question?",
    hi: 'Main thik se samajh nahi paya. Kya aap apna sawal dobara bata sakte hain?',
    gu: 'Mane barabar samjayu nahi. Krupa kari tamaro prashna farithi puchho?',
  },
  didYouMean: {
    en: 'I found multiple matching answers. Which did you mean?',
    hi: 'Mujhe kai milte-julte jawab mile. Aapka matlab kya tha?',
    gu: 'Mane ketlak malta-julta javabo malya. Tamaro matlab kayo hato?',
  },
  notSure: {
    en: "I'm not completely sure. Did you mean one of these?",
    hi: 'Mujhe pura yakeen nahi hai. Kya aapka matlab in mein se koi tha?',
    gu: 'Mane purepuri khatri nathi. Tamaro matlab aa ma thi koi hato?',
  },
  fallbackIntro: {
    en: 'I can only help with questions about {scope} — try asking about:',
    hi: 'Main sirf {scope} se jude sawalon mein madad kar sakta hoon — ye poochh sakte hain:',
    gu: 'Hu fakt {scope} vishe na prashno ma madad kari shaku — aa poochho:',
  },
  leadThanks: {
    en: 'Thank you! Our team will review your details and contact you shortly.',
    hi: 'Dhanyavaad! Hamari team aapki details dekhkar jald hi aapse sampark karegi.',
    gu: 'Aabhar! Amari team tamari vigato joine jald tamaro sampark karshe.',
  },
  noAnswer: {
    en: "I don't have an approved answer for that question.",
    hi: 'Mere paas is sawal ka koi approved jawab nahi hai.',
    gu: 'Mara paase aa prashna no koi approved javab nathi.',
  },
};

/** Get a fixed system message in the given language, interpolating {scope}. */
export function msg(key: MsgKey, lang: Lang, scope?: string): string {
  const template = MESSAGES[key][lang] ?? MESSAGES[key].en;
  return scope ? template.replace('{scope}', scope) : template;
}
