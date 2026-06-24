/**
 * Task C — Query classifier
 *
 * Returns true when the question is asking to aggregate/compare a single rule
 * across many or all sources.  Must stay conservative: a deep single-document
 * question (e.g. "explain the full LBW procedure in the ICC laws") must
 * return false so it stays on the normal path.
 */

// Patterns that strongly signal "across many sources" intent.
// All matching is case-insensitive.
const AGGREGATION_PATTERNS: RegExp[] = [
  // Enumerate everything
  /\b(list|enumerate|summarise|summarize)\b.{0,60}\b(all|every|each)\b/i,

  // "every / each <source noun>"
  /\b(every|each)\b.{0,30}\b(source|rulebook|document|competition|edition|version|code|book|set of rules)\b/i,

  // "across all …"
  /\bacross\s+all\b/i,

  // "for each source / document / …"
  /\bfor\s+each\s+(source|document|rulebook|competition|edition)\b/i,

  // "which sources … <verb>"
  /\bwhich\s+(sources?|documents?|rulebooks?|competitions?|editions?)\b/i,

  // "do … sources … allow/permit/prohibit/differ/conflict/vary/not"
  /\b(sources?|documents?|rulebooks?)\b.{0,60}\b(allow|permit|prohibit|differ|conflict|vary|do\s+not)\b/i,

  // "compare … sources / all / each"
  /\bcompare\b.{0,60}\b(sources?|all|each)\b/i,

  // "identify / flag … conflicts"
  /\b(identify|flag)\b.{0,60}\bconflicts?\b/i,

  // "how many sources …"
  /\bhow\s+many\s+(sources?|documents?|rulebooks?)\b/i,

  // "vary across" / "differ across"
  /\b(var(y|ies|ied|ying)|differ(s|ed|ing)?)\s+across\b/i,

  // bare "which sources"
  /\bwhich\s+sources?\b/i,
];

// Negative guard: patterns that are clearly single-document depth questions.
// If any fires, we return false regardless.
const SINGLE_DOC_GUARDS: RegExp[] = [
  // "in the <proper-noun> rules / laws / code"
  /\bin\s+the\s+(ICC|MCC|ECB|BCB|PCB|BCCI|CA|CSA|NZC|WICB|SLC|ZC|ACB)\b/i,

  // "explain … procedure / clause / law / rule" — depth, not breadth
  /\bexplain\b.{0,60}\b(procedure|clause|law|rule|provision|section|article)\b/i,

  // "full" + single topic
  /\bfull\b.{0,40}\b(procedure|explanation|detail|description)\b/i,
];

export function isAggregationQuery(query: string): boolean {
  // Short-circuit: if any single-doc guard fires, it is NOT an aggregation.
  for (const guard of SINGLE_DOC_GUARDS) {
    if (guard.test(query)) return false;
  }

  // Otherwise, fire on any aggregation pattern.
  for (const pattern of AGGREGATION_PATTERNS) {
    if (pattern.test(query)) return true;
  }

  return false;
}
