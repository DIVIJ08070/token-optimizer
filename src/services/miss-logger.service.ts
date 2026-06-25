import fs from 'fs';
import path from 'path';

const DATA_DIR = path.join(process.cwd(), 'data');
const MISS_LOG_FILE = path.join(DATA_DIR, 'missed-queries.jsonl');

/**
 * Logs a query that fell below the high threshold.
 * 100% local filesystem operation.
 */
export function logMissedQuery(query: string, closestMatch: string | null, score: number): void {
  if (!fs.existsSync(DATA_DIR)) {
    fs.mkdirSync(DATA_DIR, { recursive: true });
  }

  const entry = {
    timestamp: new Date().toISOString(),
    query,
    closestMatch: closestMatch || 'None',
    score: Number(score.toFixed(4)),
  };

  fs.appendFileSync(MISS_LOG_FILE, JSON.stringify(entry) + '\n', 'utf-8');
}

/**
 * Retrieves all logged missed queries for the admin UI.
 */
export function getMissedQueries(): any[] {
  if (!fs.existsSync(MISS_LOG_FILE)) return [];

  const raw = fs.readFileSync(MISS_LOG_FILE, 'utf-8');
  return raw
    .trim()
    .split('\n')
    .filter(line => line)
    .map(line => {
      try {
        return JSON.parse(line);
      } catch {
        return null;
      }
    })
    .filter(Boolean)
    .reverse(); // newest first
}

/**
 * Clears the miss log (e.g. after the admin has added new FAQs).
 */
export function clearMissedQueries(): void {
  if (fs.existsSync(MISS_LOG_FILE)) {
    fs.unlinkSync(MISS_LOG_FILE);
  }
}
