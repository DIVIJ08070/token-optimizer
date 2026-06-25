// @ts-ignore
import * as pdfjsLib from 'pdfjs-dist/legacy/build/pdf.js';
import fs from 'fs';
import path from 'path';
import { spawn } from 'child_process';

// ---------------------------------------------------------------------------
// Threshold: pages with fewer extracted characters use the vision path
// ---------------------------------------------------------------------------

export const PAGE_TEXT_THRESHOLD = 50;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface PDFPageText {
  pageNumber: number;
  text: string;
  /** True when extracted text is below PAGE_TEXT_THRESHOLD — use vision path */
  isImageOnly: boolean;
}


// ---------------------------------------------------------------------------
// Text extraction (with isImageOnly flag)
// ---------------------------------------------------------------------------


export async function extractTextFromPDF(filePath: string): Promise<PDFPageText[]> {
  const data = new Uint8Array(fs.readFileSync(filePath));
  const pdfDocument = await pdfjsLib.getDocument({
    data,
    useSystemFonts: true,
    disableFontFace: true,
  }).promise;

  const numPages = pdfDocument.numPages;
  const extracted: PDFPageText[] = [];

  for (let i = 1; i <= numPages; i++) {
    const page = await pdfDocument.getPage(i);
    const textContent = await page.getTextContent();
    const pageText = textContent.items.map((item: any) => item.str).join(' ').trim();
    extracted.push({
      pageNumber: i,
      text: pageText,
      isImageOnly: pageText.length < PAGE_TEXT_THRESHOLD,
    });
  }

  return extracted;
}

// ---------------------------------------------------------------------------
// Vision fallback: render a single page to a PNG Buffer via child process
// ---------------------------------------------------------------------------

/**
 * Renders a single PDF page to a PNG Buffer by spawning an isolated child
 * process (pdf-renderer.mjs). The child process loads `canvas` without also
 * loading @xenova/transformers, avoiding the libgio native-library conflict
 * that would cause "mysterious crashes" if both ran in the same process.
 *
 * Scale 2.0 ≈ 144 DPI — enough for GPT-4o to read dense text.
 */
export function renderPageToImageBuffer(
  filePath: string,
  pageNumber: number,
  scale = 2.0,
): Promise<Buffer> {
  return new Promise((resolve, reject) => {
    const scriptPath = path.join(process.cwd(), 'src', 'scripts', 'pdf-renderer.mjs');
    const chunks: Buffer[] = [];
    const errChunks: string[] = [];

    // Use spawn (NOT execFile) — execFile with a callback calls
    // child.stdout.setEncoding('utf8') internally, which causes all data
    // events to emit strings instead of Buffers. spawn never touches
    // stream encoding, so stdout data events always give raw Buffers.
    const child = spawn(
      process.execPath,
      [scriptPath, filePath, String(pageNumber), String(scale)],
      { stdio: ['ignore', 'pipe', 'pipe'] },
    );

    child.stdout.on('data', (chunk: Buffer) => {
      // spawn guarantees Buffers here — no encoding is ever set
      chunks.push(chunk);
    });

    child.stderr.on('data', (chunk: Buffer | string) => {
      errChunks.push(typeof chunk === 'string' ? chunk : chunk.toString('utf8'));
    });

    child.on('close', (code) => {
      if (code !== 0) {
        reject(new Error(
          `[pdf-renderer] page ${pageNumber} exited ${code}: ${errChunks.join('').trim()}`
        ));
      } else if (chunks.length === 0) {
        reject(new Error(
          `[pdf-renderer] page ${pageNumber}: child exited 0 but wrote no bytes to stdout`
        ));
      } else {
        resolve(Buffer.concat(chunks));
      }
    });

    child.on('error', (err) => {
      reject(new Error(`[pdf-renderer] spawn error for page ${pageNumber}: ${err.message}`));
    });

    console.log(`[PDF] Rendering page ${pageNumber} via isolated child process...`);
  });
}

// ---------------------------------------------------------------------------
// Chunking (unchanged — only called for text pages)
// ---------------------------------------------------------------------------

const CHUNK_SIZE = 1000; // Middle of 800-1200
const OVERLAP = 200;     // Middle of 150-250
const MAX_LIMIT = 1200;

function createOverlap(text: string): string {
  if (text.length <= OVERLAP) return text;
  const tail = text.slice(-OVERLAP);
  const spaceIdx = tail.indexOf(' ');
  return spaceIdx !== -1 ? tail.slice(spaceIdx + 1) : tail;
}

export async function splitPDFText(pages: PDFPageText[], pdfName: string) {
  const allChunks: any[] = [];
  let chunkIndex = 0;

  for (const page of pages) {
    // Skip image-only pages — they are handled separately via vision
    if (page.isImageOnly) continue;

    const text = page.text.trim();
    if (!text) continue;

    const chunks: string[] = [];
    let currentChunk = '';

    function add(segment: string) {
      if (currentChunk.length + segment.length > CHUNK_SIZE) {
        if (currentChunk) chunks.push(currentChunk.trim());
        currentChunk = createOverlap(currentChunk) + ' ' + segment.trim();
      } else {
        currentChunk += (currentChunk ? ' ' : '') + segment.trim();
      }
    }

    function process(segment: string) {
      segment = segment.trim();
      if (!segment) return;

      if (segment.length <= MAX_LIMIT) {
        add(segment);
        return;
      }

      // Priority 1: Split on paragraph boundary
      if (segment.includes('\n\n')) {
        const parts = segment.split('\n\n');
        if (parts.length > 1 && parts.every(p => p.length < segment.length)) {
          parts.forEach(p => process(p));
          return;
        }
      }

      // Priority 2: Split on sentence boundary
      if (segment.includes('. ')) {
        const parts = segment.split(/(?<=\. )/);
        if (parts.length > 1 && parts.every(p => p.length < segment.length)) {
          parts.forEach(p => process(p));
          return;
        }
      }

      // Priority 3: Split at character limit
      let start = 0;
      while (start < segment.length) {
        let end = start + CHUNK_SIZE;

        if (end < segment.length) {
          const nextSpace = segment.indexOf(' ', end);
          if (nextSpace !== -1 && nextSpace <= start + MAX_LIMIT) {
            end = nextSpace;
          }
        }

        let chunkText = segment.slice(start, end).trim();

        if (start === 0 && currentChunk) {
          chunkText = currentChunk + ' ' + chunkText;
          currentChunk = '';
        }

        chunks.push(chunkText);
        currentChunk = createOverlap(chunkText);

        start += CHUNK_SIZE - OVERLAP;
      }
    }

    process(text);

    if (currentChunk.trim().length > 50) {
      chunks.push(currentChunk.trim());
    }

    for (const chunk of chunks) {
      if (chunk.length > 50) {
        allChunks.push({
          pageContent: chunk,
          metadata: {
            pdfName,
            pageNumber: page.pageNumber,
            chunkIndex: chunkIndex++,
          },
        });
      }
    }
  }

  console.log(`[PDF] Created ${allChunks.length} text chunks from "${pdfName}"`);
  return allChunks;
}
