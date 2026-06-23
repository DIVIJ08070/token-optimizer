// @ts-ignore
import * as pdfjsLib from 'pdfjs-dist/legacy/build/pdf.js';
import fs from 'fs';

export interface PDFPageText {
  pageNumber: number;
  text: string;
}

export async function extractTextFromPDF(filePath: string): Promise<PDFPageText[]> {
  const data = new Uint8Array(fs.readFileSync(filePath));
  const pdfDocument = await pdfjsLib.getDocument({
    data,
    useSystemFonts: true,
  }).promise;

  const numPages = pdfDocument.numPages;
  const extracted: PDFPageText[] = [];

  for (let i = 1; i <= numPages; i++) {
    const page = await pdfDocument.getPage(i);
    const textContent = await page.getTextContent();
    const pageText = textContent.items.map((item: any) => item.str).join(' ');
    extracted.push({ pageNumber: i, text: pageText });
  }

  return extracted;
}

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
    const text = page.text.trim();
    if (!text) continue;

    const chunks: string[] = [];
    let currentChunk = "";

    function add(segment: string) {
      if (currentChunk.length + segment.length > CHUNK_SIZE) {
        if (currentChunk) chunks.push(currentChunk.trim());
        currentChunk = createOverlap(currentChunk) + " " + segment.trim();
      } else {
        currentChunk += (currentChunk ? " " : "") + segment.trim();
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

        // If this is the first hard-split chunk, prepend the trailing context
        if (start === 0 && currentChunk) {
          chunkText = currentChunk + " " + chunkText;
          currentChunk = ""; // Clear it since we embedded it
        }

        chunks.push(chunkText);
        currentChunk = createOverlap(chunkText);

        start += (CHUNK_SIZE - OVERLAP);
      }
    }

    process(text);

    if (currentChunk.trim().length > 50) {
      chunks.push(currentChunk.trim());
    }

    // Map to metadata format
    for (const chunk of chunks) {
      if (chunk.length > 50) {
        allChunks.push({
          pageContent: chunk,
          metadata: {
            pdfName,
            pageNumber: page.pageNumber,
            chunkIndex: chunkIndex++,
          }
        });
      }
    }
  }

  console.log(`[PDF] Created ${allChunks.length} chunks from ${pdfName}`);
  return allChunks;
}


