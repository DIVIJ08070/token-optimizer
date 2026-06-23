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

export async function splitPDFText(pages: PDFPageText[], pdfName: string): Promise<any[]> {
  const allChunks: any[] = [];
  
  let currentLaw = "General";
  let currentLawContent: string[] = [];
  let currentPage = 1;

  const saveChunk = () => {
    if (currentLawContent.length > 0) {
      allChunks.push({
        pageContent: `[Law ${currentLaw}]\n${currentLawContent.join(' ')}`,
        metadata: {
          pdfName,
          law: currentLaw,
          pageNumber: currentPage
        }
      });
      currentLawContent = [];
    }
  };

  for (const page of pages) {
    currentPage = page.pageNumber;
    if (!page.text.trim()) continue;

    // Split text into sentences using basic punctuation heuristics.
    // This looks for '.', '!', or '?' followed by a space and an uppercase letter or number.
    const sentences = page.text.split(/(?<=[.?!])\s+(?=[A-Z0-9])/);

    for (const sentence of sentences) {
      const trimmedSentence = sentence.trim();
      if (!trimmedSentence) continue;

      // Look for a Law number at the start of the sentence (e.g., "24.2.2", "1.1")
      const lawMatch = trimmedSentence.match(/^(\d{1,2}\.\d{1,2}(?:\.\d{1,2})?)\b/);
      
      if (lawMatch) {
        // We found a new Law section, save the previous chunk
        saveChunk();
        
        currentLaw = lawMatch[1];
        currentLawContent.push(trimmedSentence);
      } else {
        currentLawContent.push(trimmedSentence);
      }
    }
  }
  
  // Save any remaining content in the last chunk
  saveChunk();

  return allChunks;
}
