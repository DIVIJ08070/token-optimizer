#!/usr/bin/env node
/**
 * pdf-renderer.mjs
 *
 * Standalone child-process script for rendering a single PDF page to PNG.
 * Runs in its own process — completely isolated from @xenova/transformers
 * and sharp, avoiding the libgio native library conflict.
 *
 * Usage (called by renderPageToImageBuffer via child_process.execFile):
 *   node src/scripts/pdf-renderer.mjs <filePath> <pageNumber> <scale>
 *
 * Output: writes raw PNG bytes to stdout.
 * Errors: written to stderr, exits with code 1.
 *
 * IMPORTANT: pdfjs uses console.log (NOT console.warn) for ALL its warning
 * messages. We must redirect console.log to stderr BEFORE requiring pdfjs,
 * otherwise pdfjs warnings pollute stdout and corrupt the PNG binary output.
 */

// ---------------------------------------------------------------------------
// Redirect console to stderr FIRST — before anything else loads
// pdfjs writes Warning/Info messages via console.log which goes to stdout.
// ---------------------------------------------------------------------------
const originalLog = console.log.bind(console);
console.log   = (...args) => process.stderr.write(args.map(String).join(' ') + '\n');
console.warn  = (...args) => process.stderr.write(args.map(String).join(' ') + '\n');
console.error = (...args) => process.stderr.write(args.map(String).join(' ') + '\n');
console.info  = (...args) => process.stderr.write(args.map(String).join(' ') + '\n');

import { createCanvas } from 'canvas';
import { createRequire } from 'module';
import fs from 'fs';

// pdfjs-dist legacy build is CJS — must require() it in an .mjs file
const require = createRequire(import.meta.url);
const pdfjsLib = require('pdfjs-dist/legacy/build/pdf.js');

const [,, filePath, pageNumberStr, scaleStr] = process.argv;

if (!filePath || !pageNumberStr) {
  process.stderr.write('Usage: pdf-renderer.mjs <filePath> <pageNumber> [scale]\n');
  process.exit(1);
}

const pageNumber = parseInt(pageNumberStr, 10);
const scale     = parseFloat(scaleStr ?? '2.0');

// ---------------------------------------------------------------------------
// NodeCanvasFactory — shape pdfjs expects
// ---------------------------------------------------------------------------

const NodeCanvasFactory = {
  create(width, height) {
    const canvas  = createCanvas(width, height);
    const context = canvas.getContext('2d');
    return { canvas, context };
  },
  reset(canvasAndContext, width, height) {
    canvasAndContext.canvas.width  = width;
    canvasAndContext.canvas.height = height;
  },
  destroy(canvasAndContext) {
    canvasAndContext.canvas.width  = 0;
    canvasAndContext.canvas.height = 0;
  },
};

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

async function main() {
  const data        = new Uint8Array(fs.readFileSync(filePath));
  const pdfDocument = await pdfjsLib.getDocument({
    data,
    useSystemFonts:  true,
    disableFontFace: true,
    canvasFactory:   NodeCanvasFactory,
    // Suppress pdfjs verbose output — messages still go to stderr via console override above
    verbosity: 0,
  }).promise;

  if (pageNumber < 1 || pageNumber > pdfDocument.numPages) {
    process.stderr.write(`Page ${pageNumber} out of range (1–${pdfDocument.numPages})\n`);
    process.exit(1);
  }

  const page     = await pdfDocument.getPage(pageNumber);
  const viewport = page.getViewport({ scale });

  const { canvas, context } = NodeCanvasFactory.create(
    Math.ceil(viewport.width),
    Math.ceil(viewport.height),
  );

  await page.render({
    canvasContext: context,
    viewport,
    canvasFactory: NodeCanvasFactory,
  }).promise;

  // Write ONLY binary PNG to stdout — no other writes to stdout allowed
  const pngBuffer = canvas.toBuffer('image/png');
  process.stdout.write(pngBuffer);
}

main().catch(err => {
  process.stderr.write(`[pdf-renderer] Error: ${err.message}\n${err.stack}\n`);
  process.exit(1);
});
