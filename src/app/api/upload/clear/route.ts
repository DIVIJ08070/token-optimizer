import { NextResponse } from 'next/server';
import { clearStore } from '@/services/faq-store.service';
import { resetVectorStore } from '@/services/vector.service';

/**
 * DELETE /api/upload
 *
 * Clears ALL data:
 *   1. Wipes the in-memory FAQ store (pairs, embeddings, BM25 index)
 *   2. Deletes data/faq-store.json from disk
 *   3. Resets the in-memory vector store (PDF chunks)
 */
export async function DELETE() {
  try {
    clearStore();      // wipes FAQ pairs from memory + disk
    resetVectorStore(); // wipes PDF vector chunks from memory

    console.log('[Upload] ✅ All data cleared by user request.');

    return NextResponse.json({
      success: true,
      message: 'All data cleared. You can now upload new PDFs.',
    });
  } catch (error: any) {
    console.error('[Upload] Clear error:', error);
    return NextResponse.json({ error: error.message }, { status: 500 });
  }
}
