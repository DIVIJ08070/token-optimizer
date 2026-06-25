/**
 * api/review/index/route.ts
 *
 * POST /api/review/index
 * Embeds all approved pairs with local bge-small and saves vectors to disk.
 * This is the only route that calls the embedder — no API calls, purely local.
 */

import { NextRequest, NextResponse } from 'next/server';
import { indexApprovedPairs, getAllPairs, loadStore } from '@/services/faq-store.service';

export async function POST(_req: NextRequest) {
  try {
    loadStore();
    console.log('[Review/Index] Starting indexing of approved pairs with local bge-small...');
    const indexed       = await indexApprovedPairs();
    const approvedTotal = getAllPairs().filter(p => p.status === 'approved').length;

    return NextResponse.json({
      message:      `Indexed ${indexed} new pairs. Total approved & indexed: ${approvedTotal}.`,
      indexed,
      approvedTotal,
      chatReady:    approvedTotal > 0,
    });
  } catch (e: any) {
    console.error('[Review/Index] Indexing failed:', e);
    return NextResponse.json({ error: e.message }, { status: 500 });
  }
}
