/**
 * api/review/route.ts
 *
 * Human review API for FAQ pairs.
 *
 * GET  /api/review  — list all pairs (with optional ?status= filter)
 * POST /api/review  — approve / reject / edit / delete pairs (bulk)
 *
 * Indexing is at POST /api/review/index (separate route file).
 */

import { NextRequest, NextResponse } from 'next/server';
import {
  getAllPairs,
  updatePairStatus,
  updatePair,
  deletePair,
  loadStore,
} from '@/services/faq-store.service';

// ---------------------------------------------------------------------------
// GET — list pairs
// ---------------------------------------------------------------------------

export async function GET(req: NextRequest) {
  loadStore();

  const { searchParams } = new URL(req.url);
  const statusFilter = searchParams.get('status'); // "pending" | "approved" | "rejected" | null

  let pairs = getAllPairs();

  if (statusFilter) {
    pairs = pairs.filter(p => p.status === statusFilter);
  }

  // Strip large embeddings from GET response to keep payload small
  const slim = pairs.map(({ question_embedding, rephrasing_embeddings, ...rest }) => ({
    ...rest,
    indexed: !!question_embedding,
  }));

  return NextResponse.json({ pairs: slim, total: slim.length });
}

// ---------------------------------------------------------------------------
// POST — mutations
// ---------------------------------------------------------------------------

export async function POST(req: NextRequest) {
  loadStore();

  const body = await req.json();
  const { action, ids, pair: pairUpdate } = body;

  if (!action) {
    return NextResponse.json({ error: 'action is required' }, { status: 400 });
  }

  switch (action) {
    case 'approve': {
      if (!Array.isArray(ids) || ids.length === 0) {
        return NextResponse.json({ error: 'ids array is required' }, { status: 400 });
      }
      let count = 0;
      for (const id of ids) {
        if (updatePairStatus(id, 'approved')) count++;
      }
      return NextResponse.json({ updated: count, action: 'approved' });
    }

    case 'reject': {
      if (!Array.isArray(ids) || ids.length === 0) {
        return NextResponse.json({ error: 'ids array is required' }, { status: 400 });
      }
      let count = 0;
      for (const id of ids) {
        if (updatePairStatus(id, 'rejected')) count++;
      }
      return NextResponse.json({ updated: count, action: 'rejected' });
    }

    case 'approve_all': {
      const all = getAllPairs();
      let count = 0;
      for (const p of all) {
        if (p.status === 'pending' && !p.chunk_ref.startsWith('vision')) {
          if (updatePairStatus(p.id, 'approved')) count++;
        }
      }
      return NextResponse.json({ updated: count, action: 'approve_all' });
    }

    case 'edit': {
      if (!ids || ids.length !== 1) {
        return NextResponse.json({ error: 'Exactly one id is required for edit' }, { status: 400 });
      }
      if (!pairUpdate) {
        return NextResponse.json({ error: 'pair object is required for edit' }, { status: 400 });
      }
      const ok = updatePair(ids[0], {
        question:    pairUpdate.question,
        answer:      pairUpdate.answer,
        rephrasings: pairUpdate.rephrasings,
      });
      if (!ok) return NextResponse.json({ error: 'Pair not found' }, { status: 404 });
      return NextResponse.json({ updated: 1, action: 'edit', note: 'Re-index required after editing.' });
    }

    case 'delete': {
      if (!Array.isArray(ids) || ids.length === 0) {
        return NextResponse.json({ error: 'ids array is required' }, { status: 400 });
      }
      let count = 0;
      for (const id of ids) {
        if (deletePair(id)) count++;
      }
      return NextResponse.json({ deleted: count });
    }

    default:
      return NextResponse.json({ error: `Unknown action: ${action}` }, { status: 400 });
  }
}

