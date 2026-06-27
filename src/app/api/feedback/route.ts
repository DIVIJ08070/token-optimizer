import { NextRequest, NextResponse } from 'next/server';
import { addFeedback, loadStore } from '@/services/faq-store.service';

export async function POST(req: NextRequest) {
  try {
    const { pairId, feedback } = await req.json();

    if (!pairId || !feedback || (feedback !== 'up' && feedback !== 'down')) {
      return NextResponse.json({ error: 'Invalid payload' }, { status: 400 });
    }

    loadStore();
    const success = addFeedback(pairId, feedback);

    if (success) {
      return NextResponse.json({ success: true });
    } else {
      return NextResponse.json({ error: 'Pair not found' }, { status: 404 });
    }
  } catch (error: any) {
    console.error('[Feedback] API Error:', error);
    return NextResponse.json({ error: 'Internal Server Error' }, { status: 500 });
  }
}
