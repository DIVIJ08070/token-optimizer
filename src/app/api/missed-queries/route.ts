import { NextResponse } from 'next/server';
import { getMissedQueries, clearMissedQueries } from '@/services/miss-logger.service';

export async function GET() {
  try {
    const misses = getMissedQueries();
    return NextResponse.json({ misses });
  } catch (error: any) {
    return NextResponse.json({ error: error.message }, { status: 500 });
  }
}

export async function DELETE() {
  try {
    clearMissedQueries();
    return NextResponse.json({ success: true });
  } catch (error: any) {
    return NextResponse.json({ error: error.message }, { status: 500 });
  }
}
