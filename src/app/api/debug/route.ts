import { NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';
import { getAllPairs } from '@/services/faq-store.service';

export async function GET() {
  const storePath = path.join(process.cwd(), 'data', 'faq-store.json');
  return NextResponse.json({
    cwd: process.cwd(),
    dirname: __dirname,
    storePath,
    exists: fs.existsSync(storePath),
    pairs: getAllPairs()
  });
}
