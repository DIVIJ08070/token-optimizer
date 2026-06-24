import { NextRequest, NextResponse } from 'next/server';
import { extractTextFromPDF, splitPDFText } from '@/services/pdf.service';
import { addChunksToStore } from '@/services/vector.service';
import fs from 'fs';
import path from 'path';
import { v4 as uuidv4 } from 'uuid';

export async function POST(req: NextRequest) {
  try {
    const formData = await req.formData();
    const files = formData.getAll('files') as File[];

    if (!files || files.length === 0) {
      return NextResponse.json({ error: 'No files provided' }, { status: 400 });
    }

    if (files.length > 50) {
      return NextResponse.json({ error: 'Maximum 50 PDFs allowed' }, { status: 400 });
    }

    const uploadDir = path.join(process.cwd(), 'uploads');
    if (!fs.existsSync(uploadDir)) {
      fs.mkdirSync(uploadDir, { recursive: true });
    }

    let allChunks: any[] = [];

    for (const file of files) {
      if (!file.name.endsWith('.pdf')) continue;

      const bytes = await file.arrayBuffer();
      const buffer = Buffer.from(bytes);

      const fileName = `${uuidv4()}-${file.name}`;
      const filePath = path.join(uploadDir, fileName);

      fs.writeFileSync(filePath, buffer);

      const pagesText = await extractTextFromPDF(filePath);
      const chunks = await splitPDFText(pagesText, file.name);
      allChunks.push(...chunks);

      // Clean up temp file
      fs.unlinkSync(filePath);
    }

    // Add everything to local vector store
    await addChunksToStore(allChunks);

    return NextResponse.json({ message: 'Files processed successfully', chunkCount: allChunks.length });
  } catch (error: any) {
    console.error('Upload Error:', error);
    return NextResponse.json({ error: error.message }, { status: 500 });
  }
}
