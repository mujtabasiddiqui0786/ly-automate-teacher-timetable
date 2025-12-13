import sharp from 'sharp';
import pdf from 'pdf-parse';

const IMAGE_MIME_TYPES = ['image/png', 'image/jpeg', 'image/jpg', 'image/webp'];

export async function preprocessInput(file) {
  const meta = {
    originalName: file.originalname,
    mimeType: file.mimetype,
  };

  if (!file.buffer) {
    throw new Error('Uploaded file buffer missing');
  }

  if (file.mimetype === 'application/pdf' || file.originalname?.toLowerCase().endsWith('.pdf')) {
    const pages = await pdf(file.buffer);
    meta.pageCount = pages.numpages;
    const pngBuffer = await sharp(file.buffer, { density: 300 }).png().toBuffer();
    return {
      images: [pngBuffer],
      mimeType: 'image/png',
      meta,
    };
  }

  if (IMAGE_MIME_TYPES.includes(file.mimetype)) {
    const normalized = await sharp(file.buffer)
      .rotate()
      .normalize()
      .toBuffer();
    meta.pageCount = 1;
    return { images: [normalized], mimeType: file.mimetype, meta };
  }

  // Fallback for unknown types; attempt to coerce via sharp
  const coerced = await sharp(file.buffer).png().toBuffer();
  meta.pageCount = 1;
  return { images: [coerced], mimeType: 'image/png', meta };
}

