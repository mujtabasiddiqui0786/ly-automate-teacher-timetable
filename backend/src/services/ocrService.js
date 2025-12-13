import axios from 'axios';
import Tesseract from 'tesseract.js';

const workerPool = [];
const OCR_SERVICE_URL = process.env.OCR_SERVICE_URL;

async function getWorker() {
  if (workerPool.length > 0) return workerPool.pop();
  const worker = await Tesseract.createWorker();
  return worker;
}

async function releaseWorker(worker) {
  workerPool.push(worker);
}

export async function runOcr(imageBuffer, options = {}) {
  const { language = 'eng' } = options;

  // Prefer external EasyOCR service when configured
  if (OCR_SERVICE_URL) {
    try {
      const { data } = await axios.post(
        `${OCR_SERVICE_URL.replace(/\/$/, '')}/ocr`,
        { image_b64: imageBuffer.toString('base64') },
        { timeout: 30000 }
      );
      return {
        text: data?.text || '',
        words: data?.words || [],
        lines: data?.lines || [],
        source: 'easyocr-service',
      };
    } catch (err) {
      // Fall back to local Tesseract
      // eslint-disable-next-line no-console
      console.warn('EasyOCR service failed, falling back to Tesseract:', err.message);
    }
  }

  const worker = await getWorker();
  try {
    const { data } = await worker.recognize(imageBuffer, language, {
      rotateAuto: true,
    });

    const words = (data?.words || []).map((w) => ({
      text: w.text,
      confidence: w.confidence,
      bbox: { x0: w.bbox?.x0, y0: w.bbox?.y0, x1: w.bbox?.x1, y1: w.bbox?.y1 },
      baseline: w.baseline,
    }));

    return {
      text: data?.text || '',
      words,
      lines: data?.lines || [],
      source: 'tesseract',
    };
  } catch (err) {
    // Provide a graceful fallback so extraction can continue
    return {
      text: '',
      words: [],
      lines: [],
      error: err.message,
    };
  } finally {
    await releaseWorker(worker);
  }
}

