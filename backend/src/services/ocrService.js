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

async function runEasyOcr(imageBuffer, options = {}) {
  if (!OCR_SERVICE_URL) return null;
  try {
    const endpoint = `${OCR_SERVICE_URL.replace(/\/$/, '')}/ocr-table`;
    const FormData = (await import('form-data')).default;
    const formData = new FormData();
    formData.append('file', imageBuffer, {
      filename: 'image.png',
      contentType: 'image/png',
    });
    
    const { data } = await axios.post(
      endpoint,
      formData,
      {
        headers: formData.getHeaders(),
        timeout: 30000,
      }
    );
    return {
      text: data?.text || '',
      words: data?.words || [],
      cells: data?.cells || [],
      source: 'easyocr',
    };
  } catch (err) {
    // eslint-disable-next-line no-console
    console.warn('EasyOCR service failed, falling back to Tesseract:', err.message);
    return null;
  }
}

async function runTesseract(imageBuffer, options = {}) {
  const { language = 'eng' } = options;
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
    return {
      text: '',
      words: [],
      lines: [],
      error: err.message,
      source: 'tesseract',
    };
  } finally {
    await releaseWorker(worker);
  }
}

export async function runOcr(imageBuffer, options = {}) {
  // Try EasyOCR-first
  const easy = await runEasyOcr(imageBuffer, options);
  if (easy) return easy;
  // Fallback to Tesseract
  return runTesseract(imageBuffer, options);
}

