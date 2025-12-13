import Tesseract from 'tesseract.js';

const workerPool = [];

async function getWorker() {
  if (workerPool.length > 0) return workerPool.pop();
  const worker = await Tesseract.createWorker();
  await worker.loadLanguage('eng');
  await worker.initialize('eng');
  return worker;
}

async function releaseWorker(worker) {
  workerPool.push(worker);
}

export async function runOcr(imageBuffer, options = {}) {
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

