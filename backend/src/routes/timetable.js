import express from 'express';
import multer from 'multer';
import { preprocessInput } from '../services/imageProcessor.js';
import { runOcr } from '../services/ocrService.js';
import { detectTable } from '../services/tableDetector.js';
import { parseTimetable } from '../services/timetableParser.js';

const router = express.Router();

const upload = multer({
  storage: multer.memoryStorage(),
  limits: { fileSize: 15 * 1024 * 1024 },
});

// Simple in-memory store; swap with DB as needed
const timetableStore = new Map();

router.post('/extract', upload.single('file'), async (req, res, next) => {
  try {
    if (!req.file) {
      return res.status(400).json({ error: 'File is required under "file" field.' });
    }

    const preprocessed = await preprocessInput(req.file);
    const ocrResult = await runOcr(preprocessed.images[0], {
      mimeType: preprocessed.mimeType,
      language: 'eng',
    });

    const tableStructure = detectTable(ocrResult, preprocessed.meta);
    const timetable = parseTimetable(tableStructure);

    const id = Date.now().toString();
    const payload = { id, ...timetable };
    timetableStore.set(id, payload);

    return res.json({ id, timetable: payload, meta: preprocessed.meta });
  } catch (err) {
    return next(err);
  }
});

router.get('/:id', (req, res) => {
  const item = timetableStore.get(req.params.id);
  if (!item) {
    return res.status(404).json({ error: 'Timetable not found' });
  }
  return res.json(item);
});

router.put('/:id', (req, res) => {
  const item = timetableStore.get(req.params.id);
  if (!item) {
    return res.status(404).json({ error: 'Timetable not found' });
  }
  const updated = { ...item, ...req.body, id: req.params.id };
  timetableStore.set(req.params.id, updated);
  return res.json(updated);
});

export default router;

