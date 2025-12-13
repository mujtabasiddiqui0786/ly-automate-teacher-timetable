import express from 'express';
import multer from 'multer';
import { preprocessInput } from '../services/imageProcessor.js';
import { runOcr } from '../services/ocrService.js';
import { detectTable } from '../services/tableDetector.js';
import { parseTimetable } from '../services/timetableParser.js';
import { Tenant } from '../models/Tenant.js';
import { Teacher } from '../models/Teacher.js';
import { Timetable } from '../models/Timetable.js';

const router = express.Router();

const upload = multer({
  storage: multer.memoryStorage(),
  limits: { fileSize: 15 * 1024 * 1024 },
});

async function findTenant(tenantId) {
  const query = /^[a-f\d]{24}$/i.test(tenantId) ? { _id: tenantId } : { slug: tenantId.toLowerCase() };
  return Tenant.findOne(query);
}

async function findTeacher(tenantId, teacherId) {
  const query =
    /^[a-f\d]{24}$/i.test(teacherId) && teacherId.length === 24
      ? { _id: teacherId, tenantId }
      : { _id: teacherId, tenantId }; // keep strict _id for now
  return Teacher.findOne(query);
}

router.post('/', async (req, res, next) => {
  try {
    const { name, slug } = req.body;
    if (!slug || !name) return res.status(400).json({ error: 'name and slug are required' });
    const tenant = await Tenant.findOneAndUpdate(
      { slug: slug.toLowerCase() },
      { slug: slug.toLowerCase(), name },
      { upsert: true, new: true, setDefaultsOnInsert: true }
    );
    return res.json(tenant);
  } catch (err) {
    return next(err);
  }
});

router.post('/:tenantId/teachers', async (req, res, next) => {
  try {
    const tenant = await findTenant(req.params.tenantId);
    if (!tenant) return res.status(404).json({ error: 'Tenant not found' });
    const { name, email, subjects = [], grade, meta = {} } = req.body;
    if (!name) return res.status(400).json({ error: 'Teacher name is required' });
    const teacher = await Teacher.create({
      tenantId: tenant._id,
      name,
      email,
      subjects,
      grade,
      meta,
    });
    return res.status(201).json(teacher);
  } catch (err) {
    return next(err);
  }
});

router.get('/:tenantId/teachers', async (req, res, next) => {
  try {
    const tenant = await findTenant(req.params.tenantId);
    if (!tenant) return res.status(404).json({ error: 'Tenant not found' });
    const teachers = await Teacher.find({ tenantId: tenant._id }).sort({ createdAt: -1 });
    return res.json(teachers);
  } catch (err) {
    return next(err);
  }
});

router.get('/:tenantId/teachers/:teacherId', async (req, res, next) => {
  try {
    const tenant = await findTenant(req.params.tenantId);
    if (!tenant) return res.status(404).json({ error: 'Tenant not found' });
    const teacher = await findTeacher(tenant._id, req.params.teacherId);
    if (!teacher) return res.status(404).json({ error: 'Teacher not found' });
    return res.json(teacher);
  } catch (err) {
    return next(err);
  }
});

router.post(
  '/:tenantId/teachers/:teacherId/timetable/extract',
  upload.single('file'),
  async (req, res, next) => {
    try {
      const tenant = await findTenant(req.params.tenantId);
      if (!tenant) return res.status(404).json({ error: 'Tenant not found' });
      const teacher = await findTeacher(tenant._id, req.params.teacherId);
      if (!teacher) return res.status(404).json({ error: 'Teacher not found' });
      if (!req.file) return res.status(400).json({ error: 'File is required under "file" field.' });

      const { term, week } = req.body;
      const preprocessed = await preprocessInput(req.file);
      const ocrResult = await runOcr(preprocessed.images[0], {
        mimeType: preprocessed.mimeType,
        language: 'eng',
      });
      const tableStructure = detectTable(ocrResult, preprocessed.meta);
      const parsed = parseTimetable(tableStructure);

      const timetable = await Timetable.create({
        tenantId: tenant._id,
        teacherId: teacher._id,
        term,
        week,
        days: parsed.days || [],
        timeSlots: parsed.timeSlots || [],
        events: (parsed.events || []).map((evt) => ({
          ...evt,
          start: evt.timeSlot?.start,
          end: evt.timeSlot?.end,
          timeSlot: evt.timeSlot,
          teacherName: teacher.name,
          source: ocrResult?.source || 'tesseract',
        })),
        rawText: ocrResult?.text || '',
        sourceImageMeta: preprocessed.meta,
      });

      return res.status(201).json(timetable);
    } catch (err) {
      return next(err);
    }
  }
);

router.get('/:tenantId/teachers/:teacherId/timetable', async (req, res, next) => {
  try {
    const tenant = await findTenant(req.params.tenantId);
    if (!tenant) return res.status(404).json({ error: 'Tenant not found' });
    const teacher = await findTeacher(tenant._id, req.params.teacherId);
    if (!teacher) return res.status(404).json({ error: 'Teacher not found' });

    const timetable = await Timetable.findOne({ tenantId: tenant._id, teacherId: teacher._id })
      .sort({ createdAt: -1 })
      .lean();
    if (!timetable) return res.status(404).json({ error: 'Timetable not found' });
    return res.json(timetable);
  } catch (err) {
    return next(err);
  }
});

export default router;

