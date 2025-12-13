import express from 'express';
import cors from 'cors';
import dotenv from 'dotenv';
import timetableRouter from './routes/timetable.js';

dotenv.config();

const app = express();

app.use(cors());
app.use(express.json({ limit: '10mb' }));
app.use(express.urlencoded({ extended: true }));

app.get('/health', (req, res) => {
  res.json({ status: 'ok', service: 'timetable-backend' });
});

app.use('/api/timetable', timetableRouter);

app.use((err, req, res, next) => {
  // eslint-disable-line no-unused-vars
  const status = err.status || 500;
  res.status(status).json({
    error: err.message || 'Internal Server Error',
    details: process.env.NODE_ENV === 'production' ? undefined : err.stack,
  });
});

export default app;

