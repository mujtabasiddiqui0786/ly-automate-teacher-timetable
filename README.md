# Teacher Timetable Automation System

This project extracts teacher timetables from images (PNG, JPEG) and PDFs, converts them into a structured digital representation, and exposes APIs plus a light React UI for preview/editing.

## Structure
- `backend/` – Express API for uploads, preprocessing, OCR orchestration, parsing, and timetable storage.
- `frontend/` – React UI for uploading, previewing, and editing extracted timetables.
- `ocr-service/` – Optional Python EasyOCR microservice (placeholder scaffold).
- `examples/` – Sample timetables supplied with the assignment.

## Getting Started
1. Install Node.js (18+ recommended) and Python (if using EasyOCR service).
2. Install dependencies in each package:
   ```bash
   cd backend && npm install
   cd ../frontend && npm install
   cd ../ocr-service && pip install -r requirements.txt  # optional
   ```
3. Start backend:
   ```bash
   cd backend && npm run dev
   ```
4. Start frontend:
   ```bash
   cd frontend && npm start
   ```

## Notes
- OCR can be powered by Tesseract.js (Node) or EasyOCR (Python). The code is structured to allow switching with minimal changes.
- The API returns structured timetable JSON including days, time slots, events, grey-block indicators, and inferred durations when missing.

