# Teacher Timetable Automation System

This project extracts teacher timetables from images (PNG, JPEG) and PDFs, converts them into a structured digital representation, and exposes APIs plus a light React UI for onboarding teachers and viewing teacher-wise schedules (multi-tenant).

## Structure
- `backend/` – Express API for uploads, preprocessing, OCR orchestration, parsing, MongoDB persistence (tenants/teachers/timetables).
- `frontend/` – React UI for tenant selection, teacher onboarding, upload, preview, and light editing of extracted timetables.
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
3. Set env for backend (see below) and start:
   ```bash
   cd backend
   MONGO_URI="mongodb+srv://.../..." TENANT_DEFAULT=default npm run dev
   ```
4. Start frontend (Vite):
   ```bash
   cd frontend && npm run dev
   ```
5. (Optional) Start EasyOCR service:
   ```bash
   cd ocr-service && uvicorn app:app --host 0.0.0.0 --port 8001
   ```

## Notes
- OCR can be powered by Tesseract.js (Node) or EasyOCR (Python) by setting `OCR_SERVICE_URL`; otherwise defaults to Tesseract.
- Multi-tenant: tenants (slug), teachers under tenant, timetables per teacher.
- The API returns structured timetable JSON including days, time slots, events, grey-block indicators, and inferred durations when missing.

