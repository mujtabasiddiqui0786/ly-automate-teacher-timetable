# Backend (Express)

Express API that ingests timetable images/PDFs, preprocesses them, performs OCR, detects table structures, parses events into structured JSON, and persists multi-tenant data (tenants, teachers, timetables) in MongoDB.

## Scripts
- `npm start` – run server
- `npm run dev` – run with nodemon
- `npm run lint` – lint source

## Env
- `PORT` (default 4000)
- `MONGO_URI` (required)
- `TENANT_DEFAULT` (optional slug to auto-create on boot)
- `OCR_SERVICE_URL` (optional: EasyOCR microservice; falls back to local Tesseract if unavailable)

## Key endpoints
- `POST /api/tenants` – create/upsert tenant `{name, slug}`
- `POST /api/tenants/:tenantId/teachers` – create teacher
- `GET /api/tenants/:tenantId/teachers` – list teachers
- `GET /api/tenants/:tenantId/teachers/:teacherId` – get teacher
- `POST /api/tenants/:tenantId/teachers/:teacherId/timetable/extract` – upload image/PDF, OCR+parse, persist timetable
- `GET /api/tenants/:tenantId/teachers/:teacherId/timetable` – fetch latest timetable for teacher

