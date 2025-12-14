# Frontend (React + Vite)

Multi-tenant UI for teacher onboarding, timetable upload, and viewing parsed timetables.

## Run locally
```bash
cd frontend
npm install
npm run dev -- --host 0.0.0.0 --port 3000
```
The Vite dev proxy forwards `/api` to the backend on port 4000.

## Docker (dev server)
```bash
docker build -t timetable-frontend ./frontend
docker run -p 3000:3000 timetable-frontend
```

## Env / Notes
- No additional env needed; axios uses relative `/api` calls.
- Ensure backend is reachable at `http://localhost:4000` (or adjust proxy).

