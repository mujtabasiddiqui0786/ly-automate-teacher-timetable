# OCR Service (Optional EasyOCR)

FastAPI service wrapping EasyOCR. Backend prefers this service when `OCR_SERVICE_URL` is set; otherwise falls back to Tesseract.js.

## Run locally (pip)
```bash
cd ocr-service
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8001
```

## Poetry (optional)
```bash
poetry install
poetry run uvicorn app:app --host 0.0.0.0 --port 8001
```

## Docker
```bash
docker build -t timetable-ocr ./ocr-service
docker run -p 8001:8001 timetable-ocr
```

## API
- `POST /ocr` with `file` upload or `image_b64` base64 payload.
- `GET /health`

