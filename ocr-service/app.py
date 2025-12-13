import io
import base64
from typing import Optional

try:
    import easyocr
except ImportError:  # pragma: no cover
    easyocr = None

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse

app = FastAPI(title="Optional EasyOCR Service")


def load_reader():
    if not easyocr:
        return None
    return easyocr.Reader(["en"], gpu=False)


reader = load_reader()


@app.post("/ocr")
async def run_ocr(file: Optional[UploadFile] = File(default=None), image_b64: Optional[str] = None):
    if not file and not image_b64:
        return JSONResponse({"error": "file or image_b64 is required"}, status_code=400)

    if reader is None:
        # Fallback stub when easyocr is not available
        return {"text": "", "words": [], "warning": "easyocr not installed"}

    if file:
        content = await file.read()
    else:
        content = base64.b64decode(image_b64)

    results = reader.readtext(io.BytesIO(content), detail=1, paragraph=False)
    words = []
    text = []
    for bbox, word, conf in results:
        text.append(word)
        xs = [p[0] for p in bbox]
        ys = [p[1] for p in bbox]
        words.append(
            {
                "text": word,
                "confidence": conf,
                "bbox": {"x0": min(xs), "y0": min(ys), "x1": max(xs), "y1": max(ys)},
            }
        )

    return {"text": " ".join(text), "words": words}


@app.get("/health")
async def health():
    return {"status": "ok"}

