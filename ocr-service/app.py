import io
import os
import base64
import tempfile
from typing import Optional, List, Tuple

try:
    import easyocr
    import cv2
    import numpy as np
    from PIL import Image
except ImportError:  # pragma: no cover
    easyocr = None
    cv2 = None
    np = None
    Image = None

# Optional PDF support
try:
    from pdf2image import convert_from_bytes
except ImportError:  # pragma: no cover
    convert_from_bytes = None

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder

app = FastAPI(title="Optional EasyOCR Service")


def load_reader():
    if not easyocr:
        return None
    return easyocr.Reader(["en"], gpu=False)


reader = load_reader()


def to_native(o):
    import numpy as np
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return o


def is_pdf(content: bytes, filename: Optional[str] = None) -> bool:
    if filename and filename.lower().endswith(".pdf"):
        return True
    return content.startswith(b"%PDF")


def _read_image_from_bytes(content: bytes, filename: Optional[str] = None):
    """
    Robust image reader with PDF handling and PIL fallback.
    """
    if cv2 is None:
        return None

    if is_pdf(content, filename):
        if convert_from_bytes is None:
            return None
        try:
            images = convert_from_bytes(content, dpi=300)
            if not images:
                return None
            first = images[0].convert("RGB")
            img = cv2.cvtColor(np.array(first), cv2.COLOR_RGB2BGR)
            return img
        except Exception:
            return None

    # Try OpenCV direct decode
    data = np.frombuffer(content, np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is not None:
        return img

    # PIL fallback
    try:
        pil_img = Image.open(io.BytesIO(content)).convert("RGB")
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    except Exception:
        return None


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
                "confidence": float(conf),
                "bbox": {"x0": int(min(xs)), "y0": int(min(ys)), "x1": int(max(xs)), "y1": int(max(ys))},
            }
        )

    return jsonable_encoder({"text": " ".join(text), "words": words}, custom_encoder={np.generic: to_native, np.ndarray: to_native})


@app.get("/health")
async def health():
    return {"status": "ok"}


def _deskew(gray):
    coords = np.column_stack(np.where(gray < 128))
    if coords.size == 0:
        return gray
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = gray.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated


def _adaptive_kernels(w: int, h: int) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    # Dynamic kernel sizes with guard rails (tuned upward for better line detection)
    hor_kernel_div = int(os.environ.get("HOR_KERNEL_DIV", "40"))  # was 50
    ver_kernel_div = int(os.environ.get("VER_KERNEL_DIV", "40"))  # was 50
    hk = max(3, min(80, w // hor_kernel_div))
    vk = max(3, min(80, h // ver_kernel_div))
    return (hk, 1), (1, vk)


def _cluster_lines(coords: List[int], eps: int) -> List[int]:
    if not coords:
        return []
    coords = sorted(coords)
    clusters = []
    cur = [coords[0]]
    for v in coords[1:]:
        if v - cur[-1] <= eps:
            cur.append(v)
        else:
            clusters.append(int(round(sum(cur) / len(cur))))
            cur = [v]
    clusters.append(int(round(sum(cur) / len(cur))))
    return clusters


def _find_dominant_spacing(coords: List[int]) -> float:
    """
    Find the dominant spacing between coordinates using statistical analysis.
    This helps identify the true grid spacing vs noise.
    """
    if len(coords) < 2:
        return 0.0
    spacings = [coords[i+1] - coords[i] for i in range(len(coords) - 1)]
    if not spacings:
        return 0.0
    
    # Use median as robust estimate of dominant spacing
    median_spacing = float(np.median(spacings))
    
    # Find mode by clustering spacings and taking the most common cluster
    if len(spacings) > 1:
        # Group spacings within 20% of each other
        tolerance = median_spacing * 0.2
        clusters = {}
        for s in spacings:
            matched = False
            for key in clusters:
                if abs(s - key) <= tolerance:
                    clusters[key].append(s)
                    matched = True
                    break
            if not matched:
                clusters[s] = [s]
        
        # Find largest cluster
        if clusters:
            largest_cluster = max(clusters.values(), key=len)
            return float(np.median(largest_cluster))
    
    return median_spacing


def _adaptive_eps_from_spacing(coords: List[int], image_dim: int) -> int:
    """
    Automatically determine clustering epsilon based on detected spacing patterns.
    This adapts to different image layouts.
    """
    if len(coords) < 2:
        return max(4, int(image_dim * 0.01))
    
    dominant_spacing = _find_dominant_spacing(coords)
    if dominant_spacing > 0:
        # Use 30% of dominant spacing as epsilon (adaptive to grid structure)
        eps = int(dominant_spacing * 0.3)
        # But ensure minimum and maximum bounds
        eps = max(4, min(eps, int(image_dim * 0.02)))
        return eps
    
    # Fallback to image-based heuristic
    return max(4, int(image_dim * 0.01))


def _collapse_close_lines(coords: List[int], median_cell_dim: int, threshold_fraction: float = 0.5) -> List[int]:
    """
    Collapse consecutive lines with tiny gaps to reduce over-segmentation.
    """
    if not coords or len(coords) < 2:
        return coords
    collapsed = []
    prev = coords[0]
    collapsed.append(prev)
    for x in coords[1:]:
        gap = x - prev
        threshold = threshold_fraction * median_cell_dim
        if gap < threshold:
            # merge: use midpoint
            collapsed[-1] = int(round((collapsed[-1] + x) / 2))
            prev = collapsed[-1]
        else:
            collapsed.append(x)
            prev = x
    return collapsed


def _line_positions(mask, axis=0, min_ratio=0.5, eps=5):
    proj = np.sum(mask, axis=axis)
    maxv = proj.max() if proj.size else 0
    if maxv == 0:
        return []
    thresh = maxv * min_ratio
    raw = [i for i, v in enumerate(proj) if v >= thresh]
    if not raw:
        return []
    clustered = _cluster_lines(raw, eps)
    return clustered


def _split_by_projection(binary_crop, axis="horizontal", min_gap=6):
    proj = binary_crop.sum(axis=1 if axis == "horizontal" else 0)
    if proj.size == 0:
        return []
    threshold = max(3, int(0.1 * proj.max()))
    minima = [i for i, v in enumerate(proj) if v < threshold]
    if not minima:
        return []
    splits = []
    run_start = minima[0]
    prev = minima[0]
    for m in minima[1:]:
        if m - prev > 1:
            splits.append((run_start + prev) // 2)
            run_start = m
        prev = m
    splits.append((run_start + prev) // 2)
    # filter too-close splits
    filtered = [s for s in splits if min_gap < s < (len(proj) - min_gap)]
    return filtered


def _merge_narrow_columns(cells, x_lines, y_lines, empty_threshold=0.7, narrow_frac=0.25):
    """
    Merge adjacent narrow mostly-empty columns to reduce over-segmentation.
    """
    if not cells or len(x_lines) < 3:
        return cells, x_lines
    
    # Compute column widths
    widths = [x_lines[i+1] - x_lines[i] for i in range(len(x_lines) - 1)]
    if not widths:
        return cells, x_lines
    median_w = float(np.median(widths))
    
    # Build cell map by row,col
    cell_map = {}
    for c in cells:
        r, col = c["rowIndex"], c["colIndex"]
        if (r, col) not in cell_map:
            cell_map[(r, col)] = c
    
    num_rows = max([c["rowIndex"] for c in cells]) + 1 if cells else 0
    to_merge = []
    
    for i, w in enumerate(widths):
        if w < narrow_frac * median_w and i < len(widths) - 1:
            # Check empty ratio in column i
            col_cells = [cell_map.get((r, i), {}).get("text", "") for r in range(num_rows)]
            empty_ratio = sum(1 for t in col_cells if not t or not t.strip()) / max(len(col_cells), 1)
            if empty_ratio > empty_threshold:
                to_merge.append(i)
    
    # Merge columns (merge i into i+1, update coordinates)
    if not to_merge:
        return cells, x_lines
    
    # Remove merged x_lines and update cell colIndices
    new_x_lines = [x_lines[0]]
    col_mapping = {}  # old_col -> new_col
    new_col = 0
    for i in range(len(x_lines) - 1):
        if i not in to_merge:
            new_x_lines.append(x_lines[i + 1])
            col_mapping[i] = new_col
            new_col += 1
        else:
            # Merge: keep the next x_line, map both i and i+1 to same new_col
            if i + 1 < len(x_lines) - 1:
                col_mapping[i] = new_col
                col_mapping[i + 1] = new_col
            else:
                col_mapping[i] = new_col
    
    # Update cells with new colIndices
    new_cells = []
    for c in cells:
        old_col = c["colIndex"]
        if old_col in col_mapping:
            new_c = c.copy()
            new_c["colIndex"] = col_mapping[old_col]
            new_cells.append(new_c)
    
    return new_cells, new_x_lines


def _grid_cells_from_lines(img_bgr, debug_dir=None):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = _deskew(gray)
    th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 9)
    h, w = th.shape
    (hk_w, hk_h), (vk_w, vk_h) = _adaptive_kernels(w, h)
    horiz = cv2.morphologyEx(th, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (hk_w, hk_h)))
    vert = cv2.morphologyEx(th, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (vk_w, vk_h)))

    # Adaptive eps: detect initial lines first, then determine optimal eps from spacing
    # First pass: use conservative eps to get initial line candidates
    initial_eps = max(4, int(min(w, h) * 0.008))
    x_lines_initial = _line_positions(vert, axis=0, min_ratio=0.4, eps=initial_eps)
    y_lines_initial = _line_positions(horiz, axis=1, min_ratio=0.4, eps=initial_eps)
    
    # Determine adaptive eps from detected spacing patterns
    eps_x = _adaptive_eps_from_spacing(x_lines_initial, w) if x_lines_initial else max(4, int(w * 0.01))
    eps_y = _adaptive_eps_from_spacing(y_lines_initial, h) if y_lines_initial else max(4, int(h * 0.01))
    
    # Allow override via env var, but prefer adaptive
    if os.environ.get("LINE_EPS"):
        eps_x = eps_y = max(4, int(float(os.environ.get("LINE_EPS"))))
    
    # Second pass: cluster with adaptive eps
    x_lines = _line_positions(vert, axis=0, min_ratio=0.45, eps=eps_x)
    y_lines = _line_positions(horiz, axis=1, min_ratio=0.45, eps=eps_y)

    # further prune very-close lines to avoid exploding columns/rows
    def _prune(lines, min_gap):
        pruned = []
        for v in sorted(lines):
            if not pruned or (v - pruned[-1]) >= min_gap:
                pruned.append(v)
        return pruned

    min_gap_base = float(os.environ.get("MIN_GAP", min(w, h) * 0.012))
    min_gap = max(10, int(min_gap_base))
    x_lines = _prune(x_lines, min_gap)
    y_lines = _prune(y_lines, min_gap)

    # Collapse close lines using adaptive threshold based on detected spacing
    if len(x_lines) > 2 and len(y_lines) > 2:
        # Find dominant spacing to determine collapse threshold
        dominant_w = _find_dominant_spacing(x_lines)
        dominant_h = _find_dominant_spacing(y_lines)
        
        # Use 40% of dominant spacing as collapse threshold (adaptive)
        threshold_w = int(dominant_w * 0.4) if dominant_w > 0 else w // 20
        threshold_h = int(dominant_h * 0.4) if dominant_h > 0 else h // 20
        
        x_lines = _collapse_close_lines(x_lines, threshold_w, threshold_fraction=0.5)
        y_lines = _collapse_close_lines(y_lines, threshold_h, threshold_fraction=0.5)
        
        # Adaptive dedupe: if grid is unusually large, analyze spacing distribution
        # to detect if we have multiple spacing modes (real grid vs decorative)
        if len(x_lines) > 15:  # Likely over-segmented
            spacings = [x_lines[i+1] - x_lines[i] for i in range(len(x_lines)-1)]
            if len(spacings) > 1:
                # Find if there are two distinct spacing modes
                median_sp = float(np.median(spacings))
                # Group into "narrow" (likely decorative) and "wide" (likely real grid)
                narrow_spacings = [s for s in spacings if s < median_sp * 0.7]
                wide_spacings = [s for s in spacings if s >= median_sp * 0.7]
                
                # If we have many narrow spacings, they're likely decorative
                if len(narrow_spacings) > len(wide_spacings) * 0.5:
                    # Rebuild keeping only lines that create wide spacings
                    new_x_lines = [x_lines[0]]
                    for i in range(len(x_lines) - 1):
                        spacing = x_lines[i+1] - x_lines[i]
                        if spacing >= median_sp * 0.7:
                            new_x_lines.append(x_lines[i+1])
                    if len(new_x_lines) < len(x_lines):
                        x_lines = new_x_lines
        
        # Same for y_lines
        if len(y_lines) > 15:
            spacings = [y_lines[i+1] - y_lines[i] for i in range(len(y_lines)-1)]
            if len(spacings) > 1:
                median_sp = float(np.median(spacings))
                narrow_spacings = [s for s in spacings if s < median_sp * 0.7]
                wide_spacings = [s for s in spacings if s >= median_sp * 0.7]
                
                if len(narrow_spacings) > len(wide_spacings) * 0.5:
                    new_y_lines = [y_lines[0]]
                    for i in range(len(y_lines) - 1):
                        spacing = y_lines[i+1] - y_lines[i]
                        if spacing >= median_sp * 0.7:
                            new_y_lines.append(y_lines[i+1])
                    if len(new_y_lines) < len(y_lines):
                        y_lines = new_y_lines

    # ensure boundaries
    if 0 not in x_lines:
        x_lines = [0] + x_lines
    if w - 1 not in x_lines:
        x_lines.append(w - 1)
    if 0 not in y_lines:
        y_lines = [0] + y_lines
    if h - 1 not in y_lines:
        y_lines.append(h - 1)

    x_lines = sorted(list(set(x_lines)))
    y_lines = sorted(list(set(y_lines)))

    # build cells and filter narrow columns (ignore thin decorative separators)
    cells = []
    cell_areas = []
    cell_widths = []
    cell_heights = []
    
    for ri in range(len(y_lines) - 1):
        for ci in range(len(x_lines) - 1):
            x0, x1 = x_lines[ci], x_lines[ci + 1]
            y0, y1 = y_lines[ri], y_lines[ri + 1]
            width = x1 - x0
            height = y1 - y0
            area = width * height
            cell_areas.append(area)
            cell_widths.append(width)
            cell_heights.append(height)
    
    # Compute median dimensions for filtering (adaptive threshold)
    median_width = float(np.median(cell_widths)) if cell_widths else w / 10
    median_height = float(np.median(cell_heights)) if cell_heights else h / 10
    
    # Use statistical analysis: find the dominant cell size and filter outliers
    # This adapts to different layouts automatically
    if cell_widths and len(cell_widths) > 5:
        # Find dominant width (mode) by clustering
        width_clusters = {}
        tolerance = median_width * 0.3
        for width in cell_widths:
            matched = False
            for key in width_clusters:
                if abs(width - key) <= tolerance:
                    width_clusters[key].append(width)
                    matched = True
                    break
            if not matched:
                width_clusters[width] = [width]
        
        # Use the largest cluster as dominant width
        if width_clusters:
            dominant_width_cluster = max(width_clusters.values(), key=len)
            dominant_width = float(np.median(dominant_width_cluster))
            # Filter threshold: 20% of dominant width (adaptive)
            min_width = dominant_width * 0.2
        else:
            min_width = median_width * 0.15
    else:
        min_width = median_width * 0.15
    
    # Same for height
    if cell_heights and len(cell_heights) > 5:
        height_clusters = {}
        tolerance = median_height * 0.3
        for height in cell_heights:
            matched = False
            for key in height_clusters:
                if abs(height - key) <= tolerance:
                    height_clusters[key].append(height)
                    matched = True
                    break
            if not matched:
                height_clusters[height] = [height]
        
        if height_clusters:
            dominant_height_cluster = max(height_clusters.values(), key=len)
            dominant_height = float(np.median(dominant_height_cluster))
            min_height = dominant_height * 0.2
        else:
            min_height = median_height * 0.15
    else:
        min_height = median_height * 0.15
    
    # Build cells, filtering out narrow decorative separators
    for ri in range(len(y_lines) - 1):
        for ci in range(len(x_lines) - 1):
            x0, x1 = x_lines[ci], x_lines[ci + 1]
            y0, y1 = y_lines[ri], y_lines[ri + 1]
            width = x1 - x0
            height = y1 - y0
            area = width * height
            
            # Filter: minimum area and minimum width/height to ignore thin separators
            if area >= 400 and width >= min_width and height >= min_height:
                cells.append({
                    "rowIndex": int(ri),
                    "colIndex": int(ci),
                    "bbox": {"x0": int(x0), "y0": int(y0), "x1": int(x1), "y1": int(y1)},
                })

    # Merge narrow columns
    cells, x_lines = _merge_narrow_columns(cells, x_lines, y_lines)

    # Optional debug artifacts
    if debug_dir:
        os.makedirs(debug_dir, exist_ok=True)
        cv2.imwrite(os.path.join(debug_dir, "binary.png"), th)
        cv2.imwrite(os.path.join(debug_dir, "horiz.png"), horiz)
        cv2.imwrite(os.path.join(debug_dir, "vert.png"), vert)
        overlay = cv2.cvtColor(th, cv2.COLOR_GRAY2BGR)
        for x in x_lines:
            cv2.line(overlay, (x, 0), (x, h), (0, 0, 255), 1)
        for y in y_lines:
            cv2.line(overlay, (0, y), (w, y), (0, 255, 0), 1)
        cv2.imwrite(os.path.join(debug_dir, "grid_lines.png"), overlay)
        
        # Save parameters
        import json
        params = {
            "eps_x": eps_x if 'eps_x' in locals() else None,
            "eps_y": eps_y if 'eps_y' in locals() else None,
            "min_gap": min_gap,
            "image_dims": {"w": int(w), "h": int(h)},
            "grid_shape": {"rows": len(y_lines) - 1, "cols": len(x_lines) - 1},
            "num_cells": len(cells),
        }
        with open(os.path.join(debug_dir, "params.json"), "w") as f:
            json.dump(params, f, indent=2)

    return cells


def _maybe_split_cell(cell, th_img, median_cell_w=None, median_cell_h=None):
    """
    Split cell by projection only if it's large (width > 1.5 * median or height > 1.5 * median).
    This prevents over-splitting normal cells.
    """
    x0, y0, x1, y1 = cell["bbox"]["x0"], cell["bbox"]["y0"], cell["bbox"]["x1"], cell["bbox"]["y1"]
    width = x1 - x0
    height = y1 - y0
    
    # Only split if cell is significantly larger than median (merged box)
    should_split = False
    if median_cell_w and width > 1.5 * median_cell_w:
        should_split = True
    if median_cell_h and height > 1.5 * median_cell_h:
        should_split = True
    
    if not should_split:
        return [cell]
    
    crop = th_img[y0:y1, x0:x1]
    if crop.size == 0:
        return [cell]
    splits_y = _split_by_projection(crop, axis="horizontal")
    splits_x = _split_by_projection(crop, axis="vertical")

    # no splits
    if not splits_y and not splits_x:
        return [cell]

    y_positions = [0] + splits_y + [crop.shape[0]]
    x_positions = [0] + splits_x + [crop.shape[1]]
    new_cells = []
    for ri in range(len(y_positions) - 1):
        for ci in range(len(x_positions) - 1):
            ny0 = y0 + y_positions[ri]
            ny1 = y0 + y_positions[ri + 1]
            nx0 = x0 + x_positions[ci]
            nx1 = x0 + x_positions[ci + 1]
            if (nx1 - nx0) * (ny1 - ny0) < 200:
                continue
            new_cells.append({
                "rowIndex": cell["rowIndex"],
                "colIndex": cell["colIndex"],  # keep original indices; parser can reindex
                "bbox": {"x0": int(nx0), "y0": int(ny0), "x1": int(nx1), "y1": int(ny1)},
            })
    return new_cells or [cell]


def _merge_fragments(texts):
    """
    Merge nearby word fragments from EasyOCR output with better spacing.
    """
    if not texts:
        return "", 0.0
    texts_sorted = sorted(texts, key=lambda v: (v["bbox"][0][0], v["bbox"][0][1]))
    if not texts_sorted:
        return "", 0.0
    
    # Compute approximate character width for spacing
    widths = []
    for t in texts_sorted:
        bbox = t.get("bbox", [])
        if len(bbox) >= 2:
            w = abs(bbox[1][0] - bbox[0][0]) if isinstance(bbox[0], (list, tuple)) else 0
            if w > 0 and t["text"]:
                widths.append(w / max(len(t["text"]), 1))
    
    char_width = float(np.median(widths)) if widths else 10.0
    
    # Merge words with smart spacing - join fragments that are close together
    merged_words = []
    prev_end = None
    for t in texts_sorted:
        text = t["text"].strip()
        if not text:
            continue
        bbox = t.get("bbox", [])
        if len(bbox) >= 2:
            x_start = bbox[0][0] if isinstance(bbox[0], (list, tuple)) else 0
            x_end = bbox[1][0] if isinstance(bbox[1], (list, tuple)) else x_start
            
            # If fragments are very close (within 1.5 char widths), join without space
            # Otherwise add space
            if prev_end is not None:
                gap = x_start - prev_end
                if gap > char_width * 1.5:
                    merged_words.append(" ")
                # If gap is very small, join directly (likely same word split)
            
            merged_words.append(text)
            prev_end = x_end
        else:
            merged_words.append(text)
    
    merged_text = "".join(merged_words).strip()
    # Clean up multiple spaces
    merged_text = " ".join(merged_text.split())
    conf = float(np.mean([t.get("confidence", 0.0) for t in texts_sorted])) if texts_sorted else 0.0
    return merged_text, conf


def _per_cell_ocr(img_bgr, cells, debug_dir=None):
    out = []
    h, w, _ = img_bgr.shape
    confidence_floor = float(os.environ.get("CONFIDENCE_FLOOR", "0.2"))
    cell_crops_dir = None
    if debug_dir:
        cell_crops_dir = os.path.join(debug_dir, "cell_crops")
        os.makedirs(cell_crops_dir, exist_ok=True)
    
    for cell in cells:
        b = cell["bbox"]
        x0, y0, x1, y1 = b["x0"], b["y0"], b["x1"], b["y1"]
        pad = 2
        crop = img_bgr[max(0, y0 - pad): min(h, y1 + pad), max(0, x0 - pad): min(w, x1 + pad)]
        if crop.size == 0 or reader is None:
            text = ""
            conf = 0.0
        else:
            # Try paragraph mode first for better word joining, fallback to detail mode
            try:
                r_para = reader.readtext(crop, detail=0, paragraph=True)
                if r_para and len(r_para) > 0:
                    # Paragraph mode returns text directly
                    text = " ".join(r_para).strip()
                    # Get confidence from detail mode
                    r_detail = reader.readtext(crop, detail=1, paragraph=False)
                    conf = float(np.mean([v[2] for v in r_detail])) if r_detail else 0.5
                else:
                    # Fallback to detail mode
                    r = reader.readtext(crop, detail=1, paragraph=False)
                    if r:
                        text, conf = _merge_fragments([{"text": v[1], "confidence": float(v[2]), "bbox": v[0]} for v in r])
                    else:
                        text = ""
                        conf = 0.0
            except Exception:
                # Fallback to detail mode on error
                r = reader.readtext(crop, detail=1, paragraph=False)
                if r:
                    text, conf = _merge_fragments([{"text": v[1], "confidence": float(v[2]), "bbox": v[0]} for v in r])
                else:
                    text = ""
                    conf = 0.0
            
            # Apply confidence floor
            if conf < confidence_floor:
                text = ""
                conf = 0.0
        
        # Save cell crop for debugging
        if cell_crops_dir:
            crop_file = os.path.join(cell_crops_dir, f"{cell['rowIndex']}_{cell['colIndex']}.png")
            cv2.imwrite(crop_file, crop)
        
        entry = {
            "rowIndex": int(cell["rowIndex"]),
            "colIndex": int(cell["colIndex"]),
            "bbox": {k: int(v) for k, v in b.items()},
            "text": text.strip(),
            "confidence": float(conf),
        }
        out.append(entry)

    if debug_dir:
        os.makedirs(debug_dir, exist_ok=True)
        # save a small summary
        with open(os.path.join(debug_dir, "cells_summary.txt"), "w") as f:
            for c in out[:200]:
                f.write(f"{c['rowIndex']},{c['colIndex']} -> {c['text']} (conf={c['confidence']:.2f})\n")
        
        # Save metrics
        import json
        empty_count = sum(1 for c in out if not c["text"] or not c["text"].strip())
        metrics = {
            "total_cells": len(out),
            "empty_cells": empty_count,
            "empty_cell_pct": float(empty_count / len(out) * 100) if out else 0.0,
            "avg_confidence": float(np.mean([c["confidence"] for c in out])) if out else 0.0,
            "avg_word_count": float(np.mean([len(c["text"].split()) for c in out if c["text"]])) if out else 0.0,
        }
        with open(os.path.join(debug_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)

    return out


@app.post("/ocr-table")
async def run_ocr_with_table(file: Optional[UploadFile] = File(default=None), image_b64: Optional[str] = None):
    if not file and not image_b64:
        return JSONResponse({"error": "file or image_b64 is required"}, status_code=400)
    if reader is None or cv2 is None or np is None:
        return JSONResponse({"error": "easyocr/cv2 not installed"}, status_code=500)

    filename = file.filename if file else None
    if file:
        content = await file.read()
    else:
        content = base64.b64decode(image_b64)

    debug_dir = None
    if os.environ.get("DEBUG_OCR"):
        base_dir = os.environ.get("DEBUG_DIR", "/tmp/ocr_debug")
        os.makedirs(base_dir, exist_ok=True)
        debug_dir = tempfile.mkdtemp(prefix="ocr_debug_", dir=base_dir)

    img = _read_image_from_bytes(content, filename)
    if img is None:
        return JSONResponse({"error": "failed to load image"}, status_code=400)

    # page-level OCR (fallback and text aggregation)
    page_results = reader.readtext(img, detail=1, paragraph=False)
    words = []
    for bbox, word, conf in page_results:
        xs = [p[0] for p in bbox]
        ys = [p[1] for p in bbox]
        words.append(
            {
                "text": word,
                "confidence": float(conf),
                "bbox": {"x0": int(min(xs)), "y0": int(min(ys)), "x1": int(max(xs)), "y1": int(max(ys))},
            }
        )
    page_text = " ".join(w["text"] for w in words if w.get("text"))

    # grid detection
    cells = _grid_cells_from_lines(img, debug_dir=debug_dir)
    if not cells:
        resp = {"status": "fallback_page_ocr", "source": "easyocr", "text": page_text, "words": words, "cells": []}
        return JSONResponse(jsonable_encoder(resp, custom_encoder={np.generic: to_native, np.ndarray: to_native}))

    # Optional split of merged cells using projection on thresholded image
    # Only split large cells (conditional projection split)
    th = cv2.adaptiveThreshold(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 9)
    
    # Compute median cell dimensions for conditional splitting
    median_cell_w = float(np.median([c["bbox"]["x1"] - c["bbox"]["x0"] for c in cells])) if cells else None
    median_cell_h = float(np.median([c["bbox"]["y1"] - c["bbox"]["y0"] for c in cells])) if cells else None
    
    split_cells = []
    for c in cells:
        split_cells.extend(_maybe_split_cell(c, th, median_cell_w, median_cell_h))
    
    # Deduplicate cells: if multiple cells share same (rowIndex, colIndex), keep the largest one
    cell_map = {}
    for c in split_cells:
        key = (c["rowIndex"], c["colIndex"])
        area = (c["bbox"]["x1"] - c["bbox"]["x0"]) * (c["bbox"]["y1"] - c["bbox"]["y0"])
        if key not in cell_map or area > (cell_map[key]["bbox"]["x1"] - cell_map[key]["bbox"]["x0"]) * (cell_map[key]["bbox"]["y1"] - cell_map[key]["bbox"]["y0"]):
            cell_map[key] = c
    split_cells = list(cell_map.values())

    cell_entries = _per_cell_ocr(img, split_cells, debug_dir=debug_dir)
    
    # Limit max columns/rows: if grid is too large, run aggressive collapse
    grid_rows = int(max(c["rowIndex"] for c in cell_entries) + 1 if cell_entries else 0)
    grid_cols = int(max(c["colIndex"] for c in cell_entries) + 1 if cell_entries else 0)
    expected_cols = int(os.environ.get("EXPECTED_COLS", "10"))  # typical timetable has ~6-8 days + headers
    expected_rows = int(os.environ.get("EXPECTED_ROWS", "15"))  # typical timetable has ~10-12 time slots + headers
    
    # If grid exploded, log warning but still return
    if grid_cols > expected_cols * 2 or grid_rows > expected_rows * 2:
        import logging
        logging.warning(f"Grid size {grid_rows}x{grid_cols} exceeds expected {expected_rows}x{expected_cols}. Consider tuning LINE_EPS and MIN_GAP.")
    
    resp = {
        "status": "ok",
        "source": "easyocr",
        "text": page_text,
        "words": words,
        "cells": cell_entries,
        "grid_rows": grid_rows,
        "grid_cols": grid_cols,
        "debug_dir": debug_dir if debug_dir else None,
    }
    return JSONResponse(jsonable_encoder(resp, custom_encoder={np.generic: to_native, np.ndarray: to_native}))

