export function detectOrientation(bbox) {
  if (!bbox || [bbox.x0, bbox.y0, bbox.x1, bbox.y1].some((v) => v === undefined)) {
    return 'unknown';
  }
  const width = Math.abs(bbox.x1 - bbox.x0);
  const height = Math.abs(bbox.y1 - bbox.y0);
  if (height > width * 1.2) return 'vertical';
  if (width > height * 1.2) return 'horizontal';
  return 'unknown';
}

