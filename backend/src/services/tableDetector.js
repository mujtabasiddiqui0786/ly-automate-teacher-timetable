function clusterByTolerance(values, tolerance = 12) {
  const sorted = [...values].sort((a, b) => a - b);
  const clusters = [];
  sorted.forEach((v) => {
    const cluster = clusters.find((c) => Math.abs(c.center - v) <= tolerance);
    if (cluster) {
      cluster.values.push(v);
      cluster.center = cluster.values.reduce((sum, x) => sum + x, 0) / cluster.values.length;
    } else {
      clusters.push({ center: v, values: [v] });
    }
  });
  return clusters.map((c) => c.center);
}

export function detectTable(ocrResult, meta = {}) {
  const words = ocrResult?.words || [];
  if (!words.length) {
    return { rows: [], columns: [], cells: [] };
  }

  const yCenters = words.map((w) => (w.bbox.y0 + w.bbox.y1) / 2);
  const xCenters = words.map((w) => (w.bbox.x0 + w.bbox.x1) / 2);

  const rowLines = clusterByTolerance(yCenters, 18);
  const colLines = clusterByTolerance(xCenters, 18);

  const cells = [];
  rowLines.forEach((rowCenter, rowIndex) => {
    colLines.forEach((colCenter, colIndex) => {
      const matched = words.filter((w) => {
        const yc = (w.bbox.y0 + w.bbox.y1) / 2;
        const xc = (w.bbox.x0 + w.bbox.x1) / 2;
        return Math.abs(yc - rowCenter) <= 18 && Math.abs(xc - colCenter) <= 18;
      });
      if (matched.length) {
        cells.push({
          rowIndex,
          colIndex,
          text: matched.map((m) => m.text).join(' ').trim(),
          bbox: matched.reduce(
            (acc, m) => ({
              x0: Math.min(acc.x0, m.bbox.x0),
              y0: Math.min(acc.y0, m.bbox.y0),
              x1: Math.max(acc.x1, m.bbox.x1),
              y1: Math.max(acc.y1, m.bbox.y1),
            }),
            { x0: Infinity, y0: Infinity, x1: -Infinity, y1: -Infinity }
          ),
        });
      }
    });
  });

  return {
    rows: rowLines,
    columns: colLines,
    cells,
    meta,
  };
}

