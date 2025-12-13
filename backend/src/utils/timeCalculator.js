function toMinutes(h, m) {
  return h * 60 + m;
}

function toClock(mins) {
  const h = Math.floor(mins / 60);
  const m = mins % 60;
  return `${h}:${m.toString().padStart(2, '0')}`;
}

export function parseTimeRange(text) {
  const match = text?.match(/(\d{1,2}):(\d{2})\s*-?\s*(\d{1,2}):(\d{2})/);
  if (!match) return null;
  const start = toMinutes(parseInt(match[1], 10), parseInt(match[2], 10));
  const end = toMinutes(parseInt(match[3], 10), parseInt(match[4], 10));
  const duration = end > start ? end - start : null;
  return {
    start: toClock(start),
    end: toClock(end),
    duration,
    label: `${toClock(start)}-${toClock(end)}`,
  };
}

export function distributeDuration(startClock, endClock, parts) {
  if (!startClock || !endClock || !parts) return [];
  const [sh, sm] = startClock.split(':').map((v) => parseInt(v, 10));
  const [eh, em] = endClock.split(':').map((v) => parseInt(v, 10));
  const start = toMinutes(sh, sm);
  const end = toMinutes(eh, em);
  if (end <= start) return [];
  const total = end - start;
  const slice = Math.floor(total / parts);
  return Array.from({ length: parts }, () => slice);
}

