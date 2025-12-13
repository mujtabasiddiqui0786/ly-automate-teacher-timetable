import { parseTimeRange, distributeDuration } from '../utils/timeCalculator.js';
import { detectOrientation } from '../utils/textOrientation.js';

function extractHeaders(cells) {
  const headerRow = cells.filter((c) => c.rowIndex === 0);
  const headerCol = cells.filter((c) => c.colIndex === 0);

  const timeSlots = headerRow
    .filter((c) => c.colIndex > 0)
    .map((c) => {
      const parsed = parseTimeRange(c.text);
      return parsed || { label: c.text || `Slot ${c.colIndex}`, start: null, end: null };
    });

  const days = headerCol
    .filter((c) => c.rowIndex > 0)
    .map((c) => c.text || `Row ${c.rowIndex}`);

  return { days, timeSlots };
}

function buildEvents(cells, days, timeSlots) {
  const events = [];

  cells.forEach((cell) => {
    if (cell.rowIndex === 0 || cell.colIndex === 0) return; // skip headers
    const day = days[cell.rowIndex - 1];
    const slot = timeSlots[cell.colIndex - 1];
    const orientation = detectOrientation(cell.bbox);
    const text = cell.text || '';
    const isFixed = /registration|lunch|break|snack/i.test(text);

    const timingInside = parseTimeRange(text);
    const duration = timingInside?.duration || slot?.duration || null;
    events.push({
      day,
      timeSlot: slot,
      subject: text.replace(/\d{1,2}:\d{2}\s*-?\s*\d{1,2}:\d{2}/g, '').trim() || 'Unlabeled',
      isFixed,
      duration,
      orientation,
    });
  });

  // Handle empty cells: inherit from top header if blank
  if (!events.length && days.length && timeSlots.length) {
    days.forEach((day) => {
      timeSlots.forEach((slot) => {
        events.push({
          day,
          timeSlot: slot,
          subject: slot.label || 'Unlabeled',
          isFixed: false,
          duration: slot?.duration || distributeDuration(slot?.start, slot?.end, 1)[0],
          orientation: 'horizontal',
        });
      });
    });
  }

  return events;
}

export function parseTimetable(tableStructure) {
  const { cells = [] } = tableStructure || {};
  const { days, timeSlots } = extractHeaders(cells);
  const events = buildEvents(cells, days, timeSlots);

  return {
    days,
    timeSlots,
    events,
  };
}

