import React from 'react';

function Legend() {
  return (
    <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', marginBottom: 8 }}>
      <span style={{ fontSize: 12, color: '#555' }}>
        <span style={{ display: 'inline-block', width: 12, height: 12, background: '#e2f3ff', marginRight: 4 }} />
        Fixed block
      </span>
      <span style={{ fontSize: 12, color: '#555' }}>
        <span style={{ display: 'inline-block', width: 12, height: 12, background: '#fff3e0', marginRight: 4 }} />
        Has timing
      </span>
      <span style={{ fontSize: 12, color: '#555' }}>
        <span style={{ display: 'inline-block', width: 12, height: 12, background: '#f8fafc', marginRight: 4 }} />
        Label only
      </span>
    </div>
  );
}

function TimetableDisplay({ timetable }) {
  if (!timetable) return null;
  const { days = [], timeSlots = [], events = [], term, week, source = 'tesseract' } = timetable;

  const getEvent = (day, slot) =>
    events.find(
      (e) =>
        e.day === day &&
        ((e.timeSlot?.label && slot?.label && e.timeSlot.label === slot.label) ||
          (e.timeSlot?.start === slot?.start && e.timeSlot?.end === slot?.end))
    );

  const renderCell = (day, slot) => {
    const event = getEvent(day, slot);
    const key = `${day}-${slot?.label || slot?.start || Math.random()}`;
    if (!event) return <td key={key} style={{ border: '1px solid #eee', background: '#f8fafc' }} />;

    const hasTime = Boolean((event.start && event.end) || (slot?.start && slot?.end));
    const bg = event.isFixed ? '#e2f3ff' : hasTime ? '#fff3e0' : '#f8fafc';
    const label = event.timeSlot?.label || slot?.label || '';
    const start = event.start || event.timeSlot?.start || slot?.start;
    const end = event.end || event.timeSlot?.end || slot?.end;
    const duration = event.duration;

    const tooltip = [start && end ? `${start} - ${end}` : null, duration ? `${duration} min` : null]
      .filter(Boolean)
      .join(' · ');

    return (
      <td
        key={key}
        title={tooltip || event.subject}
        style={{
          border: '1px solid #ddd',
          padding: 8,
          background: bg,
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
          {event.isFixed && (
            <span
              style={{
                display: 'inline-block',
                width: 8,
                height: 8,
                borderRadius: 999,
                background: '#0284c7',
              }}
              title="Fixed block"
            />
          )}
          <div style={{ fontWeight: 600 }}>{event.subject}</div>
        </div>
        {label && <div style={{ fontSize: 12, color: '#555' }}>{label}</div>}
        {start && end && (
          <div style={{ fontSize: 12, color: '#555' }}>
            {start} - {end}
          </div>
        )}
        {duration && <div style={{ fontSize: 12, color: '#777' }}>{duration} mins</div>}
      </td>
    );
  };

  return (
    <section style={{ marginTop: 24 }}>
      <h2>Extracted Timetable</h2>
      <div style={{ color: '#555', marginBottom: 8, display: 'flex', gap: 12, flexWrap: 'wrap' }}>
        {term && <span>Term: {term}</span>}
        {week && <span>Week: {week}</span>}
        <span>Source: {source}</span>
      </div>
      <Legend />
      <table style={{ width: '100%', borderCollapse: 'collapse', tableLayout: 'fixed' }}>
        <thead>
          <tr>
            <th style={{ width: 120 }} />
            {timeSlots.map((slot, idx) => (
              <th key={slot.label || idx} style={{ border: '1px solid #ddd', padding: 8 }}>
                {slot.label || `${slot.start || ''} ${slot.end || ''}`}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {days.map((day) => (
            <tr key={day}>
              <th style={{ border: '1px solid #ddd', padding: 8, textAlign: 'left' }}>{day}</th>
              {timeSlots.map((slot) => renderCell(day, slot))}
            </tr>
          ))}
        </tbody>
      </table>
    </section>
  );
}

export default TimetableDisplay;

