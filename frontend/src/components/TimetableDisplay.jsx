import React from 'react';

function TimetableDisplay({ timetable }) {
  if (!timetable) return null;
  const { days = [], timeSlots = [], events = [], term, week } = timetable;

  const getEvent = (day, slot) =>
    events.find(
      (e) =>
        e.day === day &&
        ((e.timeSlot?.label && slot?.label && e.timeSlot.label === slot.label) ||
          (e.timeSlot?.start === slot?.start && e.timeSlot?.end === slot?.end))
    );

  const renderCell = (day, slot) => {
    const event = getEvent(day, slot);
    if (!event) return <td key={`${day}-${slot?.label || slot?.start}`} />;
    return (
      <td
        key={`${day}-${slot?.label || slot?.start}`}
        style={{
          border: '1px solid #ddd',
          padding: 8,
          background: event.isFixed ? '#f1f5f9' : '#fff',
        }}
      >
        <div style={{ fontWeight: 600 }}>{event.subject}</div>
        {slot?.start && slot?.end && (
          <div style={{ fontSize: 12, color: '#555' }}>
            {slot.start} - {slot.end}
          </div>
        )}
        {event.duration && (
          <div style={{ fontSize: 12, color: '#777' }}>{event.duration} mins</div>
        )}
      </td>
    );
  };

  return (
    <section style={{ marginTop: 24 }}>
      <h2>Extracted Timetable</h2>
      <div style={{ color: '#555', marginBottom: 8 }}>
        {term && <span style={{ marginRight: 12 }}>Term: {term}</span>}
        {week && <span>Week: {week}</span>}
      </div>
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

