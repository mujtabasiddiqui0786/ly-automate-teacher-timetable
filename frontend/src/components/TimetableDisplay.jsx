import React from 'react';

function Legend() {
  return (
    <div style={{ display: 'flex', gap: 16, flexWrap: 'wrap', marginBottom: 20, padding: '12px 16px', background: '#f9fafb', borderRadius: 8 }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
        <div
          style={{
            width: 16,
            height: 16,
            borderRadius: 4,
            background: '#dbeafe',
            border: '1px solid #93c5fd',
          }}
        />
        <span style={{ fontSize: 13, color: '#374151', fontWeight: 500 }}>Fixed Block</span>
      </div>
      <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
        <div
          style={{
            width: 16,
            height: 16,
            borderRadius: 4,
            background: '#fef3c7',
            border: '1px solid #fcd34d',
          }}
        />
        <span style={{ fontSize: 13, color: '#374151', fontWeight: 500 }}>Has Timing</span>
      </div>
      <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
        <div
          style={{
            width: 16,
            height: 16,
            borderRadius: 4,
            background: '#f9fafb',
            border: '1px solid #e5e7eb',
          }}
        />
        <span style={{ fontSize: 13, color: '#374151', fontWeight: 500 }}>Empty</span>
      </div>
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
    if (!event) {
      return (
        <td
          key={key}
          style={{
            border: '1px solid #e5e7eb',
            background: '#f9fafb',
            padding: 12,
            minWidth: 120,
            verticalAlign: 'top',
          }}
        />
      );
    }

    const hasTime = Boolean((event.start && event.end) || (slot?.start && slot?.end));
    const bg = event.isFixed ? '#dbeafe' : hasTime ? '#fef3c7' : '#ffffff';
    const borderColor = event.isFixed ? '#93c5fd' : hasTime ? '#fcd34d' : '#e5e7eb';
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
          border: `1px solid ${borderColor}`,
          padding: 12,
          background: bg,
          minWidth: 120,
          verticalAlign: 'top',
          borderRadius: 4,
        }}
      >
        <div style={{ display: 'flex', alignItems: 'flex-start', gap: 8 }}>
          {event.isFixed && (
            <span
              style={{
                display: 'inline-flex',
                alignItems: 'center',
                justifyContent: 'center',
                width: 20,
                height: 20,
                borderRadius: '50%',
                background: '#2563eb',
                color: 'white',
                fontSize: 10,
                fontWeight: 600,
                flexShrink: 0,
              }}
              title="Fixed block"
            >
              🔒
            </span>
          )}
          <div style={{ flex: 1, minWidth: 0 }}>
            <div style={{ fontWeight: 600, fontSize: 14, color: '#111827', marginBottom: 4 }}>
              {event.subject}
            </div>
            {label && (
              <div style={{ fontSize: 12, color: '#6b7280', marginBottom: 2 }}>{label}</div>
            )}
            {start && end && (
              <div style={{ fontSize: 12, color: '#4b5563', fontWeight: 500, marginBottom: 2 }}>
                {start} - {end}
              </div>
            )}
            {duration && (
              <div style={{ fontSize: 11, color: '#9ca3af', marginTop: 4 }}>
                {duration} minutes
              </div>
            )}
          </div>
        </div>
      </td>
    );
  };

  return (
    <div>
      {(term || week || source) && (
        <div
          style={{
            display: 'flex',
            gap: 16,
            flexWrap: 'wrap',
            marginBottom: 20,
            padding: '12px 16px',
            background: '#f0f4ff',
            borderRadius: 8,
            border: '1px solid #c7d2fe',
          }}
        >
          {term && (
            <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
              <span style={{ fontSize: 14 }}>📅</span>
              <span style={{ fontSize: 13, color: '#374151', fontWeight: 500 }}>Term: {term}</span>
            </div>
          )}
          {week && (
            <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
              <span style={{ fontSize: 14 }}>📆</span>
              <span style={{ fontSize: 13, color: '#374151', fontWeight: 500 }}>Week: {week}</span>
            </div>
          )}
          <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
            <span style={{ fontSize: 14 }}>🔍</span>
            <span style={{ fontSize: 13, color: '#374151', fontWeight: 500 }}>
              Source: {source === 'easyocr' ? 'EasyOCR' : 'Tesseract'}
            </span>
          </div>
        </div>
      )}
      <Legend />
      <div style={{ overflowX: 'auto', borderRadius: 12, border: '1px solid #e5e7eb' }}>
        <table
          style={{
            width: '100%',
            borderCollapse: 'separate',
            borderSpacing: 0,
            background: '#ffffff',
            minWidth: 600,
          }}
        >
          <thead>
            <tr>
              <th
                style={{
                  width: 140,
                  padding: '14px 16px',
                  background: '#f9fafb',
                  borderBottom: '2px solid #e5e7eb',
                  textAlign: 'left',
                  fontWeight: 600,
                  fontSize: 13,
                  color: '#374151',
                  position: 'sticky',
                  left: 0,
                  zIndex: 10,
                }}
              >
                Day
              </th>
              {timeSlots.map((slot, idx) => (
                <th
                  key={slot.label || idx}
                  style={{
                    padding: '14px 16px',
                    background: '#f9fafb',
                    borderBottom: '2px solid #e5e7eb',
                    borderLeft: '1px solid #e5e7eb',
                    textAlign: 'center',
                    fontWeight: 600,
                    fontSize: 13,
                    color: '#374151',
                    minWidth: 140,
                  }}
                >
                  {slot.label || (slot.start && slot.end ? `${slot.start} - ${slot.end}` : `Slot ${idx + 1}`)}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {days.map((day, dayIdx) => (
              <tr key={day}>
                <th
                  style={{
                    padding: '14px 16px',
                    background: dayIdx % 2 === 0 ? '#ffffff' : '#f9fafb',
                    borderRight: '2px solid #e5e7eb',
                    borderBottom: '1px solid #e5e7eb',
                    textAlign: 'left',
                    fontWeight: 600,
                    fontSize: 14,
                    color: '#111827',
                    position: 'sticky',
                    left: 0,
                    zIndex: 5,
                  }}
                >
                  {day}
                </th>
                {timeSlots.map((slot) => renderCell(day, slot))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

export default TimetableDisplay;
