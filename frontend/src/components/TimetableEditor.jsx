import React, { useEffect, useState } from 'react';

function TimetableEditor({ timetable, onSave }) {
  const [draft, setDraft] = useState(timetable);

  useEffect(() => {
    setDraft(timetable);
  }, [timetable]);

  if (!draft) return null;

  const updateEvent = (index, key, value) => {
    const nextEvents = draft.events.map((evt, idx) => (idx === index ? { ...evt, [key]: value } : evt));
    setDraft({ ...draft, events: nextEvents });
  };

  return (
    <section style={{ marginTop: 24 }}>
      <h2>Edit Events</h2>
      <div style={{ display: 'grid', gap: 12 }}>
        {draft.events?.map((evt, idx) => (
          <div
            key={`${evt.day}-${evt.timeSlot?.label || idx}`}
            style={{ border: '1px solid #eee', borderRadius: 8, padding: 12 }}
          >
            <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
              <label style={{ flex: 1, minWidth: 180 }}>
                Subject
                <input
                  value={evt.subject || ''}
                  onChange={(e) => updateEvent(idx, 'subject', e.target.value)}
                  style={{ width: '100%', padding: 6, marginTop: 4 }}
                />
              </label>
              <label style={{ width: 120 }}>
                Duration (min)
                <input
                  type="number"
                  value={evt.duration || ''}
                  onChange={(e) => updateEvent(idx, 'duration', Number(e.target.value) || null)}
                  style={{ width: '100%', padding: 6, marginTop: 4 }}
                />
              </label>
              <label style={{ width: 120 }}>
                Fixed?
                <input
                  type="checkbox"
                  checked={evt.isFixed || false}
                  onChange={(e) => updateEvent(idx, 'isFixed', e.target.checked)}
                  style={{ marginLeft: 8 }}
                />
              </label>
            </div>
            <div style={{ marginTop: 8, color: '#555' }}>
              {evt.day} — {evt.timeSlot?.label || `${evt.timeSlot?.start || ''} ${evt.timeSlot?.end || ''}`}
            </div>
          </div>
        ))}
      </div>
      <button
        type="button"
        onClick={() => onSave(draft)}
        style={{
          marginTop: 16,
          padding: '10px 16px',
          background: '#2563eb',
          color: '#fff',
          border: 'none',
          borderRadius: 6,
          cursor: 'pointer',
        }}
      >
        Save timetable
      </button>
    </section>
  );
}

export default TimetableEditor;

