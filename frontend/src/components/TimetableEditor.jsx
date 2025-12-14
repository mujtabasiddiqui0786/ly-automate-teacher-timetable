import React, { useEffect, useState } from 'react';

function TimetableEditor({ timetable, onSave }) {
  const [draft, setDraft] = useState(timetable);
  const [hasChanges, setHasChanges] = useState(false);

  useEffect(() => {
    setDraft(timetable);
    setHasChanges(false);
  }, [timetable]);

  if (!draft || !draft.events?.length) return null;

  const updateEvent = (index, key, value) => {
    const nextEvents = draft.events.map((evt, idx) => (idx === index ? { ...evt, [key]: value } : evt));
    setDraft({ ...draft, events: nextEvents });
    setHasChanges(true);
  };

  const handleSave = () => {
    onSave(draft);
    setHasChanges(false);
  };

  return (
    <div style={{ marginTop: 24, padding: 20, background: '#f9fafb', borderRadius: 12, border: '1px solid #e5e7eb' }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 20 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <span style={{ fontSize: 18 }}>✏️</span>
          <h3 style={{ margin: 0, fontSize: 16, fontWeight: 600, color: '#111827' }}>Edit Events</h3>
        </div>
        {hasChanges && (
          <span style={{ fontSize: 12, color: '#f59e0b', fontWeight: 500 }}>• Unsaved changes</span>
        )}
      </div>
      
      <div style={{ display: 'grid', gap: 12, maxHeight: 400, overflowY: 'auto', paddingRight: 8 }}>
        {draft.events?.slice(0, 50).map((evt, idx) => (
          <div
            key={`${evt.day}-${evt.timeSlot?.label || idx}`}
            style={{
              border: '1px solid #e5e7eb',
              borderRadius: 10,
              padding: 16,
              background: '#ffffff',
              transition: 'all 0.2s ease',
            }}
            onMouseEnter={(e) => {
              e.currentTarget.style.borderColor = '#c7d2fe';
              e.currentTarget.style.boxShadow = '0 2px 4px rgba(0, 0, 0, 0.05)';
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.borderColor = '#e5e7eb';
              e.currentTarget.style.boxShadow = 'none';
            }}
          >
            <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', alignItems: 'flex-end' }}>
              <label style={{ flex: 1, minWidth: 200 }}>
                <div style={{ fontSize: 12, fontWeight: 500, color: '#6b7280', marginBottom: 6 }}>Subject</div>
                <input
                  value={evt.subject || ''}
                  onChange={(e) => updateEvent(idx, 'subject', e.target.value)}
                  style={{
                    width: '100%',
                    padding: '10px 12px',
                    borderRadius: 8,
                    border: '1px solid #d1d5db',
                    fontSize: 14,
                    transition: 'all 0.2s ease',
                  }}
                  onFocus={(e) => {
                    e.currentTarget.style.borderColor = '#667eea';
                    e.currentTarget.style.boxShadow = '0 0 0 3px rgba(102, 126, 234, 0.1)';
                  }}
                  onBlur={(e) => {
                    e.currentTarget.style.borderColor = '#d1d5db';
                    e.currentTarget.style.boxShadow = 'none';
                  }}
                />
              </label>
              <label style={{ width: 140 }}>
                <div style={{ fontSize: 12, fontWeight: 500, color: '#6b7280', marginBottom: 6 }}>Duration (min)</div>
                <input
                  type="number"
                  value={evt.duration || ''}
                  onChange={(e) => updateEvent(idx, 'duration', Number(e.target.value) || null)}
                  style={{
                    width: '100%',
                    padding: '10px 12px',
                    borderRadius: 8,
                    border: '1px solid #d1d5db',
                    fontSize: 14,
                    transition: 'all 0.2s ease',
                  }}
                  onFocus={(e) => {
                    e.currentTarget.style.borderColor = '#667eea';
                    e.currentTarget.style.boxShadow = '0 0 0 3px rgba(102, 126, 234, 0.1)';
                  }}
                  onBlur={(e) => {
                    e.currentTarget.style.borderColor = '#d1d5db';
                    e.currentTarget.style.boxShadow = 'none';
                  }}
                />
              </label>
              <label
                style={{
                  width: 120,
                  display: 'flex',
                  flexDirection: 'column',
                  gap: 6,
                }}
              >
                <div style={{ fontSize: 12, fontWeight: 500, color: '#6b7280' }}>Fixed Block</div>
                <div
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    padding: '10px',
                    background: evt.isFixed ? '#dbeafe' : '#f3f4f6',
                    borderRadius: 8,
                    border: `1px solid ${evt.isFixed ? '#93c5fd' : '#d1d5db'}`,
                    cursor: 'pointer',
                    transition: 'all 0.2s ease',
                  }}
                  onClick={() => updateEvent(idx, 'isFixed', !evt.isFixed)}
                >
                  <input
                    type="checkbox"
                    checked={evt.isFixed || false}
                    onChange={(e) => updateEvent(idx, 'isFixed', e.target.checked)}
                    style={{ cursor: 'pointer', width: 18, height: 18 }}
                  />
                </div>
              </label>
            </div>
            <div
              style={{
                marginTop: 12,
                padding: '8px 12px',
                background: '#f3f4f6',
                borderRadius: 6,
                fontSize: 12,
                color: '#6b7280',
                display: 'flex',
                alignItems: 'center',
                gap: 8,
              }}
            >
              <span>📅</span>
              <span>
                {evt.day} — {evt.timeSlot?.label || `${evt.timeSlot?.start || ''} ${evt.timeSlot?.end || ''}`}
              </span>
            </div>
          </div>
        ))}
        {draft.events.length > 50 && (
          <div style={{ textAlign: 'center', padding: '16px', color: '#9ca3af', fontSize: 13 }}>
            Showing first 50 events. Total: {draft.events.length}
          </div>
        )}
      </div>
      
      <button
        type="button"
        onClick={handleSave}
        disabled={!hasChanges}
        style={{
          marginTop: 20,
          padding: '12px 24px',
          background: hasChanges
            ? 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)'
            : '#d1d5db',
          color: '#fff',
          border: 'none',
          borderRadius: 8,
          cursor: hasChanges ? 'pointer' : 'not-allowed',
          fontSize: 14,
          fontWeight: 600,
          width: '100%',
          transition: 'all 0.2s ease',
          boxShadow: hasChanges ? '0 2px 4px rgba(102, 126, 234, 0.2)' : 'none',
        }}
        onMouseEnter={(e) => {
          if (hasChanges) {
            e.currentTarget.style.transform = 'translateY(-1px)';
            e.currentTarget.style.boxShadow = '0 4px 6px rgba(102, 126, 234, 0.3)';
          }
        }}
        onMouseLeave={(e) => {
          if (hasChanges) {
            e.currentTarget.style.transform = 'translateY(0)';
            e.currentTarget.style.boxShadow = '0 2px 4px rgba(102, 126, 234, 0.2)';
          }
        }}
      >
        {hasChanges ? '💾 Save Changes' : '✓ No changes to save'}
      </button>
    </div>
  );
}

export default TimetableEditor;
