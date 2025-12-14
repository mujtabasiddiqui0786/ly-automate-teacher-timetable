import React from 'react';

function TeacherList({ teachers, loading, onSelect, selectedId }) {
  if (loading) {
    return (
      <div style={{ textAlign: 'center', padding: '32px 0', color: '#9ca3af' }}>
        <div style={{ fontSize: 32, marginBottom: 8 }}>⏳</div>
        <p style={{ margin: 0, fontSize: 14 }}>Loading teachers...</p>
      </div>
    );
  }

  if (teachers.length === 0) {
    return (
      <div style={{ textAlign: 'center', padding: '32px 0', color: '#9ca3af' }}>
        <div style={{ fontSize: 32, marginBottom: 8 }}>👥</div>
        <p style={{ margin: 0, fontSize: 14 }}>No teachers yet. Create one above!</p>
      </div>
    );
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
      {teachers.map((t) => {
        const isSelected = t._id === selectedId;
        return (
          <button
            key={t._id}
            type="button"
            onClick={() => onSelect(t)}
            style={{
              textAlign: 'left',
              padding: '16px',
              borderRadius: 12,
              border: isSelected ? '2px solid #667eea' : '1px solid #e5e7eb',
              background: isSelected ? '#f0f4ff' : '#ffffff',
              cursor: 'pointer',
              transition: 'all 0.2s ease',
              boxShadow: isSelected ? '0 2px 4px rgba(102, 126, 234, 0.1)' : 'none',
            }}
            onMouseEnter={(e) => {
              if (!isSelected) {
                e.currentTarget.style.background = '#f9fafb';
                e.currentTarget.style.borderColor = '#d1d5db';
              }
            }}
            onMouseLeave={(e) => {
              if (!isSelected) {
                e.currentTarget.style.background = '#ffffff';
                e.currentTarget.style.borderColor = '#e5e7eb';
              }
            }}
          >
            <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 8 }}>
              <div
                style={{
                  width: 40,
                  height: 40,
                  borderRadius: '50%',
                  background: isSelected ? 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)' : '#f3f4f6',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  color: isSelected ? 'white' : '#6b7280',
                  fontSize: 16,
                  fontWeight: 600,
                  flexShrink: 0,
                }}
              >
                {t.name.charAt(0).toUpperCase()}
              </div>
              <div style={{ flex: 1, minWidth: 0 }}>
                <div style={{ fontWeight: 600, fontSize: 15, color: '#111827', marginBottom: 4 }}>
                  {t.name}
                </div>
                <div style={{ color: '#6b7280', fontSize: 13 }}>
                  {t.email || 'No email'} {t.grade ? `• ${t.grade}` : ''}
                </div>
              </div>
              {isSelected && (
                <span style={{ fontSize: 18, color: '#667eea' }}>✓</span>
              )}
            </div>
            {t.subjects?.length ? (
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6, marginTop: 8 }}>
                {t.subjects.slice(0, 3).map((subject, idx) => (
                  <span
                    key={idx}
                    style={{
                      padding: '2px 8px',
                      background: '#f3f4f6',
                      borderRadius: 4,
                      fontSize: 11,
                      color: '#4b5563',
                    }}
                  >
                    {subject}
                  </span>
                ))}
                {t.subjects.length > 3 && (
                  <span style={{ fontSize: 11, color: '#9ca3af', padding: '2px 4px' }}>
                    +{t.subjects.length - 3} more
                  </span>
                )}
              </div>
            ) : (
              <div style={{ marginTop: 8, fontSize: 12, color: '#9ca3af' }}>No subjects</div>
            )}
          </button>
        );
      })}
    </div>
  );
}

export default TeacherList;
