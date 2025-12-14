import React from 'react';

function TeacherList({ teachers, loading, onSelect, selectedId }) {
  return (
    <div style={{ border: '1px solid #eee', borderRadius: 8, padding: 12 }}>
      <h3>Teachers</h3>
      {loading && <p style={{ color: '#777' }}>Loading teachers...</p>}
      {!loading && teachers.length === 0 && <p style={{ color: '#777' }}>No teachers yet.</p>}
      <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
        {teachers.map((t) => (
          <button
            key={t._id}
            type="button"
            onClick={() => onSelect(t)}
            style={{
              textAlign: 'left',
              padding: '10px 12px',
              borderRadius: 6,
              border: '1px solid #ddd',
              background: t._id === selectedId ? '#e0edff' : '#fff',
              cursor: 'pointer',
            }}
          >
            <div style={{ fontWeight: 600 }}>{t.name}</div>
            <div style={{ color: '#666', fontSize: 12 }}>
              {t.email || 'No email'} {t.grade ? `• ${t.grade}` : ''}
            </div>
            {t.subjects?.length ? (
              <div style={{ color: '#555', fontSize: 12, marginTop: 4 }}>Subjects: {t.subjects.join(', ')}</div>
            ) : (
              <div style={{ color: '#777', fontSize: 12, marginTop: 4 }}>No subjects</div>
            )}
          </button>
        ))}
      </div>
    </div>
  );
}

export default TeacherList;

