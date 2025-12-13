import React, { useState } from 'react';

function TeacherForm({ onCreate }) {
  const [name, setName] = useState('');
  const [email, setEmail] = useState('');
  const [subjects, setSubjects] = useState('');
  const [grade, setGrade] = useState('');

  const handleSubmit = (e) => {
    e.preventDefault();
    onCreate({
      name,
      email,
      grade,
      subjects: subjects
        .split(',')
        .map((s) => s.trim())
        .filter(Boolean),
    });
    setName('');
    setEmail('');
    setGrade('');
    setSubjects('');
  };

  return (
    <form onSubmit={handleSubmit} style={{ border: '1px solid #eee', borderRadius: 8, padding: 12, marginBottom: 12 }}>
      <h3>Create Teacher</h3>
      <div style={{ display: 'grid', gap: 8 }}>
        <input
          required
          placeholder="Name"
          value={name}
          onChange={(e) => setName(e.target.value)}
          style={{ padding: 8, borderRadius: 6, border: '1px solid #ddd' }}
        />
        <input
          placeholder="Email"
          value={email}
          onChange={(e) => setEmail(e.target.value)}
          style={{ padding: 8, borderRadius: 6, border: '1px solid #ddd' }}
        />
        <input
          placeholder="Grade / Class"
          value={grade}
          onChange={(e) => setGrade(e.target.value)}
          style={{ padding: 8, borderRadius: 6, border: '1px solid #ddd' }}
        />
        <input
          placeholder="Subjects (comma separated)"
          value={subjects}
          onChange={(e) => setSubjects(e.target.value)}
          style={{ padding: 8, borderRadius: 6, border: '1px solid #ddd' }}
        />
      </div>
      <button
        type="submit"
        style={{
          marginTop: 10,
          padding: '10px 14px',
          background: '#2563eb',
          color: '#fff',
          border: 'none',
          borderRadius: 6,
          cursor: 'pointer',
        }}
      >
        Create
      </button>
    </form>
  );
}

export default TeacherForm;

