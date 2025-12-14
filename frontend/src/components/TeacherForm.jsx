import React, { useState } from 'react';

const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;

function TeacherForm({ onCreate }) {
  const [name, setName] = useState('');
  const [email, setEmail] = useState('');
  const [subjects, setSubjects] = useState('');
  const [grade, setGrade] = useState('');
  const [errors, setErrors] = useState({});
  const [submitting, setSubmitting] = useState(false);

  const validate = () => {
    const next = {};
    if (!name.trim()) next.name = 'Name is required';
    if (email.trim() && !emailRegex.test(email.trim())) next.email = 'Email is invalid';
    return next;
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    const nextErrors = validate();
    setErrors(nextErrors);
    if (Object.keys(nextErrors).length) return;

    const cleanedSubjects = subjects
      .split(',')
      .map((s) => s.trim())
      .filter(Boolean);

    setSubmitting(true);
    try {
      await onCreate({
        name: name.trim(),
        email: email.trim() || undefined,
        grade: grade.trim() || undefined,
        subjects: cleanedSubjects,
      });
      setName('');
      setEmail('');
      setGrade('');
      setSubjects('');
      setErrors({});
    } finally {
      setSubmitting(false);
    }
  };

  const inputStyle = (hasError) => ({
    padding: 8,
    borderRadius: 6,
    border: `1px solid ${hasError ? '#fca5a5' : '#ddd'}`,
  });

  return (
    <form onSubmit={handleSubmit} style={{ border: '1px solid #eee', borderRadius: 8, padding: 12, marginBottom: 12 }}>
      <h3>Create Teacher</h3>
      <div style={{ display: 'grid', gap: 8 }}>
        <div>
          <input
            required
            placeholder="Name"
            value={name}
            onChange={(e) => setName(e.target.value)}
            style={inputStyle(errors.name)}
          />
          {errors.name && <div style={{ color: '#b91c1c', fontSize: 12 }}>{errors.name}</div>}
        </div>
        <div>
          <input
            placeholder="Email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            style={inputStyle(errors.email)}
          />
          {errors.email && <div style={{ color: '#b91c1c', fontSize: 12 }}>{errors.email}</div>}
        </div>
        <input
          placeholder="Grade / Class"
          value={grade}
          onChange={(e) => setGrade(e.target.value)}
          style={inputStyle(false)}
        />
        <input
          placeholder="Subjects (comma separated)"
          value={subjects}
          onChange={(e) => setSubjects(e.target.value)}
          style={inputStyle(false)}
        />
      </div>
      <button
        type="submit"
        disabled={submitting}
        style={{
          marginTop: 10,
          padding: '10px 14px',
          background: submitting ? '#9cbff7' : '#2563eb',
          color: '#fff',
          border: 'none',
          borderRadius: 6,
          cursor: submitting ? 'not-allowed' : 'pointer',
        }}
      >
        {submitting ? 'Creating...' : 'Create'}
      </button>
    </form>
  );
}

export default TeacherForm;

