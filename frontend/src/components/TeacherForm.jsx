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
    padding: '12px 16px',
    borderRadius: 8,
    border: `1px solid ${hasError ? '#ef4444' : '#d1d5db'}`,
    fontSize: 14,
    width: '100%',
    transition: 'all 0.2s ease',
    background: '#ffffff',
  });

  return (
    <form
      onSubmit={handleSubmit}
      style={{
        padding: 20,
        background: '#f9fafb',
        borderRadius: 12,
        border: '1px solid #e5e7eb',
        marginBottom: 20,
      }}
    >
      <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 16 }}>
        <span style={{ fontSize: 18 }}>➕</span>
        <h3 style={{ margin: 0, fontSize: 16, fontWeight: 600, color: '#111827' }}>Add New Teacher</h3>
      </div>
      <div style={{ display: 'grid', gap: 12 }}>
        <div>
          <label style={{ display: 'block', fontSize: 13, fontWeight: 500, color: '#374151', marginBottom: 6 }}>
            Name <span style={{ color: '#ef4444' }}>*</span>
          </label>
          <input
            required
            placeholder="Teacher name"
            value={name}
            onChange={(e) => {
              setName(e.target.value);
              if (errors.name) setErrors({ ...errors, name: '' });
            }}
            style={inputStyle(errors.name)}
            onFocus={(e) => {
              e.currentTarget.style.borderColor = '#667eea';
              e.currentTarget.style.boxShadow = '0 0 0 3px rgba(102, 126, 234, 0.1)';
            }}
            onBlur={(e) => {
              e.currentTarget.style.borderColor = errors.name ? '#ef4444' : '#d1d5db';
              e.currentTarget.style.boxShadow = 'none';
            }}
          />
          {errors.name && (
            <div style={{ color: '#ef4444', fontSize: 12, marginTop: 4 }}>{errors.name}</div>
          )}
        </div>
        <div>
          <label style={{ display: 'block', fontSize: 13, fontWeight: 500, color: '#374151', marginBottom: 6 }}>
            Email
          </label>
          <input
            type="email"
            placeholder="teacher@example.com"
            value={email}
            onChange={(e) => {
              setEmail(e.target.value);
              if (errors.email) setErrors({ ...errors, email: '' });
            }}
            style={inputStyle(errors.email)}
            onFocus={(e) => {
              e.currentTarget.style.borderColor = '#667eea';
              e.currentTarget.style.boxShadow = '0 0 0 3px rgba(102, 126, 234, 0.1)';
            }}
            onBlur={(e) => {
              e.currentTarget.style.borderColor = errors.email ? '#ef4444' : '#d1d5db';
              e.currentTarget.style.boxShadow = 'none';
            }}
          />
          {errors.email && (
            <div style={{ color: '#ef4444', fontSize: 12, marginTop: 4 }}>{errors.email}</div>
          )}
        </div>
        <div>
          <label style={{ display: 'block', fontSize: 13, fontWeight: 500, color: '#374151', marginBottom: 6 }}>
            Grade / Class
          </label>
          <input
            placeholder="e.g. Grade 5, Class 2A"
            value={grade}
            onChange={(e) => setGrade(e.target.value)}
            style={inputStyle(false)}
            onFocus={(e) => {
              e.currentTarget.style.borderColor = '#667eea';
              e.currentTarget.style.boxShadow = '0 0 0 3px rgba(102, 126, 234, 0.1)';
            }}
            onBlur={(e) => {
              e.currentTarget.style.borderColor = '#d1d5db';
              e.currentTarget.style.boxShadow = 'none';
            }}
          />
        </div>
        <div>
          <label style={{ display: 'block', fontSize: 13, fontWeight: 500, color: '#374151', marginBottom: 6 }}>
            Subjects
          </label>
          <input
            placeholder="Math, English, Science (comma separated)"
            value={subjects}
            onChange={(e) => setSubjects(e.target.value)}
            style={inputStyle(false)}
            onFocus={(e) => {
              e.currentTarget.style.borderColor = '#667eea';
              e.currentTarget.style.boxShadow = '0 0 0 3px rgba(102, 126, 234, 0.1)';
            }}
            onBlur={(e) => {
              e.currentTarget.style.borderColor = '#d1d5db';
              e.currentTarget.style.boxShadow = 'none';
            }}
          />
          <p style={{ margin: '6px 0 0 0', fontSize: 12, color: '#6b7280' }}>
            Separate multiple subjects with commas
          </p>
        </div>
      </div>
      <button
        type="submit"
        disabled={submitting}
        style={{
          marginTop: 16,
          padding: '12px 24px',
          background: submitting
            ? '#9ca3af'
            : 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
          color: '#fff',
          border: 'none',
          borderRadius: 8,
          cursor: submitting ? 'not-allowed' : 'pointer',
          fontSize: 14,
          fontWeight: 600,
          width: '100%',
          transition: 'all 0.2s ease',
          boxShadow: submitting ? 'none' : '0 2px 4px rgba(102, 126, 234, 0.2)',
        }}
        onMouseEnter={(e) => {
          if (!submitting) {
            e.currentTarget.style.transform = 'translateY(-1px)';
            e.currentTarget.style.boxShadow = '0 4px 6px rgba(102, 126, 234, 0.3)';
          }
        }}
        onMouseLeave={(e) => {
          if (!submitting) {
            e.currentTarget.style.transform = 'translateY(0)';
            e.currentTarget.style.boxShadow = '0 2px 4px rgba(102, 126, 234, 0.2)';
          }
        }}
      >
        {submitting ? 'Creating...' : 'Create Teacher'}
      </button>
    </form>
  );
}

export default TeacherForm;
