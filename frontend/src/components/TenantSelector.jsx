import React, { useState, useEffect } from 'react';

function TenantSelector({ value, onChange }) {
  const [local, setLocal] = useState(value || '');
  const [error, setError] = useState('');

  useEffect(() => {
    setLocal(value || '');
  }, [value]);

  const handleBlur = () => {
    const next = (local || '').trim().toLowerCase();
    if (!next) {
      setError('Tenant slug is required');
      return;
    }
    if (!/^[a-z0-9-]+$/.test(next)) {
      setError('Only lowercase letters, numbers, and hyphens allowed');
      return;
    }
    setError('');
    onChange(next);
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
      <label style={{ fontSize: 14, fontWeight: 500, color: '#374151' }}>Organization / Tenant</label>
      <input
        value={local}
        onChange={(e) => {
          setLocal(e.target.value);
          setError('');
        }}
        onBlur={handleBlur}
        placeholder="e.g. school-a, district-1"
        style={{
          padding: '12px 16px',
          border: `1px solid ${error ? '#ef4444' : '#d1d5db'}`,
          borderRadius: 8,
          fontSize: 14,
          transition: 'all 0.2s ease',
          background: '#ffffff',
        }}
        onFocus={(e) => {
          e.currentTarget.style.borderColor = '#667eea';
          e.currentTarget.style.boxShadow = '0 0 0 3px rgba(102, 126, 234, 0.1)';
        }}
        onBlur={(e) => {
          e.currentTarget.style.borderColor = error ? '#ef4444' : '#d1d5db';
          e.currentTarget.style.boxShadow = 'none';
        }}
      />
      {error && (
        <span style={{ color: '#ef4444', fontSize: 12, display: 'flex', alignItems: 'center', gap: 4 }}>
          <span>⚠</span> {error}
        </span>
      )}
      {!error && local && (
        <span style={{ color: '#6b7280', fontSize: 12, display: 'flex', alignItems: 'center', gap: 4 }}>
          <span>✓</span> Valid tenant slug
        </span>
      )}
    </div>
  );
}

export default TenantSelector;
