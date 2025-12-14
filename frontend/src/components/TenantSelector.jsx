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
    setError('');
    onChange(next);
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
      <label style={{ fontWeight: 600 }}>Tenant slug</label>
      <input
        value={local}
        onChange={(e) => setLocal(e.target.value)}
        onBlur={handleBlur}
        placeholder="e.g. school-a"
        style={{
          padding: 8,
          border: `1px solid ${error ? '#fca5a5' : '#ddd'}`,
          borderRadius: 6,
          minWidth: 180,
        }}
      />
      {error && <span style={{ color: '#b91c1c', fontSize: 12 }}>{error}</span>}
    </div>
  );
}

export default TenantSelector;

