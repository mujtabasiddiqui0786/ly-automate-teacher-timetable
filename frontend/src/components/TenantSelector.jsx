import React from 'react';

function TenantSelector({ value, onChange }) {
  return (
    <div style={{ display: 'flex', gap: 8, alignItems: 'center', marginBottom: 12 }}>
      <label style={{ fontWeight: 600 }}>Tenant slug</label>
      <input
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder="e.g. school-a"
        style={{ padding: 6, border: '1px solid #ddd', borderRadius: 6, minWidth: 180 }}
      />
    </div>
  );
}

export default TenantSelector;

