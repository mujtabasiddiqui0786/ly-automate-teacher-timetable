import React, { useRef, useState } from 'react';

function ImageUpload({ onUpload, loading }) {
  const inputRef = useRef(null);
  const [fileName, setFileName] = useState('');
  const [isDragging, setIsDragging] = useState(false);

  const handleFiles = (files) => {
    const file = files?.[0];
    if (!file) return;
    setFileName(file.name);
    onUpload(file);
  };

  const onDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);
    handleFiles(e.dataTransfer.files);
  };

  const onDragOver = (e) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const onDragLeave = (e) => {
    e.preventDefault();
    setIsDragging(false);
  };

  return (
    <div>
      <div
        onClick={() => inputRef.current?.click()}
        onDrop={onDrop}
        onDragOver={onDragOver}
        onDragLeave={onDragLeave}
        style={{
          border: `2px dashed ${isDragging ? '#667eea' : '#d1d5db'}`,
          borderRadius: 12,
          padding: '40px 24px',
          textAlign: 'center',
          background: isDragging ? '#f3f4f6' : '#ffffff',
          cursor: 'pointer',
          transition: 'all 0.2s ease',
          position: 'relative',
          overflow: 'hidden',
        }}
      >
        {loading ? (
          <>
            <div style={{ fontSize: 48, marginBottom: 16 }}>⏳</div>
            <p style={{ margin: 0, fontSize: 16, fontWeight: 500, color: '#374151' }}>Processing timetable...</p>
            <p style={{ margin: '8px 0 0 0', fontSize: 14, color: '#6b7280' }}>This may take a few moments</p>
          </>
        ) : (
          <>
            <div style={{ fontSize: 48, marginBottom: 16 }}>📎</div>
            <p style={{ margin: 0, fontSize: 16, fontWeight: 500, color: '#374151', marginBottom: 8 }}>
              {isDragging ? 'Drop file here' : 'Drag & drop or click to browse'}
            </p>
            <p style={{ margin: 0, fontSize: 14, color: '#6b7280' }}>
              {fileName || 'PNG, JPEG, or PDF files'}
            </p>
            {fileName && (
              <div
                style={{
                  marginTop: 12,
                  padding: '8px 12px',
                  background: '#eff6ff',
                  borderRadius: 6,
                  display: 'inline-block',
                  fontSize: 13,
                  color: '#2563eb',
                }}
              >
                ✓ {fileName}
              </div>
            )}
          </>
        )}
        <input
          ref={inputRef}
          type="file"
          accept=".png,.jpg,.jpeg,.pdf"
          style={{ display: 'none' }}
          onChange={(e) => handleFiles(e.target.files)}
        />
      </div>
    </div>
  );
}

export default ImageUpload;
