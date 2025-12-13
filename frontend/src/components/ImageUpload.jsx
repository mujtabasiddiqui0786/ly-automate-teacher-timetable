import React, { useRef, useState } from 'react';

const dropZoneStyle = {
  border: '2px dashed #ccc',
  padding: 24,
  borderRadius: 8,
  textAlign: 'center',
  marginBottom: 16,
};

function ImageUpload({ onUpload, loading }) {
  const inputRef = useRef(null);
  const [fileName, setFileName] = useState('');

  const handleFiles = (files) => {
    const file = files?.[0];
    if (!file) return;
    setFileName(file.name);
    onUpload(file);
  };

  const onDrop = (e) => {
    e.preventDefault();
    handleFiles(e.dataTransfer.files);
  };

  return (
    <section>
      <div
        style={dropZoneStyle}
        onDrop={onDrop}
        onDragOver={(e) => e.preventDefault()}
        onClick={() => inputRef.current?.click()}
      >
        <p>Drag & drop a timetable image/PDF or click to browse.</p>
        <p style={{ color: '#666', fontSize: 14 }}>Supported: PNG, JPEG, PDF</p>
        <input
          ref={inputRef}
          type="file"
          accept=".png,.jpg,.jpeg,.pdf"
          style={{ display: 'none' }}
          onChange={(e) => handleFiles(e.target.files)}
        />
        {loading && <p>Extracting...</p>}
        {fileName && !loading && <p>Selected: {fileName}</p>}
      </div>
    </section>
  );
}

export default ImageUpload;

