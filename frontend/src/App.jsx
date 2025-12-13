import React, { useState } from 'react';
import axios from 'axios';
import ImageUpload from './components/ImageUpload.jsx';
import TimetableDisplay from './components/TimetableDisplay.jsx';
import TimetableEditor from './components/TimetableEditor.jsx';

function App() {
  const [timetable, setTimetable] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const handleUpload = async (file) => {
    setLoading(true);
    setError('');
    try {
      const formData = new FormData();
      formData.append('file', file);
      const { data } = await axios.post('/api/timetable/extract', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });
      setTimetable(data.timetable);
    } catch (err) {
      setError(err.response?.data?.error || 'Failed to extract timetable');
    } finally {
      setLoading(false);
    }
  };

  const handleSave = async (updatedTimetable) => {
    if (!updatedTimetable?.id) {
      setTimetable(updatedTimetable);
      return;
    }
    try {
      const { data } = await axios.put(`/api/timetable/${updatedTimetable.id}`, updatedTimetable);
      setTimetable(data);
    } catch (err) {
      setError(err.response?.data?.error || 'Failed to save timetable');
    }
  };

  return (
    <main style={{ maxWidth: 1200, margin: '0 auto', padding: 24, fontFamily: 'Inter, sans-serif' }}>
      <h1>Teacher Timetable Automation</h1>
      <p style={{ color: '#555' }}>
        Upload a timetable image or PDF to extract days, time slots, and events into a digital format.
      </p>

      <ImageUpload onUpload={handleUpload} loading={loading} />
      {error && <p style={{ color: 'red' }}>{error}</p>}

      {timetable && (
        <>
          <TimetableDisplay timetable={timetable} />
          <TimetableEditor timetable={timetable} onSave={handleSave} />
        </>
      )}
    </main>
  );
}

export default App;

