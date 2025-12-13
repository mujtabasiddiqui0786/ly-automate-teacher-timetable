import React, { useEffect, useState } from 'react';
import axios from 'axios';
import ImageUpload from './components/ImageUpload.jsx';
import TimetableDisplay from './components/TimetableDisplay.jsx';
import TimetableEditor from './components/TimetableEditor.jsx';
import TenantSelector from './components/TenantSelector.jsx';
import TeacherForm from './components/TeacherForm.jsx';
import TeacherList from './components/TeacherList.jsx';

function App() {
  const [tenantSlug, setTenantSlug] = useState('default');
  const [teachers, setTeachers] = useState([]);
  const [selectedTeacher, setSelectedTeacher] = useState(null);
  const [timetable, setTimetable] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  useEffect(() => {
    async function bootstrap() {
      setError('');
      try {
        if (!tenantSlug) return;
        await axios.post('/api/tenants', { name: tenantSlug, slug: tenantSlug });
        const { data } = await axios.get(`/api/tenants/${tenantSlug}/teachers`);
        setTeachers(data);
        setSelectedTeacher(null);
        setTimetable(null);
      } catch (err) {
        setError(err.response?.data?.error || 'Failed to load tenant/teachers');
      }
    }
    bootstrap();
  }, [tenantSlug]);

  const handleCreateTeacher = async (teacher) => {
    setError('');
    try {
      await axios.post(`/api/tenants/${tenantSlug}/teachers`, teacher);
      const { data } = await axios.get(`/api/tenants/${tenantSlug}/teachers`);
      setTeachers(data);
    } catch (err) {
      setError(err.response?.data?.error || 'Failed to create teacher');
    }
  };

  const handleSelectTeacher = async (teacher) => {
    setSelectedTeacher(teacher);
    setTimetable(null);
    setError('');
    try {
      const { data } = await axios.get(`/api/tenants/${tenantSlug}/teachers/${teacher._id}/timetable`);
      setTimetable(data);
    } catch (err) {
      // ok if not found yet
      if (err.response?.status !== 404) {
        setError(err.response?.data?.error || 'Failed to load timetable');
      }
    }
  };

  const handleUpload = async (file) => {
    if (!selectedTeacher) {
      setError('Select a teacher first');
      return;
    }
    setLoading(true);
    setError('');
    try {
      const formData = new FormData();
      formData.append('file', file);
      const { data } = await axios.post(
        `/api/tenants/${tenantSlug}/teachers/${selectedTeacher._id}/timetable/extract`,
        formData,
        { headers: { 'Content-Type': 'multipart/form-data' } }
      );
      setTimetable(data);
    } catch (err) {
      setError(err.response?.data?.error || 'Failed to extract timetable');
    } finally {
      setLoading(false);
    }
  };

  const handleSave = async (updatedTimetable) => {
    setTimetable(updatedTimetable);
  };

  return (
    <main style={{ maxWidth: 1280, margin: '0 auto', padding: 24, fontFamily: 'Inter, sans-serif' }}>
      <h1>Teacher Timetable Automation</h1>
      <p style={{ color: '#555' }}>
        Multi-tenant: onboard teachers, upload timetables (image/PDF), and view teacher-wise schedules.
      </p>

      <TenantSelector value={tenantSlug} onChange={setTenantSlug} />

      <section style={{ display: 'grid', gridTemplateColumns: '2fr 3fr', gap: 16, marginTop: 12 }}>
        <div>
          <TeacherForm onCreate={handleCreateTeacher} />
          <TeacherList teachers={teachers} onSelect={handleSelectTeacher} selectedId={selectedTeacher?._id} />
        </div>
        <div>
          <ImageUpload onUpload={handleUpload} loading={loading} />
        </div>
      </section>

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

