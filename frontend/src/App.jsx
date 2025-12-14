import React, { useEffect, useMemo, useState } from 'react';
import axios from 'axios';
import ImageUpload from './components/ImageUpload.jsx';
import TimetableDisplay from './components/TimetableDisplay.jsx';
import TimetableEditor from './components/TimetableEditor.jsx';
import TenantSelector from './components/TenantSelector.jsx';
import TeacherForm from './components/TeacherForm.jsx';
import TeacherList from './components/TeacherList.jsx';

const STORE_KEYS = {
  TENANT: 'tt_tenant',
  TEACHER: 'tt_teacher',
};

function Toast({ message, type }) {
  if (!message) return null;
  const bg = type === 'error' ? '#fee2e2' : '#e0f2fe';
  const color = type === 'error' ? '#b91c1c' : '#075985';
  return (
    <div
      style={{
        marginTop: 12,
        padding: '10px 12px',
        borderRadius: 8,
        background: bg,
        color,
        border: `1px solid ${type === 'error' ? '#fecaca' : '#bae6fd'}`,
      }}
    >
      {message}
    </div>
  );
}

function Section({ title, children }) {
  return (
    <section style={{ padding: 12, border: '1px solid #eee', borderRadius: 10, background: '#fff' }}>
      <h3 style={{ margin: '0 0 8px 0' }}>{title}</h3>
      {children}
    </section>
  );
}

function App() {
  const [tenantSlug, setTenantSlug] = useState(() => localStorage.getItem(STORE_KEYS.TENANT) || 'default');
  const [teachers, setTeachers] = useState([]);
  const [teacherLoading, setTeacherLoading] = useState(false);
  const [selectedTeacher, setSelectedTeacher] = useState(() => {
    const stored = localStorage.getItem(STORE_KEYS.TEACHER);
    return stored ? JSON.parse(stored) : null;
  });
  const [timetable, setTimetable] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [toast, setToast] = useState({ message: '', type: 'info' });
  const [error, setError] = useState('');

  const hasTimetable = useMemo(() => !!timetable, [timetable]);

  useEffect(() => {
    localStorage.setItem(STORE_KEYS.TENANT, tenantSlug || '');
  }, [tenantSlug]);

  useEffect(() => {
    async function bootstrap() {
      setError('');
      setTeacherLoading(true);
      setTimetable(null);
      try {
        if (!tenantSlug) return;
        await axios.post('/api/tenants', { name: tenantSlug, slug: tenantSlug });
        const { data } = await axios.get(`/api/tenants/${tenantSlug}/teachers`);
        setTeachers(data);
        // if the stored teacher is not part of the new list, clear it
        if (selectedTeacher && !data.find((t) => t._id === selectedTeacher._id)) {
          setSelectedTeacher(null);
          localStorage.removeItem(STORE_KEYS.TEACHER);
        }
      } catch (err) {
        setError(err.response?.data?.error || 'Failed to load tenant/teachers');
      } finally {
        setTeacherLoading(false);
      }
    }
    bootstrap();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [tenantSlug]);

  const handleCreateTeacher = async (teacher) => {
    setError('');
    try {
      await axios.post(`/api/tenants/${tenantSlug}/teachers`, teacher);
      const { data } = await axios.get(`/api/tenants/${tenantSlug}/teachers`);
      setTeachers(data);
      setToast({ message: 'Teacher created', type: 'info' });
    } catch (err) {
      setToast({ message: err.response?.data?.error || 'Failed to create teacher', type: 'error' });
    }
  };

  const handleSelectTeacher = async (teacher) => {
    setSelectedTeacher(teacher);
    localStorage.setItem(STORE_KEYS.TEACHER, JSON.stringify(teacher));
    setTimetable(null);
    setError('');
    try {
      const { data } = await axios.get(`/api/tenants/${tenantSlug}/teachers/${teacher._id}/timetable`);
      setTimetable(data);
    } catch (err) {
      if (err.response?.status !== 404) {
        setToast({ message: err.response?.data?.error || 'Failed to load timetable', type: 'error' });
      }
    }
  };

  const handleUpload = async (file) => {
    if (!selectedTeacher) {
      setToast({ message: 'Select a teacher first', type: 'error' });
      return;
    }
    if (!file) {
      setToast({ message: 'Please select a file', type: 'error' });
      return;
    }
    const allowed = ['image/png', 'image/jpeg', 'application/pdf'];
    if (!allowed.includes(file.type)) {
      setToast({ message: 'Only pdf/png/jpg are allowed', type: 'error' });
      return;
    }
    if (file.size > 15 * 1024 * 1024) {
      setToast({ message: 'File too large (max 15MB)', type: 'error' });
      return;
    }

    setUploading(true);
    setToast({ message: '', type: 'info' });
    try {
      const formData = new FormData();
      formData.append('file', file);
      const { data } = await axios.post(
        `/api/tenants/${tenantSlug}/teachers/${selectedTeacher._id}/timetable/extract`,
        formData,
        { headers: { 'Content-Type': 'multipart/form-data' } }
      );
      setTimetable(data);
      setToast({ message: 'Timetable uploaded and parsed', type: 'info' });
    } catch (err) {
      setToast({ message: err.response?.data?.error || 'Failed to extract timetable', type: 'error' });
    } finally {
      setUploading(false);
    }
  };

  const handleSave = async (updatedTimetable) => {
    setTimetable(updatedTimetable);
    setToast({ message: 'Timetable updated locally', type: 'info' });
  };

  return (
    <main style={{ maxWidth: 1280, margin: '0 auto', padding: 24, fontFamily: 'Inter, sans-serif' }}>
      <h1 style={{ marginBottom: 4 }}>Teacher Timetable Automation</h1>
      <p style={{ color: '#555', marginTop: 0 }}>
        Multi-tenant: onboard teachers, upload timetables (image/PDF), and view teacher-wise schedules.
      </p>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12, marginBottom: 12 }}>
        <Section title="Tenants">
          <TenantSelector value={tenantSlug} onChange={setTenantSlug} />
        </Section>
        <Section title="Upload / Re-run OCR">
          <ImageUpload onUpload={handleUpload} loading={uploading} />
          <p style={{ color: '#666', fontSize: 12, marginTop: 4 }}>
            Allowed: pdf, png, jpg • Max 15MB • Re-upload replaces latest timetable for the selected teacher.
          </p>
        </Section>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1.4fr 2fr', gap: 16 }}>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
          <Section title="Teachers">
            <TeacherForm onCreate={handleCreateTeacher} />
            <TeacherList
              teachers={teachers}
              loading={teacherLoading}
              onSelect={handleSelectTeacher}
              selectedId={selectedTeacher?._id}
            />
          </Section>
        </div>

        <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
          <Section title="Teacher Details">
            {selectedTeacher ? (
              <div>
                <div style={{ fontWeight: 700, fontSize: 16 }}>{selectedTeacher.name}</div>
                <div style={{ color: '#555', fontSize: 13 }}>
                  {selectedTeacher.email || 'No email'} • {selectedTeacher.grade || 'No grade'}
                </div>
                {selectedTeacher.subjects?.length ? (
                  <div style={{ marginTop: 6, color: '#444', fontSize: 13 }}>
                    Subjects: {selectedTeacher.subjects.join(', ')}
                  </div>
                ) : (
                  <div style={{ marginTop: 6, color: '#777', fontSize: 13 }}>No subjects set</div>
                )}
              </div>
            ) : (
              <p style={{ color: '#777' }}>Select a teacher to view details.</p>
            )}
          </Section>

          <Section title="Timetable">
            {!selectedTeacher && <p style={{ color: '#777' }}>Select a teacher to view timetable.</p>}
            {selectedTeacher && !hasTimetable && <p style={{ color: '#777' }}>No timetable uploaded yet.</p>}
            {selectedTeacher && timetable && (
              <>
                <TimetableDisplay timetable={timetable} />
                <TimetableEditor timetable={timetable} onSave={handleSave} />
              </>
            )}
          </Section>
        </div>
      </div>

      {error && <Toast message={error} type="error" />}
      {toast.message && <Toast message={toast.message} type={toast.type} />}
    </main>
  );
}

export default App;

