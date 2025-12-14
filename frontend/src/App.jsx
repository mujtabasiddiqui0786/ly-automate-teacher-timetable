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

function Toast({ message, type, onClose }) {
  if (!message) return null;
  const bg = type === 'error' ? '#fef2f2' : type === 'success' ? '#f0fdf4' : '#eff6ff';
  const color = type === 'error' ? '#dc2626' : type === 'success' ? '#16a34a' : '#2563eb';
  const borderColor = type === 'error' ? '#fecaca' : type === 'success' ? '#bbf7d0' : '#bfdbfe';
  const icon = type === 'error' ? '✕' : type === 'success' ? '✓' : 'ℹ';

  return (
    <div
      style={{
        position: 'fixed',
        top: 20,
        right: 20,
        padding: '14px 18px',
        borderRadius: 12,
        background: bg,
        color,
        border: `1px solid ${borderColor}`,
        boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06)',
        display: 'flex',
        alignItems: 'center',
        gap: 12,
        minWidth: 280,
        maxWidth: 400,
        zIndex: 1000,
        animation: 'slideIn 0.3s ease-out',
      }}
    >
      <span style={{ fontSize: 18 }}>{icon}</span>
      <span style={{ flex: 1, fontSize: 14, fontWeight: 500 }}>{message}</span>
      {onClose && (
        <button
          onClick={onClose}
          style={{
            background: 'none',
            border: 'none',
            color: color,
            cursor: 'pointer',
            fontSize: 18,
            padding: 0,
            width: 24,
            height: 24,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            opacity: 0.7,
          }}
        >
          ×
        </button>
      )}
      <style>{`
        @keyframes slideIn {
          from {
            transform: translateX(100%);
            opacity: 0;
          }
          to {
            transform: translateX(0);
            opacity: 1;
          }
        }
      `}</style>
    </div>
  );
}

function Header({ tenantSlug, onTenantChange }) {
  return (
    <header
      style={{
        background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
        color: 'white',
        padding: '20px 0',
        boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)',
        marginBottom: 32,
      }}
    >
      <div style={{ maxWidth: 1400, margin: '0 auto', padding: '0 32px' }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: 16 }}>
          <div>
            <h1 style={{ margin: 0, fontSize: 28, fontWeight: 700, letterSpacing: '-0.5px' }}>
              📅 Teacher Timetable Automation
            </h1>
            <p style={{ margin: '4px 0 0 0', fontSize: 14, opacity: 0.9 }}>
              Intelligent timetable extraction and management system
            </p>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 16 }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8, background: 'rgba(255, 255, 255, 0.15)', padding: '8px 16px', borderRadius: 8 }}>
              <span style={{ fontSize: 14, opacity: 0.9 }}>Tenant:</span>
              <span style={{ fontWeight: 600, fontSize: 15 }}>{tenantSlug || 'Not selected'}</span>
            </div>
          </div>
        </div>
      </div>
    </header>
  );
}

function Card({ title, icon, children, style = {} }) {
  return (
    <div
      style={{
        background: '#ffffff',
        borderRadius: 16,
        padding: 24,
        boxShadow: '0 1px 3px 0 rgba(0, 0, 0, 0.1), 0 1px 2px 0 rgba(0, 0, 0, 0.06)',
        border: '1px solid #e5e7eb',
        ...style,
      }}
    >
      {title && (
        <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 20 }}>
          {icon && <span style={{ fontSize: 20 }}>{icon}</span>}
          <h2 style={{ margin: 0, fontSize: 18, fontWeight: 600, color: '#111827' }}>{title}</h2>
        </div>
      )}
      {children}
    </div>
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
      setToast({ message: 'Teacher created successfully!', type: 'success' });
      setTimeout(() => setToast({ message: '', type: 'info' }), 3000);
    } catch (err) {
      setToast({ message: err.response?.data?.error || 'Failed to create teacher', type: 'error' });
      setTimeout(() => setToast({ message: '', type: 'info' }), 5000);
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
        setTimeout(() => setToast({ message: '', type: 'info' }), 5000);
      }
    }
  };

  const handleUpload = async (file) => {
    if (!selectedTeacher) {
      setToast({ message: 'Please select a teacher first', type: 'error' });
      setTimeout(() => setToast({ message: '', type: 'info' }), 3000);
      return;
    }
    if (!file) {
      setToast({ message: 'Please select a file', type: 'error' });
      setTimeout(() => setToast({ message: '', type: 'info' }), 3000);
      return;
    }
    const allowed = ['image/png', 'image/jpeg', 'application/pdf'];
    if (!allowed.includes(file.type)) {
      setToast({ message: 'Only PDF, PNG, and JPEG files are allowed', type: 'error' });
      setTimeout(() => setToast({ message: '', type: 'info' }), 3000);
      return;
    }
    if (file.size > 15 * 1024 * 1024) {
      setToast({ message: 'File too large (max 15MB)', type: 'error' });
      setTimeout(() => setToast({ message: '', type: 'info' }), 3000);
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
      setToast({ message: 'Timetable extracted successfully!', type: 'success' });
      setTimeout(() => setToast({ message: '', type: 'info' }), 3000);
    } catch (err) {
      setToast({ message: err.response?.data?.error || 'Failed to extract timetable', type: 'error' });
      setTimeout(() => setToast({ message: '', type: 'info' }), 5000);
    } finally {
      setUploading(false);
    }
  };

  const handleSave = async (updatedTimetable) => {
    setTimetable(updatedTimetable);
    setToast({ message: 'Timetable updated successfully!', type: 'success' });
    setTimeout(() => setToast({ message: '', type: 'info' }), 3000);
  };

  return (
    <div style={{ minHeight: '100vh', background: '#f9fafb' }}>
      <style>{`
        * {
          box-sizing: border-box;
        }
        body {
          margin: 0;
          font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Oxygen', 'Ubuntu', 'Cantarell', 'Fira Sans', 'Droid Sans', 'Helvetica Neue', sans-serif;
          -webkit-font-smoothing: antialiased;
          -moz-osx-font-smoothing: grayscale;
        }
      `}</style>
      <Header tenantSlug={tenantSlug} onTenantChange={setTenantSlug} />
      
      <main style={{ maxWidth: 1400, margin: '0 auto', padding: '0 32px 48px' }}>
        {/* Top Section: Tenant & Upload */}
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 24, marginBottom: 24 }}>
          <Card title="Organization" icon="🏢">
            <TenantSelector value={tenantSlug} onChange={setTenantSlug} />
          </Card>
          
          <Card title="Upload Timetable" icon="📤">
            <ImageUpload onUpload={handleUpload} loading={uploading} />
            <p style={{ color: '#6b7280', fontSize: 13, marginTop: 12, marginBottom: 0 }}>
              Supported formats: PDF, PNG, JPEG • Maximum size: 15MB
            </p>
          </Card>
        </div>

        {/* Main Content: Teachers & Timetable */}
        <div style={{ display: 'grid', gridTemplateColumns: '1.2fr 2fr', gap: 24 }}>
          {/* Left: Teachers */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: 24 }}>
            <Card title="Teachers" icon="👥">
              <TeacherForm onCreate={handleCreateTeacher} />
              <div style={{ marginTop: 20 }}>
                <TeacherList
                  teachers={teachers}
                  loading={teacherLoading}
                  onSelect={handleSelectTeacher}
                  selectedId={selectedTeacher?._id}
                />
              </div>
            </Card>
          </div>

          {/* Right: Teacher Details & Timetable */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: 24 }}>
            <Card title="Teacher Details" icon="👤">
              {selectedTeacher ? (
                <div>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 12 }}>
                    <div
                      style={{
                        width: 48,
                        height: 48,
                        borderRadius: '50%',
                        background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        color: 'white',
                        fontSize: 20,
                        fontWeight: 600,
                      }}
                    >
                      {selectedTeacher.name.charAt(0).toUpperCase()}
                    </div>
                    <div>
                      <div style={{ fontWeight: 600, fontSize: 18, color: '#111827', marginBottom: 4 }}>
                        {selectedTeacher.name}
                      </div>
                      <div style={{ color: '#6b7280', fontSize: 14 }}>
                        {selectedTeacher.email || 'No email'} {selectedTeacher.grade ? `• ${selectedTeacher.grade}` : ''}
                      </div>
                    </div>
                  </div>
                  {selectedTeacher.subjects?.length ? (
                    <div style={{ marginTop: 16, padding: 12, background: '#f3f4f6', borderRadius: 8 }}>
                      <div style={{ fontSize: 13, fontWeight: 500, color: '#374151', marginBottom: 8 }}>Subjects</div>
                      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
                        {selectedTeacher.subjects.map((subject, idx) => (
                          <span
                            key={idx}
                            style={{
                              padding: '4px 12px',
                              background: '#ffffff',
                              borderRadius: 6,
                              fontSize: 12,
                              color: '#4b5563',
                              border: '1px solid #e5e7eb',
                            }}
                          >
                            {subject}
                          </span>
                        ))}
                      </div>
                    </div>
                  ) : (
                    <div style={{ marginTop: 16, padding: 12, background: '#f9fafb', borderRadius: 8, color: '#9ca3af', fontSize: 13 }}>
                      No subjects assigned
                    </div>
                  )}
                </div>
              ) : (
                <div style={{ textAlign: 'center', padding: '32px 0', color: '#9ca3af' }}>
                  <div style={{ fontSize: 48, marginBottom: 12 }}>👤</div>
                  <p style={{ margin: 0, fontSize: 14 }}>Select a teacher to view details</p>
                </div>
              )}
            </Card>

            <Card title="Timetable" icon="📋">
              {!selectedTeacher && (
                <div style={{ textAlign: 'center', padding: '48px 0', color: '#9ca3af' }}>
                  <div style={{ fontSize: 48, marginBottom: 12 }}>📅</div>
                  <p style={{ margin: 0, fontSize: 14 }}>Select a teacher to view timetable</p>
                </div>
              )}
              {selectedTeacher && !hasTimetable && (
                <div style={{ textAlign: 'center', padding: '48px 0', color: '#9ca3af' }}>
                  <div style={{ fontSize: 48, marginBottom: 12 }}>📤</div>
                  <p style={{ margin: 0, fontSize: 14 }}>Upload a timetable image or PDF to get started</p>
                </div>
              )}
              {selectedTeacher && timetable && (
                <>
                  <TimetableDisplay timetable={timetable} />
                  <div style={{ marginTop: 24 }}>
                    <TimetableEditor timetable={timetable} onSave={handleSave} />
                  </div>
                </>
              )}
            </Card>
          </div>
        </div>
      </main>

      {toast.message && (
        <Toast
          message={toast.message}
          type={toast.type}
          onClose={() => setToast({ message: '', type: 'info' })}
        />
      )}
    </div>
  );
}

export default App;
