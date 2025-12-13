import mongoose from 'mongoose';

const teacherSchema = new mongoose.Schema(
  {
    tenantId: { type: mongoose.Schema.Types.ObjectId, ref: 'Tenant', required: true, index: true },
    name: { type: String, required: true },
    email: { type: String, trim: true },
    subjects: [{ type: String }],
    grade: { type: String },
    meta: { type: Object },
  },
  { timestamps: true }
);

export const Teacher = mongoose.model('Teacher', teacherSchema);

