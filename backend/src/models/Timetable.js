import mongoose from 'mongoose';

const eventSchema = new mongoose.Schema(
  {
    day: String,
    start: String,
    end: String,
    subject: String,
    teacherName: String,
    isFixed: { type: Boolean, default: false },
    duration: Number,
    orientation: String,
    timeSlot: {
      start: String,
      end: String,
      label: String,
    },
    source: String,
  },
  { _id: false }
);

const timetableSchema = new mongoose.Schema(
  {
    tenantId: { type: mongoose.Schema.Types.ObjectId, ref: 'Tenant', required: true, index: true },
    teacherId: { type: mongoose.Schema.Types.ObjectId, ref: 'Teacher', required: true, index: true },
    term: String,
    week: String,
    days: [String],
    timeSlots: [
      {
        start: String,
        end: String,
        label: String,
        duration: Number,
      },
    ],
    events: [eventSchema],
    rawText: String,
    sourceImageMeta: Object,
  },
  { timestamps: true }
);

export const Timetable = mongoose.model('Timetable', timetableSchema);

