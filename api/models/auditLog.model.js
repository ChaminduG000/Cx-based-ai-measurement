// auditLog.model.js

import mongoose from "mongoose";

const auditLogSchema = new mongoose.Schema({
  timestamp: {
    type: Date,
    default: Date.now,
  },
  action: {
    type: String,
    required: true,
  },
  user: {
    type: String, // You can store user information like username or user ID
    required: true,
  },
  details: {
    type: String,
  },
});

const AuditLog = mongoose.model("AuditLog", auditLogSchema);

export default AuditLog;
