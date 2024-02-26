import AuditLog from "../models/auditLog.model.js"; // Import your AuditLog model

export const getAuditLogEntries = async (req, res) => {
  try {
    const auditLogEntries = await AuditLog.find().sort({ timestamp: -1 }); // Retrieve audit log entries and sort by timestamp in descending order
    res.status(200).json(auditLogEntries);
  } catch (error) {
    console.error("Error retrieving audit log entries:", error);
    res.status(500).json({ error: "Failed to retrieve audit log entries" });
  }
};
