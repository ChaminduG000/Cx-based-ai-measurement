import express from "express";
import { getAuditLogEntries } from "../controller/auditLog.controller.js"; 

const router = express.Router();

// Define a route to retrieve audit log entries
router.get("/", getAuditLogEntries);

export default router;
