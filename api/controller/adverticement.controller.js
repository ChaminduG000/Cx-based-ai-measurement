import moment from "moment-timezone";
import Advertisement from "../models/adverticement.model.js";
import AuditLog from "../models/auditLog.model.js";
export const createAdverticement = async (req, res) => {
  try {
    // Extract the scheduleDate, scheduleTime, endScheduleDate, and endScheduleTime from the request body
    const { scheduleDate, scheduleTime, endScheduleDate, endScheduleTime, cameras, ...restOfData } = req.body;

    // Combine scheduleDate and scheduleTime to create a valid Date object for local time
    const scheduleDateTimeLocal = moment.tz(`${scheduleDate}T${scheduleTime}`, 'YYYY-MM-DDTHH:mm:ss.SSS', 'YOUR_TIMEZONE').toDate();

    // Combine endScheduleDate and endScheduleTime to create a valid Date object for local time
    const endScheduleDateTimeLocal = moment.tz(`${endScheduleDate}T${endScheduleTime}`, 'YYYY-MM-DDTHH:mm:ss.SSS', 'YOUR_TIMEZONE').toDate();

    // Create a new Advertisement document with the local time values
    const newAdvertisement = new Advertisement({
      ...restOfData, 
      cameras,
      scheduleDateTime: scheduleDateTimeLocal,
      endScheduleDateTime: endScheduleDateTimeLocal,
    });
console.log(newAdvertisement)
    const savedAdvertisement = await newAdvertisement.save();

     // Log the action in the audit log
     const auditLogEntry = new AuditLog({
      action: "Create Advertisement",
      user: req.body.userId, // Assuming you have user information in the request
      details: `Created Advertisement with ID: ${savedAdvertisement._id}`,
    });

    await auditLogEntry.save();

    res.status(201).json(savedAdvertisement);
    
  } catch (error) {
    console.error("Error creating advertisement:", error);
    res.status(500).json({ error: "Failed to create advertisement" });
  }
};

export const editAdverticement = async (req, res) => {
  const { advertisementId } = req.params;
  const updateFields = req.body;
console.log(updateFields)
  try {
    // Extract the scheduleDate, scheduleTime, endScheduleDate, and endScheduleTime from the updateFields
    const { scheduleDate, scheduleTime, endScheduleDate, endScheduleTime, ...restOfData } = updateFields;

    // Combine scheduleDate and scheduleTime to create a valid Date object for local time
    const scheduleDateTimeLocal = moment.tz(`${scheduleDate}T${scheduleTime}`, 'YYYY-MM-DDTHH:mm:ss.SSS', 'YOUR_TIMEZONE').toDate();

    // Combine endScheduleDate and endScheduleTime to create a valid Date object for local time
    const endScheduleDateTimeLocal = moment.tz(`${endScheduleDate}T${endScheduleTime}`, 'YYYY-MM-DDTHH:mm:ss.SSS', 'YOUR_TIMEZONE').toDate();

    // Create an object with the fields you want to update
    const updatedFields = {
      ...restOfData,
      scheduleDateTime: scheduleDateTimeLocal,
      endScheduleDateTime: endScheduleDateTimeLocal,
    };

    const updatedAdvertisement = await Advertisement.findByIdAndUpdate(
      advertisementId,
      { $set: updatedFields }, // Use $set to update the specified fields
      { new: true }
    );

    if (!updatedAdvertisement) {
      return res.status(404).json({ error: "Advertisement not found" });
    }

    // Log the action in the audit log
    const auditLogEntry = new AuditLog({
      action: "Edit Advertisement",
      user: req.body.userId,  // Assuming you have user information in the request
      details: `Edited Advertisement with ID: ${advertisementId}`,
    });

    await auditLogEntry.save();

    res.status(200).json(updatedAdvertisement);
  } catch (error) {
    console.error("Error updating advertisement:", error);
    res.status(500).json({ error: "Failed to update advertisement" });
  }
};


export const getAdverticement = async (req, res) => {
  try {
    const adverticement = await Advertisement.findById(req.params.id);
    res.status(200).send(adverticement);
  } catch (error) {
    res.status(500).send("Not Found");
  }
};
export const getUserAllAdverticements = async (req, res) => {
  try {
    const { userId } = req.params; // Assuming you pass the user ID as a parameter
    console.log(userId)
    const advertisements = await Advertisement.find({userId}); // Filter by user ID
    res.status(200).json(advertisements);
  } catch (error) {
    res.status(500).json({ error: "Failed to retrieve advertisements" });
  }
};
export const getAllAdverticements = async (req, res) => {
  try {
    const adverticement = await Advertisement.find();
    res.status(200).send(adverticement);
  } catch (error) {
    res.status(500).send("Not Found");
  }
};
export const deleteAdverticement = async (req, res) => {
  try {
    const advertisement = await Advertisement.findByIdAndDelete(req.params.id);

    if (!advertisement) {
      return res.status(404).json({ message: "Advertisement not found" });
    }

    // Log the action in the audit log
    const auditLogEntry = new AuditLog({
      action: "Delete Advertisement",
      user: advertisement.userId, // Assuming you have user information in the request
      details: `Deleted Advertisement with ID: ${advertisement._id}`,
    });

    await auditLogEntry.save();

    res.status(200).json({ message: "Advertisement deleted successfully" });
  } catch (error) {
    console.error("Error deleting advertisement:", error);
    res.status(500).send("Not found");
  }
};
export const approveAdverticement = async (req, res) => {
  const { advertiseId } = req.params;
console.log(advertiseId)
  try {
    // Find the advertisement by ID and update the status to true
    const advertisement = await Advertisement.findByIdAndUpdate(
      advertiseId,
      { status: true },
      { new: true }
    );
 // Log the action in the audit log
 const auditLogEntry = new AuditLog({
  action: "Approve Advertisement",
  user: advertisement.userId, // Assuming you have user information in the request
  details: `Approved Advertisement with ID: ${advertisement._id}`,
});

await auditLogEntry.save();
    res.json({ message: 'Advertisement approved successfully', advertisement });
  } catch (error) {
    console.error('Error approving advertisement:', error);
    res.status(500).json({ error: 'Internal server error' });
  }
};
export const rejectAdverticement = async (req, res) => {
  const { advertiseId } = req.params;
console.log(advertiseId)
  try {
    // Find the advertisement by ID and update the status to true
    const advertisement = await Advertisement.findByIdAndUpdate(
      advertiseId,
      { status: false },
      { new: true }
    );
 // Log the action in the audit log
 const auditLogEntry = new AuditLog({
  action: "Reject Advertisement",
  user: advertisement.userId, // Assuming you have user information in the request
  details: `Rejected Advertisement with ID: ${advertisement._id}`,
});

await auditLogEntry.save();
    res.json({ message: 'Advertisement approved successfully', advertisement });
  } catch (error) {
    console.error('Error approving advertisement:', error);
    res.status(500).json({ error: 'Internal server error' });
  }
};

