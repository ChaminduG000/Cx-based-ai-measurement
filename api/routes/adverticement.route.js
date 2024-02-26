import express from "express";
import {
  approveAdverticement,
  createAdverticement,
  deleteAdverticement,
  editAdverticement,
  getAdverticement,
  getAllAdverticements,
  getUserAllAdverticements,
  rejectAdverticement,
} from "../controller/adverticement.controller.js";

const router = express.Router();

router.post("/", createAdverticement);
router.put("/ads/:advertisementId", editAdverticement);
router.get("/:id", getAdverticement);
router.get("/user/:userId", getUserAllAdverticements);
router.get("/", getAllAdverticements);
router.put("/approve/:advertiseId", approveAdverticement);
router.put("/reject/:advertiseId", rejectAdverticement);
router.delete("/:id", deleteAdverticement);
export default router;
