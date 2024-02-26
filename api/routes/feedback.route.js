import express from "express";
import {
  generatedFeedback,
  getAdvertisementAllFeedbacks,
  getAllFeedbacks,
  getUserAllFeedbacks,
  getfeedback,
} from "../controller/feedback.controller.js";

const router = express.Router();

router.get("/:id", getfeedback);
router.get("/user/:userId", getUserAllFeedbacks);
router.get("/", getAllFeedbacks);
router.get("/adverticement/:advertismentId", getAdvertisementAllFeedbacks);
router.post("/", generatedFeedback);

export default router;
