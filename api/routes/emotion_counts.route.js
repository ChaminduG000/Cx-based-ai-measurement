import express from "express";
import {
  getAllEmotionalCounts,
  getEmotionsForAdvertisement,
  getEmotionsUsersForAdvertisement,
} from "../controller/emotion_counts.controller.js";

const router = express.Router();
router.get("/", getAllEmotionalCounts);
router.get("/:advertisementId", getEmotionsForAdvertisement);
router.get("/users/:userId", getEmotionsUsersForAdvertisement);

export default router;
