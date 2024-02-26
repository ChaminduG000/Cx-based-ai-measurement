import express from "express";
import { createCameras, getCameras } from "../controller/camera.controller.js";



const router = express.Router();

router.post("/",createCameras)
router.get("/", getCameras);
export default router;
