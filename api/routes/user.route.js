import express from "express";
import {
  deleteUser,
  getUserById,
  getUserCount,
  getUsersWithUserRole,
} from "../controller/user.controller.js";
import { verifyToken } from "../middleware/jwt.js";

const router = express.Router();

router.delete("/:id", deleteUser);
router.get("/:id", getUserById);
router.get("/", getUsersWithUserRole);
router.get("/count", getUserCount);
export default router;
