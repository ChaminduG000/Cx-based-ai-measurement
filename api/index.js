import { configDotenv } from "dotenv";
import express from "express";
import mongoose from "mongoose";
import userRoute from "./routes/user.route.js";
import adsRoute from "./routes/adverticement.route.js";
import feedbackRoute from "./routes/feedback.route.js";
import emotion_counts from "./routes/emotion_counts.route.js";
import auditLogRoutes from "./routes/auditLog.routes.js"
import authRoute from "./routes/auth.route.js";
import camersRoute from "./routes/camera.route.js";
import cookieParser from "cookie-parser";
import cors from "cors";

const app = express();
configDotenv();

const connect = async () => {
  try {
    await mongoose.connect(process.env.MONGO);
    console.log("Connected to mongodb");
  } catch (error) {
    console.log(error);
  }
};
const corsOptions = {
  origin: "http://localhost:3000",
  credentials: true, // Allow credentials (cookies)
};

app.use(cors(corsOptions));
app.use(express.json());
app.use(cookieParser());

app.use("/api/auth", authRoute);
app.use("/api/users", userRoute);
app.use("/api/adverticements", adsRoute);
app.use("/api/feedbacks", feedbackRoute);
app.use("/api/emotions", emotion_counts);
app.use("/api/audit-logs", auditLogRoutes);
app.use("/api/camera", camersRoute);

app.use((err, req, res, next) => {
  const errorStatus = err.status || 500;
  const errorMessage = err.message || "Something went wrong!";
  return res.status(errorStatus).send(errorMessage);
});

app.listen(8800, () => {
  connect();
  console.log("Backend server is running");
});
