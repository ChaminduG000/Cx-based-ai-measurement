import mongoose from "mongoose";
const { Schema } = mongoose;
import moment from "moment-timezone";

const adverticementSchema = new Schema(
  {
    userId: {
      type: String,
      required: true,
    },
    title: {
      type: String,
      required: true,
    },
    video: {
      type: String,
      required: true,
    },
    scheduleDateTime: {
      type: Date,
      required: true,
    },
    endScheduleDateTime: {
      type: Date,
      required: true,
    },
    status: {
      type: Boolean,
      default: false,
    },
    cameras: [
      {
        name: String,
        ip: String,
      }
    ],
  },
  {
    timestamps: true,
  }
);

export default mongoose.model("Adverticements", adverticementSchema);
