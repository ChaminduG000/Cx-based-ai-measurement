import mongoose from "mongoose";
const { Schema } = mongoose;

const emotionCountsSchema = new Schema({
  emotion_counts: {
    type: {
      anger: Number,
      contempt: Number,
      disgust: Number,
      fear: Number,
      happy: Number,
      neutral: Number,
      surprise: Number,
      sad: Number,
    },
    default: {
      anger: 0,
      contempt: 0,
      disgust: 0,
      fear: 0,
      happy: 0,
      neutral: 0,
      surprise: 0,
      sad: 0,
    },
  },
  advertisement_id: {
    type: Schema.Types.ObjectId,
    required: true,
  },
  userId:{
    type: String,
    required: false,
  },
  advertisement_title:{
    type: String,
    required: true,
  },
  timestamp: {
    type: String,
  },
});
export default mongoose.model("Emotion_counts", emotionCountsSchema);
