import mongoose from "mongoose";
const { Schema } = mongoose;

const feedbackSchema = new Schema(
  {
    emotion_counts: {
    type:{
      anger: Number,
      contempt: Number,
      disgust: Number,
      fear: Number,
      happy: Number,
      neutral: Number,
      sad: Number,
      surprise: Number
    }
    },
    advertisement: {
      type: mongoose.Schema.Types.ObjectId,
      ref: 'Advertisement' // Reference to the Advertisement model
    }
  },
  {
    timestamps: true, // Corrected from 'timesTamps' to 'timestamps'
  }
);
export default mongoose.model("Feedbacks", feedbackSchema);
