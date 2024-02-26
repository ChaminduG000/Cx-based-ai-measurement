// auditLog.model.js

import mongoose from "mongoose";

const cameraSchema = new mongoose.Schema({
    name: {
    type: String,
    required: true,
    
  },
  ip: {
    type: String,
    required: true,
   
  }
 
});

const Cameras = mongoose.model("Cameras", cameraSchema);

export default Cameras;
