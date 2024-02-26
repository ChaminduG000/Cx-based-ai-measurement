import Cameras from "../models/cameras.model.js";

export const createCameras = async (req, res) => {
    try {
        // Extracting data from the request body
        const { name, ip } = req.body;
    
        // Creating a new camera instance
        const newCamera = new Cameras({
          name,
          ip,
        });
    
        // Saving the new camera to the database
        const createdCamera = await newCamera.save();
    
        res.status(201).json(createdCamera);
      } catch (error) {
        // Handling errors
        console.error("Error creating camera:", error);
        res.status(500).json({ error: "Internal Server Error" });
      }
}
export const getCameras = async (req, res) => {
  try {
    // Fetch all cameras from the database
    const cameras = await Cameras.find();

    res.status(200).json(cameras);
  } catch (error) {
    // Handling errors
    console.error("Error fetching cameras:", error);
    res.status(500).json({ error: "Internal Server Error" });
  }
};