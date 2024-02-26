import React, { useState } from "react";
import Sidebar from "../../components/sidebar/Sidebar";
import Navbar from "../../components/navbar/Navbar";
import CircularProgress from "@mui/material/CircularProgress";
import "./newCamera.scss";
import uploadFirebase from "../../utils/uploadFirebase";
import axios from "axios";
import { useNavigate } from "react-router-dom";

const NewCamera = () => {
  const [file, setFile] = useState(null);

  const [camera, setCamera] = useState({
    name: "",
    ip: "",
  });

  const [errors, setErrors] = useState({});
  const [uploadProgress, setUploadProgress] = useState(0);
  const [uploading, setUploading] = useState(false);
  const navigate = useNavigate();

  const handleChange = (e) => {
    setCamera((prev) => {
      return { ...prev, [e.target.name]: e.target.value };
    });
  };

  const validateForm = () => {
    const newErrors = {};

    if (!camera.name) {
      newErrors.name = "Camera name is required";
    }
    if (!camera.ip) {
      newErrors.ip = "Camera IP is required";
    }
    // Add other validation checks as needed

    setErrors(newErrors);

    return Object.keys(newErrors).length === 0;
  };

  const handleSubmit = async (e) => {
    e.preventDefault();

    if (!validateForm() || uploading) {
      return;
    }

    try {
      setUploading(true);

      // You can upload the file if needed
      // const url = await uploadFirebase(file, (progressEvent) => {
      //   const percentCompleted = Math.round(
      //     (progressEvent.loaded * 100) / progressEvent.total
      //   );
      //   setUploadProgress(percentCompleted);
      // });

      // Create a new camera in the database
      await axios.post("http://localhost:8800/api/camera/", camera);
console.log(camera)
      navigate("/cameras");
      setCamera({
        name: "",
        ip: "",
      });
      setFile(null);
      setUploadProgress(0);
      setUploading(false);
    } catch (error) {
      console.error("Error uploading data:", error);
      setUploading(false);
    }
  };

  return (
    <div className="new">
      <Sidebar />
      <div className="newContainer">
        <Navbar />
        <div className="top">
          <h1>Add New Camera</h1>
        </div>
        <div className="bottom">
          <div className="right">
            <form onSubmit={handleSubmit}>
              <div className="formInput">
                <label>Camera Name</label>
                <input
                  type="text"
                  name="name"
                  placeholder="Enter"
                  value={camera.name}
                  onChange={handleChange}
                />
                {errors.name && <span className="error">{errors.name}</span>}
              </div>
              <div className="formInput">
                <label>Camera IP</label>
                <input
                  type="text"
                  name="ip"
                  placeholder="Enter"
                  value={camera.ip}
                  onChange={handleChange}
                />
                {errors.ip && <span className="error">{errors.ip}</span>}
              </div>
              <div className="formInput">
                {uploading ? (
                  <div className="formInputButton">
                    <div>
                      <CircularProgress size={34} thickness={5} />
                    </div>
                  </div>
                ) : (
                  <div className="formInputButton">
                    <button type="submit">
                      {uploading ? "Uploading..." : "Send"}
                    </button>
                  </div>
                )}
              </div>
            </form>
          </div>
        </div>
      </div>
    </div>
  );
};

export default NewCamera;
