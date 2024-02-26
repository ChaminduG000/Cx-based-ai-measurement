import React, { useEffect, useState } from "react";
import Sidebar from "../../components/sidebar/Sidebar";
import Navbar from "../../components/navbar/Navbar";
import "./editAdverticement.scss";
import axios from "axios";
import { useNavigate, useParams } from "react-router-dom";

const EditAdverticement = () => {
  const navigate = useNavigate();
  const { adverticementId } = useParams();

  const [formData, setFormData] = useState({
    title: "",
    scheduleDate: "",
    scheduleTime: "",
    endScheduleDate: "",
    endScheduleTime: "",
    video: "",
  });

  const [cameras, setCameras] = useState([]);
  const [cameraSelection, setCameraSelection] = useState({});
  const [errors, setErrors] = useState({});
  const currentUser = JSON.parse(localStorage.getItem("currentUser"));
  const [advertisement, setAdvertisement] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchAdvertisementDetails = async () => {
      try {
        const response = await axios.get(
          `http://localhost:8800/api/adverticements/${adverticementId}`
        );
        setAdvertisement(response.data);
        setLoading(false);

        setFormData({
          title: response.data.title,
          scheduleDate: response.data.scheduleDate,
          scheduleTime: response.data.scheduleTime,
          endScheduleDate: response.data.endScheduleDate,
          endScheduleTime: response.data.endScheduleTime,
          video: response.data.video,
        });
      } catch (error) {
        console.error("Error fetching advertisement details", error);
      }
    };

    const fetchCameras = async () => {
      try {
        const response = await axios.get("http://localhost:8800/api/camera/");
        const cameraData = response.data;
        setCameras(cameraData);

        const initialCameraSelection = {};
        cameraData.forEach((camera) => {
          initialCameraSelection[camera.name] = false;
        });
        setCameraSelection(initialCameraSelection);
      } catch (error) {
        console.error("Error fetching cameras:", error);
      }
    };

    fetchAdvertisementDetails();
    fetchCameras();
  }, [adverticementId]);

  if (loading) {
    return <div>Loading...</div>;
  }

  if (!advertisement) {
    return <div>No advertisement data found.</div>;
  }

  const handleChange = (e) => {
    setFormData((prev) => {
      return { ...prev, [e.target.name]: e.target.value };
    });
  };

  const handleCameraCheckboxChange = (camera) => {
    setCameraSelection((prev) => ({
      ...prev,
      [camera]: !prev[camera],
    }));
  };

  const validateForm = () => {
    const newErrors = {};

    if (!formData.title) {
      newErrors.title = "Title is required";
    }

    if (!formData.scheduleDate) {
      newErrors.scheduleDate = "Schedule Date is required";
    }

    if (!formData.scheduleTime) {
      newErrors.scheduleTime = "Schedule Time is required";
    }

    if (!formData.endScheduleDate) {
      newErrors.endScheduleDate = "End Schedule Date is required";
    }

    if (!formData.endScheduleTime) {
      newErrors.endScheduleTime = "End Schedule Time is required";
    }

    // Validate camera selection
    if (Object.values(cameraSelection).filter(Boolean).length === 0) {
      newErrors.cameras = "Select at least one camera";
    }

    setErrors(newErrors);

    return Object.keys(newErrors).length === 0;
  };

  const handleSubmit = async (e) => {
    e.preventDefault();

    if (!validateForm()) {
      return;
    }

    try {
      const selectedCameras = Object.keys(cameraSelection)
        .filter((camera) => cameraSelection[camera])
        .map((camera) => ({ name: camera }));

      await axios.put(
        `http://localhost:8800/api/adverticements/ads/${adverticementId}`,
        {
          ...formData,
          userId: currentUser._id,
          cameras: selectedCameras,
        }
      );

      navigate("/adverticement");
    } catch (error) {
      console.error("Error updating data:", error);
    }
  };
  return (
    <div className="new">
      <Sidebar />
      <div className="newContainer">
        <Navbar />
        <div className="top">
          <h1>Edit Advertisement</h1>
        </div>
        <div className="bottom">
          <div className="left">
            {formData.video ? (
              <video width="320" height="240" controls>
                <source src={formData.video} type="video/mp4" />
                Your browser does not support the video tag.
              </video>
            ) : (
              <img
                src="https://icon-library.com/images/upload_video_162306_9899.png"
                alt=""
              />
            )}
          </div>
          <div className="right">
            <form onSubmit={handleSubmit}>
              <div className="formInput">
                <label>Video Link</label>
                <input
                  type="text"
                  name="video"
                  placeholder="Enter"
                  defaultValue={formData.video}
                  onChange={handleChange}
                />
                {errors.video && <span className="error">{errors.video}</span>}
              </div>
              <div className="formInput">
                <label>Title</label>
                <input
                  type="text"
                  name="title"
                  placeholder="Enter"
                  value={formData.title}
                  onChange={handleChange}
                />
                {errors.title && <span className="error">{errors.title}</span>}
              </div>
              <div className="formInput">
                <label>Schedule Date</label>
                <input
                  type="date"
                  name="scheduleDate"
                  placeholder=""
                  value={formData.scheduleDate}
                  onChange={handleChange}
                />
                {errors.scheduleDate && (
                  <span className="error">{errors.scheduleDate}</span>
                )}
              </div>
              <div className="formInput">
                <label>Schedule Time</label>
                <input
                  type="time"
                  name="scheduleTime"
                  placeholder=""
                  value={formData.scheduleTime}
                  onChange={handleChange}
                />
                {errors.scheduleTime && (
                  <span className="error">{errors.scheduleTime}</span>
                )}
              </div>

              <div className="formInput">
                <label>End Schedule Date</label>
                <input
                  type="date"
                  name="endScheduleDate"
                  placeholder=""
                  value={formData.endScheduleDate}
                  onChange={handleChange}
                />
                {errors.endScheduleDate && (
                  <span className="error">{errors.endScheduleDate}</span>
                )}
              </div>
              <div className="formInput">
                <label>End Schedule Time</label>
                <input
                  type="time"
                  name="endScheduleTime"
                  placeholder=""
                  value={formData.endScheduleTime}
                  onChange={handleChange}
                />
                {errors.endScheduleTime && (
                  <span className="error">{errors.endScheduleTime}</span>
                )}
              </div>

              {/* Add checkboxes for camera selection */}
              <div className="CheckBoxes">
                {cameras.map((camera) => (
                  <div key={camera.name}>
                    <label>{camera.name} :</label>
                    <input
                      type="checkbox"
                      name={camera.name}
                      checked={cameraSelection[camera.name]}
                      onChange={() => handleCameraCheckboxChange(camera.name)}
                    />
                  </div>
                ))}
              </div>

              <button type="submit">Update</button>
            </form>
          </div>
        </div>
      </div>
    </div>
  );
};

export default EditAdverticement;
