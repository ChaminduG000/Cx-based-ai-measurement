import React, { useState, useEffect } from "react";
import axios from "axios";
import { DataGrid } from "@mui/x-data-grid";
import { format } from "date-fns";

const ReactionTable = () => {
  const [data, setData] = useState([]);
  const [advertisements, setAdvertisements] = useState({});
  const currentUser = JSON.parse(localStorage.getItem("currentUser"));
  let apiUrl = "http://localhost:8800/api/emotions";
  if (currentUser.role === "user") {
    // If the user is a regular user, filter advertisements accordingly
    apiUrl = `http://localhost:8800/api/emotions/users/${currentUser._id}`;
  }
  useEffect(() => {
    const fetchFeedbackDetails = async () => {
      try {
        const response = await axios.get(apiUrl);
        setData(response.data);
      } catch (error) {
        console.error("Error fetching feedback details", error);
      }
    };

    fetchFeedbackDetails();
  }, [apiUrl]);
  console.log(data);

  const formattedData = data.map((item, index) => {
    // const createdAt = new Date(item.scheduleDateTime);
    // const date = format(createdAt, "yyyy-MM-dd");
    // const time = format(createdAt, "HH:mm:ss");

    return {
      id: index + 1,
      title: item.advertisement_title,

      anger: item.emotion_counts.anger,
      contempt: item.emotion_counts.contempt,
      disgust: item.emotion_counts.disgust,
      fear: item.emotion_counts.fear,
      happy: item.emotion_counts.happy,
      neutral: item.emotion_counts.neutral,
      surprise: item.emotion_counts.surprise,
      sad: item.emotion_counts.sad,
      dateTime: item.timestamp,
      // time,
    };
  });

  const userColumns = [
    { field: "id", headerName: "ID", width: 50 },
    { field: "title", headerName: "Title", width: 200 },
    { field: "anger", headerName: "Anger", width: 75 },
    { field: "contempt", headerName: "Contempt", width: 75 },
    { field: "disgust", headerName: "Disgust", width: 75 },
    { field: "fear", headerName: "Fear", width: 75 },
    { field: "happy", headerName: "Happy", width: 75 },
    { field: "neutral", headerName: "Neutral", width: 75 },
    { field: "surprise", headerName: "Surprise", width: 75 },
    { field: "sad", headerName: "Sad", width: 75 },
    { field: "dateTime", headerName: "Date & Time", width: 250 },
    // { field: "time", headerName: "Time", width: 100 },
  ];

  return (
    <div className="datatable">
      <div className="datatableTitle">Reactions History</div>
      <DataGrid
        className="datagrid"
        rows={formattedData}
        columns={userColumns}
        pageSize={9}
        rowsPerPageOptions={[9]}
      />
    </div>
  );
};

export default ReactionTable;
