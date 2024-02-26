import React, { useState, useEffect } from "react";
import axios from "axios";
import { DataGrid } from "@mui/x-data-grid";
import { format } from "date-fns";
import { useParams } from "react-router-dom";

const FeedBackTable = ({ advertisementId }) => {
  const [data, setData] = useState([]); // Initialize data as an empty array

  useEffect(() => {
    const fetchFeedbackDetails = async () => {
      try {
        const response = await axios.get(
          `http://localhost:8800/api/emotions/${advertisementId}`
        );
        setData(response.data);
      } catch (error) {
        console.error("Error fetching feedback details", error);
      }
    };

    fetchFeedbackDetails();
  }, [advertisementId]);
  console.log(data);
  const formattedData = data.map((item, index) => {
    return {
      id: index + 1,
      title: item.advertisement_title, // Display the advertisement title
      anger: item.emotion_counts.anger,
      contempt: item.emotion_counts.contempt,
      disgust: item.emotion_counts.disgust,
      fear: item.emotion_counts.fear,
      happy: item.emotion_counts.happy,
      neutral: item.emotion_counts.neutral,
      surprise: item.emotion_counts.surprise,
      sad: item.emotion_counts.sad,
      dateTime: item.timestamp,
    };
  });

  const userColumns = [
    { field: "id", headerName: "ID", width: 50 },
    { field: "title", headerName: "Title", width: 200 },
    { field: "anger", headerName: "Anger", width: 74 },
    { field: "contempt", headerName: "Contempt", width: 75 },
    { field: "disgust", headerName: "Disgust", width: 75 },
    { field: "fear", headerName: "Fear", width: 75 },
    { field: "happy", headerName: "Happy", width: 75 },
    { field: "neutral", headerName: "Neutral", width: 75 },
    { field: "surprise", headerName: "Surprise", width: 75 },
    { field: "sad", headerName: "Sad", width: 75 },
    { field: "dateTime", headerName: "Date & Time", width: 250 },
    // Other column definitions...
  ];

  return (
    <div className="datatable">
      <div className="datatableTitle">Feedbacks Details</div>
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

export default FeedBackTable;
