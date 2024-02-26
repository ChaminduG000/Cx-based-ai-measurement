import "./requestTable.scss";
import { DataGrid } from "@mui/x-data-grid";

import { Link } from "react-router-dom";
import { useEffect, useState } from "react";
import { userColumns, userRows } from "../../requestsource";
import axios from "axios";
import { format } from "date-fns";

const RequestTable = () => {
  const [data, setData] = useState([]);

  const currentUser = JSON.parse(localStorage.getItem("currentUser"));
  let apiUrl = "http://localhost:8800/api/adverticements/";
  if (currentUser.role === "user") {
    // If the user is a regular user, filter advertisements accordingly
    apiUrl = `http://localhost:8800/api/adverticements/user/${currentUser._id}`;
  }

  console.log(apiUrl);
  useEffect(() => {
    const fetchData = async () => {
      try {
        const response = await axios.get(apiUrl);
        setData(response.data);
      } catch (error) {
        console.error("Error fetching data", error);
      }
    };

    fetchData();
  }, [apiUrl]);

  const formattedData = data.map((item, index) => {
    const createdAt = new Date(item.scheduleDateTime);
    const date = format(createdAt, "yyyy-MM-dd");
    const time = format(createdAt, "HH:mm:ss");

    return {
      id: item._id, // Manually assign a unique identifier
      title: item.title,
      video: item.video,
      date,
      time,
      status: item.status,
    };
  });
  const handleApprove = async (advertiseId) => {
    try {
      // Send a PUT request to your server to update the status to true
      await axios.put(
        `http://localhost:8800/api/adverticements/approve/${advertiseId}`
      );
      // After successful update, refetch the data
      const response = await axios.get(apiUrl);
      setData(response.data);
    } catch (error) {
      console.error("Error approving advertisement", error);
    }
  };
  const handleReject = async (advertiseId) => {
    try {
      // Send a PUT request to your server to update the status to false
      await axios.put(
        `http://localhost:8800/api/adverticements/reject/${advertiseId}`
      );
      // After successful update, refetch the data
      const response = await axios.get(apiUrl);
      setData(response.data);
    } catch (error) {
      console.error("Error rejecting advertisement", error);
    }
  };
  const userColumns = [
    { field: "id", headerName: "ID", width: 150 },
    { field: "title", headerName: "Title", width: 150 },
    { field: "video", headerName: "Video", width: 150 },
    {
      field: "date",
      headerName: "Schedule Date",
      width: 200,
    },
    {
      field: "time",
      headerName: "Schedule Time",
      width: 200,
    },
    {
      field: "status",
      headerName: "Status",
      width: 160,
      renderCell: (params) => {
        const statusText = params.row.status ? "Active" : "Inactive";
        const statusClassName = params.row.status
          ? "activeStatus"
          : "inactiveStatus";

        return (
          <div className={`cellWithStatus ${statusClassName}`}>
            {statusText}
          </div>
        );
      },
    },

    {
      field: "action",
      headerName: "Action",
      width: 300,
      renderCell: (params) => {
        return (
          <div className="cellAction">
            {/* <Link to="/users/test" style={{ textDecoration: "none" }}>
              <div className="viewButton">View</div>
            </Link> */}
            {/* <div
              className="deleteButton"
              onClick={() => handleDelete(params.row.id)}
            >
              Delete
            </div> */}
            <div
              className="approveButton"
              onClick={() => handleApprove(params.row.id)} // Call handleApprove with the advertisement ID
            >
              Approve
            </div>
            <div
              className="rejectButton"
              onClick={() => handleReject(params.row.id)} // Call handleReject with the advertisement ID
            >
              Reject
            </div>
          </div>
        );
      },
    },
  ];

  return (
    <div className="datatable">
      <div className="datatableTitle">
        Requests Details
        {/* <Link to="/users/new" className="link">
          Add New
        </Link> */}
      </div>
      <DataGrid
        className="datagrid"
        rows={formattedData}
        columns={userColumns}
        pageSize={9}
        rowsPerPageOptions={[9]}
        checkboxSelection
      />
    </div>
  );
};

export default RequestTable;
