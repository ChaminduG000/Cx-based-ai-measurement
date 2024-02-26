import "./adverticementTable.scss";
import { DataGrid } from "@mui/x-data-grid";
import { Link } from "react-router-dom";
import { useEffect, useState } from "react";
import axios from "axios";
import { format, isValid, parseISO } from "date-fns"; // Import isValid and parseISO
import ReactPlayer from "react-player";

const AdverticementTable = () => {
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
    // Directly extract and format the date and time components
    const scheduleDate = item.scheduleDateTime.slice(0, 10);
    const scheduleTime = item.scheduleDateTime.slice(11, 19);
    const endScheduleDate = item.endScheduleDateTime.slice(0, 10);
    const endScheduleTime = item.endScheduleDateTime.slice(11, 19);

    return {
      id: item._id,
      title: item.title,
      video: item.video,
      scheduleDate,
      scheduleTime,
      endScheduleDate,
      endScheduleTime,
      cameras: item.cameras.map((camera) => camera.name).join(", "),
      status: item.status,
    };
  });

  const handleDelete = async (advertiseId) => {
    try {
      await axios.delete(
        `http://localhost:8800/api/adverticements/${advertiseId}`
      );
      // After successful deletion, refetch the data
      const response = await axios.get(apiUrl);
      setData(response.data);
    } catch (error) {
      console.error("Error deleting advertisement", error);
    }
  };
  console.log(data)
  const filteredData =
    currentUser.role === "admin"
      ? formattedData.filter((item) => item.status === true)
      : formattedData;

  const userColumns = [
    { field: "id", headerName: "ID", width: 200 },
    { field: "title", headerName: "Title", width: 100 },
    {
      field: "video",
      headerName: "Video",
      width: 50,
      renderCell: (params) => {
        return (
          <div className="cellWithImg">
            <ReactPlayer
              height={30}
              width={40}
              url={params.row.video}
              controls={true}
            />
            {params.row.adDetails}
          </div>
        );
      },
    },
    {
      field: "scheduleDate",
      headerName: "Schedule Date",
      width: 150,
    },
    {
      field: "scheduleTime",
      headerName: "Schedule Time",
      width: 150,
    },
    {
      field: "endScheduleDate",
      headerName: "End Schedule Date",
      width: 150,
    },
    {
      field: "endScheduleTime",
      headerName: "End Schedule Time",
      width: 150,
    },
    {
      field: "cameras",
      headerName: "Cameras",
      width: 200,
    },
    {
      field: "status",
      headerName: "Status",
      width: 75,
      renderCell: (params) => {
        const statusText = params.row.status ? "Active" : "Pending";
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
      width: 200,
      renderCell: (params) => {
        return (
          <div className="cellAction">
            <Link
              to={`/adverticement/${params.row.id}`}
              style={{ textDecoration: "none" }}
            >
              <div className="viewButton">View</div>
            </Link>
            <Link
              to={`/adverticement/editads/${params.row.id}`}
              style={{ textDecoration: "none" }}
            >
              <div className="editButton">Edit</div>
            </Link>
            <div
              className="deleteButton"
              onClick={() => handleDelete(params.row.id)}
            >
              Delete
            </div>
          </div>
        );
      },
    },
  ];
  const [userRole, setUserRole] = useState("");

  useEffect(() => {
    // Retrieve user role from local storage
    const storedUserRole = JSON.parse(localStorage.getItem("currentUser"));
    console.log(storedUserRole.role);

    if (storedUserRole) {
      setUserRole(storedUserRole.role);
    }
  }, []);
  return (
    <div className="datatable">
      <div className="datatableTitle">
        Adverticements Details
        {userRole === "user" && (
          <Link to="/adverticement/newads" className="link">
            Add New
          </Link>
        )}
      </div>

      <DataGrid
        className="datagrid"
        rows={filteredData}
        columns={userColumns}
        pageSize={9}
        rowsPerPageOptions={[9]}
        checkboxSelection
      />
    </div>
  );
};

export default AdverticementTable;
