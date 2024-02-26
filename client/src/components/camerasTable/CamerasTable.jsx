import "./camerasTable.scss";
import { DataGrid } from "@mui/x-data-grid";
import { useEffect, useState } from "react";
import axios from "axios";
import { format } from "date-fns";
import { Link } from "react-router-dom";

const CamerasTable = () => {
  const [data, setData] = useState([]);

  const currentUser = JSON.parse(localStorage.getItem("currentUser"));
  let apiUrl = "http://localhost:8800/api/camera/";

  console.log(apiUrl);
  useEffect(() => {
    const fetchData = async () => {
      try {
        const response = await axios.get(apiUrl);

        // Rename the '_id' property to 'id' for each row
        const formattedData = response.data.map((row) => ({
          ...row,
          id: row._id,
        }));

        setData(formattedData);
      } catch (error) {
        console.error("Error fetching data", error);
      }
    };

    fetchData();
  }, [apiUrl]);

  const handleDelete = async (id) => {
    try {
      // Make an API call to delete the camera
      await axios.delete(`http://localhost:8800/api/camera/${id}`);

      // Update the state to remove the deleted camera
      setData((prevData) => prevData.filter((row) => row.id !== id));
    } catch (error) {
      console.error("Error deleting camera", error);
    }
  };

  const userColumns = [
    { field: "id", headerName: "ID", width: 150 },
    { field: "name", headerName: "Name", width: 150 },
    { field: "ip", headerName: "IP", width: 150 },
    {
      field: "action",
      headerName: "Action",
      width: 300,
      renderCell: (params) => {
        return (
          <div className="cellAction">
            {/* Commented out Link import since it's not used
            <Link to="/users/test" style={{ textDecoration: "none" }}>
              <div className="viewButton">View</div>
            </Link> */}
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

  return (
    <div className="datatable">
      <div className="datatableTitle">
        Requests Details
        {/* Commented out Link since it's not used */}
        <Link to="/addcamera" className="link">
          Add New
        </Link>
      </div>
      <DataGrid
        className="datagrid"
        rows={data} // Use data instead of formattedData
        columns={userColumns}
        pageSize={9}
        rowsPerPageOptions={[9]}
        checkboxSelection
      />
    </div>
  );
};

export default CamerasTable;