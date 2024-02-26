import "./usersTable.scss";
import { DataGrid } from "@mui/x-data-grid";

import { Link } from "react-router-dom";
import { useEffect, useState } from "react";
import { userColumns, userRows } from "../../userssource";
import axios from "axios";
import ReactPlayer from "react-player";

const UsersTable = () => {
  const [data, setData] = useState([]);

  const currentUser = JSON.parse(localStorage.getItem("currentUser"));
  let apiUrl = "http://localhost:8800/api/users";

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
    return {
      id: item._id, // Manually assign a unique identifier
      username: item.username,
      email: item.email,
    };
  });
  const handleDelete = async (id) => {
    console.log(id);
    try {
      await axios.delete(`http://localhost:8800/api/users/${id}`);
      // After successful deletion, refetch the data
      const response = await axios.get(apiUrl);
      setData(response.data);
    } catch (error) {
      console.error("Error deleting advertisement", error);
    }
  };
  const userColumns = [
    { field: "id", headerName: "ID", width: 200 },
    {
      field: "username",
      headerName: "User",
      width: 230,
      renderCell: (params) => {
        return (
          <div className="cellWithImg">
            <img className="cellImg" src={params.row.img} />
            {params.row.username}
          </div>
        );
      },
    },

    { field: "email", headerName: "Email", width: 250 },

    {
      field: "action",
      headerName: "Action",
      width: 200,
      renderCell: (params) => {
        return (
          <div className="cellAction">
            <Link
              to={`/users/${params.row.id}`}
              style={{ textDecoration: "none" }}
            >
              <div className="viewButton">View</div>
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
  return (
    <div className="datatable">
      <div className="datatableTitle">
        Customers Details
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

export default UsersTable;
