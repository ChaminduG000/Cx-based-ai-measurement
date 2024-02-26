// import "./myadsTable.scss";
// import { DataGrid} from "@mui/x-data-grid";

// import { Link } from "react-router-dom";
// import { UseState, useState } from "react";


// const MyAdsTable = () => {
//   const [data, setData] = useState(adsRows);

//   const handleDelete = (id) => {
//     setData(data.filter((item) => item.id !== id));
//   };

//   const actionColumn = [
//     {
//       field: "action",
//       headerName: "Action",
//       width: 200,
//       renderCell: (params) => {
//         return (
//           <div className="cellAction">
//             <Link to="/MyAds/test" style={{ textDecoration: "none" }}>
//               <div className="viewButton">View</div>
//             </Link>
//             <div
//               className="deleteButton"
//               onClick={() => handleDelete(params.row.id)}
//             >
//               Delete
//             </div>
//           </div>
//         );
//       },
//     },
//   ];
//   return (
//     <div className="datatable">
//       <div className="datatableTitle">
//         Adverticements Details
//          <Link to="/users/new" className="link">
//           Add New
//         </Link> 
//       </div>
//       <DataGrid
//         className="datagrid"
//         rows={data}
//         columns={myadsColumns.concat(actionColumn)}
//         pageSize={9}
//         rowsPerPageOptions={[9]}
//         checkboxSelection
//       />
//     </div>
//   );
// };




// export default MyAdsTable;


