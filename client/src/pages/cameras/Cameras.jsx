import React from "react";
import Navbar from "../../components/navbar/Navbar";
import Sidebar from "../../components/sidebar/Sidebar";
import Datatable from "../../components/datatable/Datatable";
import RequestTable from "../../components/requestsTable/RequestTable";
import Footer from "../../components/footer/Footer";
import CamerasTable from "../../components/camerasTable/CamerasTable";

export default function Cameras() {
  return (
    <div className="list">
      <Sidebar />
      <div className="listContainer">
        <Navbar />
        <CamerasTable />
        <Footer/>
      </div>
    </div>
  );
}
