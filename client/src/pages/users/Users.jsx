import React from "react";
import Sidebar from "../../components/sidebar/Sidebar";
import Navbar from "../../components/navbar/Navbar";
import UsersTable from "../../components/usersTable/UsersTable";
import Footer from "../../components/footer/Footer";

export const Users = () => {
  return (
    <div className="list">
      <Sidebar />
      <div className="listContainer">
        <Navbar />
        <UsersTable />
        <Footer/>
      </div>
    </div>
  );
};
