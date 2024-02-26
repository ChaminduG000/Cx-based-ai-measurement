import Sidebar from "../../components/sidebar/Sidebar";
import Navbar from "../../components/navbar/Navbar";
import "./home.scss";
import Widget from "../../components/widget/Widget";
import Featured from "../../components/featured/Featured";
import Chart from "../../components/chart/Chart";
import Table from "../../components/table/Table";
import { useEffect, useState } from "react";
import ReactionTable from "../../components/reactionTable/ReactionTable";

const Home = () => {
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
    <div className="home">
      <Sidebar />
      <div className="homeContainer">
        <Navbar />
        <div className="widgets">
          {userRole === "user" ? (
            <div className="widgets-admin">
              <div className="widgets-wrapper">
                <Widget type="myadsApprove" />
                <Widget type="myadsInactive" />
                <Widget type="user" />
              </div>
              <ReactionTable></ReactionTable>
            </div>
          ) : (
            <div className="widgets-admin">
              <div className="widgets-wrapper">
                <Widget type="user" />
                <Widget type="order" />
                <Widget type="balance" />
              </div>
              <ReactionTable></ReactionTable>
            </div>
          )}
        </div>
        <div className="charts"></div>
      </div>
    </div>
  );
};

export default Home;
