import "./navbar.scss";
import SearchOutlinedIcon from "@mui/icons-material/SearchOutlined";
import LanguageOutlinedIcon from "@mui/icons-material/LanguageOutlined";
import DarkModeOutlinedIcon from "@mui/icons-material/DarkModeOutlined";
import FullscreenExitOutlinedIcon from "@mui/icons-material/FullscreenExitOutlined";
import NotificationsNoneOutlinedIcon from "@mui/icons-material/NotificationsNoneOutlined";
import ChatBubbleOutlineOutlinedIcon from "@mui/icons-material/ChatBubbleOutlineOutlined";
import ListOutlinedIcon from "@mui/icons-material/ListOutlined";
import { DarkModeContext } from "../../context/darkModeContext";
import { useContext, useEffect, useState } from "react";
import { Link } from "react-router-dom";

const Navbar = () => {
  const { dispatch } = useContext(DarkModeContext);
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
    <div className="navbar">
      <div className="wrapper">
        <div className="items">
          {userRole === "admin" && (
            <Link to="/auditlogs" style={{ textDecoration: "none" }}>
              <div className="item">
                {/* <NotificationsNoneOutlinedIcon className="icon" /> */}
                {/* <div className="counter">1</div> */}
              </div>
            </Link>
          )}
        </div>
      </div>
    </div>
  );
};

export default Navbar;
