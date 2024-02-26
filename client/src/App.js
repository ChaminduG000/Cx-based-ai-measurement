import Home from "./pages/home/Home";
import Login from "./pages/login/Login";
import Register from "./pages/register/Register";

import New from "./pages/new/New";
import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";
import { productInputs, userInputs } from "./formSource";
import "./style/dark.scss";
import { useContext } from "react";
import { DarkModeContext } from "./context/darkModeContext";
import AdverticementList from "./pages/AdverticementList/AdverticementList";
import Adverticement from "./pages/adverticement/Adverticement";
import Requests from "./pages/requests/Requests";
import { Users } from "./pages/users/Users";
import MyAds from "./pages/myads/MyAds";
import NewAdverticement from "./pages/newadverticement/NewAdverticement";
import EditAdverticement from "./pages/editadverticement/EditAdverticement";
import SingleUser from "./pages/singleUser/SingleUser";
import EditUser from "./pages/editUser/EditUser";
import Profile from "./pages/profile/Profile";
import { useAuth } from "./AuthContext";
import { AuditLog } from "./pages/auditLog/AuditLog";
import Cameras from "./pages/cameras/Cameras";
import NewCamera from "./pages/newCamera/NewCamera";

function App() {
  const { darkMode } = useContext(DarkModeContext);
  const { currentUser, login, logout } = useAuth();
  console.log(currentUser);
  return (
    <div className="app">
      <BrowserRouter>
        <Routes>
          {/* Redirect to login page if not logged in */}
          {currentUser ? (
            <Route path="/" element={<Home />} />
          ) : (
            <Route path="/login" element={<Login></Login>} />
          )}
          <Route path="/register" element={<Register />} />
          <Route path="/login" element={<Login />} />
          <Route path="/users" element={<Users />} />
          <Route path="/profile" element={<Profile />} />
          <Route path="/auditlogs" element={<AuditLog />} />

          <Route path="/users/:userId" element={<SingleUser />} />
          <Route path="/users/new" element={<New />} />
          <Route path="/users/editads/:userId" element={<EditUser />} />
          <Route path="/adverticement" element={<AdverticementList />} />
          <Route
            path="/adverticement/newads"
            element={<NewAdverticement title="Add New Ads" />}
          />

          <Route
            path="/adverticement/:adverticementId"
            element={<Adverticement />}
          />
          <Route
            path="/adverticement/editads/:adverticementId"
            element={<EditAdverticement />}
          />
          <Route path="/requests" element={<Requests />} />
          <Route path="/cameras" element={<Cameras />} />
          <Route path="/addcamera" element={<NewCamera />} />
          {/* <Route path="users" element={<YourComponentHere />} /> */}
        </Routes>
      </BrowserRouter>
    </div>
  );
}

export default App;
