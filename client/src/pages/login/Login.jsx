import React, { useState } from "react";
import "./login.scss";
import { Link, useHistory, useNavigate } from "react-router-dom";
import axios from "axios";
import { useAuth } from "../../AuthContext";

export default function Login() {
  const [formData, setFormData] = useState({
    username: "",
    password: "",
  });

  const [formErrors, setFormErrors] = useState({
    username: "",
    password: "",
  });

  const [isLoading, setIsLoading] = useState(false);
  const [errorMessage, setErrorMessage] = useState(""); // State for error message

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData((prevData) => ({
      ...prevData,
      [name]: value,
    }));
  };

  const validateForm = () => {
    const errors = {
      username: "",
      password: "",
    };

    // Basic validation checks
    if (!formData.username) {
      errors.username = "Username is required";
    }

    if (!formData.password) {
      errors.password = "Password is required";
    }

    setFormErrors(errors);

    // Return true if there are no errors
    return Object.values(errors).every((error) => !error);
  };
  const { login } = useAuth();
  const navigate = useNavigate();
  const handleSubmit = async (e) => {
    e.preventDefault();

    if (validateForm()) {
      setIsLoading(true);

      try {
        const response = await axios.post(
          "http://localhost:8800/api/auth/login",
          formData
        );
        login(response.data); // Use the login function to set the currentUser

        navigate("/"); // Redirect to the desired page
        window.location.reload();
        console.log("Login successful:", response.data);
      } catch (error) {
        console.error("Login error:", error.response.data);
        setErrorMessage("Invalid username or password"); // Set error message
      } finally {
        setIsLoading(false);
      }
    }
  };

  return (
    <div className="login">
      <div className="loginWrapper">
        {" "}
        <div className="loginLeft">
          <img
            src="http://localhost:3000/assets/logo.png"
            class="coverlogo"
            alt="Login cover image"
          ></img>
          <div className="tagline">
            Unlocking insights empowering experience
          </div>
        </div>
        <div className="loginBox">
          <div className="loginRight">
            <h3 className="loginLogo">Login</h3>
            {errorMessage && <div className="error">{errorMessage}</div>}{" "}
            {/* Display error message */}
            <form onSubmit={handleSubmit}>
              <label>User Name</label>
              <input
                type="text"
                name="username"
                placeholder="Enter User Name"
                className="loginInput"
                value={formData.username}
                onChange={handleChange}
              />
              {formErrors.username && (
                <div className="error">{formErrors.username}</div>
              )}

              <label>Password</label>
              <input
                type="password"
                name="password"
                placeholder="Enter Password"
                className="loginInput"
                value={formData.password}
                onChange={handleChange}
              />
              {formErrors.password && (
                <div className="error">{formErrors.password}</div>
              )}

              <button
                type="submit"
                className="loginButton"
                disabled={isLoading}
              >
                {isLoading ? "Logging In..." : "Log In"}
              </button>
            </form>
            <br></br>
            <Link to="/register">
              <span className="dontAccount">Don't have an account?</span>
            </Link>
          </div>
        </div>
      </div>
    </div>
  );
}
