import React, { useState } from "react";
import "./register.css";
import axios from "axios";
import { Link, useNavigate } from "react-router-dom";

export default function Register() {
  const [formData, setFormData] = useState({
    username: "",
    email: "",
    password: "",
    confirmPassword: "",
  });

  const [formErrors, setFormErrors] = useState({
    username: "",
    email: "",
    password: "",
    confirmPassword: "",
  });

  const [isLoading, setIsLoading] = useState(false);
  const [errorMessage, setErrorMessage] = useState("");

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
      email: "",
      password: "",
      confirmPassword: "",
    };

    // Basic validation checks
    if (!formData.username) {
      errors.username = "Username is required";
    }

    if (!formData.email) {
      errors.email = "Email is required";
    } else if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(formData.email)) {
      errors.email = "Invalid email address";
    }

    if (!formData.password) {
      errors.password = "Password is required";
    } else if (formData.password.length < 6) {
      errors.password = "Password must be at least 6 characters";
    }

    if (formData.password !== formData.confirmPassword) {
      errors.confirmPassword = "Passwords do not match";
    }

    setFormErrors(errors);

    return Object.values(errors).every((error) => !error);
  };

  const navigate = useNavigate();
  const handleSubmit = async (e) => {
    e.preventDefault();

    if (validateForm()) {
      setIsLoading(true);

      try {
        const response = await axios.post(
          "http://localhost:8800/api/auth/register",
          formData
        );
        console.log("Registration successful:", response.data);
        navigate("/login");
        window.location.reload();
      } catch (error) {
        console.error("Registration error:", error.response.data);
        setErrorMessage("Registration failed. Please try again."); // Set error message
      } finally {
        setIsLoading(false);
      }
    }
  };

  return (
    <div className="reg">
      <div className="reg-wrapper">
        <div className="regbox">
          <div className="reg-left">
            <h3>Register Your Account</h3>
            {errorMessage && <div className="error">{errorMessage}</div>}{" "}
            {/* Display error message */}
            <form onSubmit={handleSubmit}>
              <label>User Name</label>
              <input
                type="text"
                name="username"
                placeholder="Enter a user name"
                className="regInput"
                value={formData.username}
                onChange={handleChange}
              />
              {formErrors.username && (
                <div className="error">{formErrors.username}</div>
              )}

              <label>Email Address</label>
              <input
                type="email"
                name="email"
                placeholder="Enter email address"
                className="regInput"
                value={formData.email}
                onChange={handleChange}
              />
              {formErrors.email && (
                <div className="error">{formErrors.email}</div>
              )}

              <label>Password</label>
              <input
                type="password"
                name="password"
                placeholder="Enter your password"
                className="regInput"
                value={formData.password}
                onChange={handleChange}
              />
              {formErrors.password && (
                <div className="error">{formErrors.password}</div>
              )}

              <label>Confirm Password</label>
              <input
                type="password"
                name="confirmPassword"
                placeholder="Enter your password again"
                className="regInput"
                value={formData.confirmPassword}
                onChange={handleChange}
              />
              {formErrors.confirmPassword && (
                <div className="error">{formErrors.confirmPassword}</div>
              )}

              <br />
              <button className="sinBtn" type="submit" disabled={isLoading}>
                {isLoading ? "Registering..." : "Register"}
              </button>
            </form>
          </div>
          <Link to="/login">
            <span className="dontAccount">Already have an account? Login</span>
          </Link>
        </div>
        <div className="reg-right">
          <img
            src="http://localhost:3000/assets/logo.png"
            class="coverlogo"
            alt="Login cover image"
          ></img>
          <div className="tagline">
            Unlocking insights empowering experience
          </div>
        </div>
      </div>
    </div>
  );
}
