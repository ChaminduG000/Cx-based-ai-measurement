import React, { useEffect, useState } from "react";
import "./auditLog.scss";
import Sidebar from "../../components/sidebar/Sidebar";
import Navbar from "../../components/navbar/Navbar";

export const AuditLog = () => {
  const [auditLogEntries, setAuditLogEntries] = useState([]);

  useEffect(() => {
    // Replace 'YOUR_API_ENDPOINT' with the actual URL of your backend API
    fetch("http://localhost:8800/api/audit-logs")
      .then((response) => response.json())
      .then((data) => {
        // Fetch user details for all user IDs in parallel
        Promise.all(
          data.map((entry) =>
            fetch(`http://localhost:8800/api/users/${entry.user}`)
              .then((response) => response.json())
              .then((userData) => ({
                ...entry,
                user: userData.username, // Adjust the property as per your user data structure
              }))
              .catch((error) => {
                console.error(
                  "Error fetching user data for audit log entry",
                  error
                );
                return entry; // Keep the original entry on error
              })
          )
        ).then((auditLogWithUserDetails) => {
          setAuditLogEntries(auditLogWithUserDetails);
        });
      })
      .catch((error) => {
        console.error("Error fetching audit log entries", error);
      });
  }, []);

  return (
    <div className="audit">
      <Sidebar />
      <div className="auditContainer">
        <Navbar />

        <div>
          <h1>Audit Log</h1>
          <table>
            <thead>
              <tr>
                <th>Timestamp</th>
                <th>Action</th>
                <th>User</th>
                <th>Details</th>
              </tr>
            </thead>
            <tbody>
              {auditLogEntries.map((entry, index) => (
                <tr key={index}>
                  <td>{new Date(entry.timestamp).toLocaleString()}</td>
                  <td>{entry.action}</td>
                  <td>{entry.user}</td>
                  <td>{entry.details}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};
