// AuthContext.js
import React, { createContext, useContext, useState, useEffect } from 'react';

const AuthContext = createContext();

export function AuthProvider({ children }) {
  const [currentUser, setCurrentUser] = useState(null);

  // Load the user from local storage when the app starts
  useEffect(() => {
    const storedUser = localStorage.getItem('currentUser');
    if (storedUser) {
      setCurrentUser(JSON.parse(storedUser));
    }
  }, []);

  const login = (user) => {
    // Perform login logic and set currentUser
    setCurrentUser(user);

    // Save the user to local storage
    localStorage.setItem('currentUser', JSON.stringify(user));
  };

  const logout = () => {
    // Perform logout logic and clear currentUser
    setCurrentUser(null);

    // Remove the user from local storage
    localStorage.removeItem('currentUser');
  };

  return (
    <AuthContext.Provider value={{ currentUser, login, logout }}>
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  return useContext(AuthContext);
}
