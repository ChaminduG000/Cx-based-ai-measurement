// Import the functions you need from the SDKs you need
import firebase from "firebase/compat/app";
import "firebase/compat/storage";

const firebaseConfig = {
  apiKey: "AIzaSyCDDMvGm_LE6etFl715nbtO5zZteMB9MSw",
  authDomain: "customer-experience-e3a95.firebaseapp.com",
  projectId: "customer-experience-e3a95",
  storageBucket: "customer-experience-e3a95.appspot.com",
  messagingSenderId: "735831940673",
  appId: "1:735831940673:web:2752a62aab8ee71985b0c0",
};

// Initialize Firebase
firebase.initializeApp(firebaseConfig);
const storage = firebase.storage();

export default storage;
