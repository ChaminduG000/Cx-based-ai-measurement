// Make sure to import your Firebase configuration properly

import storage from "./firebase";

const uploadFirebase = async (file) => {
  if (!file) return null; // Return null if no file is provided

  try {
    // Generate a unique filename or use the original file name
    const filename = `${Date.now()}-${file.name}`;

    // Create a reference to the storage location
    const storageRef = storage.ref(`user-uploads/${filename}`);

    // Upload the file to Firebase Storage and get the download URL
    const uploadTaskSnapshot = await storageRef.put(file);
    const downloadURL = await uploadTaskSnapshot.ref.getDownloadURL();

    // Return the download URL
    return downloadURL;
  } catch (error) {
    console.error("Error uploading file:", error);
    throw error; // Rethrow the error for handling in the calling component
  }
};

export default uploadFirebase;
