import User from "../models/user.model.js";
import createError from "../utils/createError.js";

export const deleteUser = async (req, res, next) => {
  const user = await User.findById(req.params.id);

  await User.findByIdAndDelete(req.params.id);
  res.status(200).send("deleted.");
};
export const getUser = async (req, res, next) => {
  const user = await User.findById(req.params.id);

  res.status(200).send(user);
};
export const getUsersWithUserRole = async (req, res) => {
  try {
    // Find users where the role is 'user'
    const users = await User.find({ role: 'user' });
    res.json(users);
    
  } catch (error) {
    console.error('Error fetching users with role "user"', error);
    res.status(500).json({ error: 'Internal server error' });
  }
};
export const getUserById = async (req, res) => {
  const userId = req.params.id;

  if (!userId) {
    return res.status(400).json({ error: 'Invalid user ID' });
  }

  try {
    const user = await User.findById(userId);

    if (!user) {
      return res.status(404).json({ error: 'User not found' });
    }

    res.json(user);
  } catch (error) {
    console.error('Error fetching user by ID', error);
    res.status(500).json({ error: 'Internal server error' });
  }
};

export const getUserCount = async (req, res) => {
  try {
    const userCount = await User.countDocuments();
    console.log(userCount)
    res.json({ count: userCount });
  } catch (error) {
    console.error("Error fetching user count", error);
    res.status(500).json({ error: "Internal server error" });
  }
};