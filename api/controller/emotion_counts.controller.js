import Advertisement from "../models/adverticement.model.js";
import EmotionCounts from "../models/emotion_counts.model.js";
import MatchingData from "../models/feedback.model.js";


export const getEmotionsForAdvertisement = async (req, res) => {
  try {
    const { advertisementId } = req.params;
    console.log(advertisementId)
    // Find users where the role is 'user'
    const users = await EmotionCounts.find({ advertisement_id: advertisementId });
    res.json(users);
    
  } catch (error) {
    console.error('Error fetching users with role "user"', error);
    res.status(500).json({ error: 'Internal server error' });
  }
};
export const getEmotionsUsersForAdvertisement = async (req, res) => {
  try {
    const { userId } = req.params;
    console.log("userid",userId)
    // Find users where the role is 'user'
    const users = await EmotionCounts.find({ userId: userId });
    res.json(users);
    
  } catch (error) {
    console.error('Error fetching users with role "user"', error);
    res.status(500).json({ error: 'Internal server error' });
  }
};
export const getAllEmotionalCounts = async (req, res) => {
  try {
    const emotionCounts = await EmotionCounts.find();
    res.status(200).send(emotionCounts);
  } catch (error) {
    res.status(500).send("Not Found");
  }
};

