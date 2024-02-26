from pathlib import Path
import numpy as np
import argparse
import time
import torch
import os
import threading
from datetime import datetime
from emotion import detect_emotion, init
from models.experimental import attempt_load
from utils.datasets import LoadStreams
from utils.general import check_img_size, set_logging, create_folder, non_max_suppression, scale_coords
from utils.torch_utils import select_device, time_synchronized
from PIL import Image, ImageDraw, ImageFont
from pymongo import MongoClient

# Define emotions
EMOTIONS = ("anger", "contempt", "disgust", "fear",
            "happy", "neutral", "sad", "surprise")

# Replace with your MongoDB Atlas connection string
mongo_uri = "mongodb+srv://root:root@cluster0.zy96vnm.mongodb.net/?retryWrites=true&w=majority"
mongo_client = MongoClient(mongo_uri)

# Replace with your database name
mongo_db = mongo_client['test']

# Replace with your collection name
mongo_collection = mongo_db['emotion_counts']

# Function to check if it's time for an ad and return the ad's data
def get_advertisement_data(collection):
    current_time = datetime.now()
    ad_data = collection.find_one(
        {
            'scheduleDateTime': {
                '$lte': current_time
            },
            'endScheduleDateTime': {
                '$gte': current_time
            }
        }
    )
    return ad_data

# Function to print emotion count data after the end time
def print_emotion_count_after_end(emotion_counts, end_time, collection, ad_id, ad_title, camera_name):
    print("Advertisement End ! | Emotion Counts Summary:")
    for emotion, count in emotion_counts.items():
        print(f"{emotion}: {count} detected", end=", ")
    print("")

    # Send emotion count data to MongoDB only if it hasn't been sent before
    if not emotion_counts.get('_sent_to_mongodb', False):
        emotion_counts['_sent_to_mongodb'] = True  # Mark as sent
        send_emotion_counts_to_mongodb(
            emotion_counts, ad_id, ad_title, collection, camera_name)

# Function to send emotion count data to MongoDB with advertisement details
def send_emotion_counts_to_mongodb(emotion_counts, ad_id, ad_title, collection, camera_name):
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    emotion_count_data = {
        'advertisement_id': ad_id,
        'advertisement_title': ad_title,
        'camera_name': camera_name,
        'timestamp': current_time,
        'emotion_counts': emotion_counts
    }

    collection.insert_one(emotion_count_data)

# Function to reset emotion counts for a new advertisement
def reset_emotion_counts(emotions):
    return {emotion: 0 for emotion in emotions}

# Function to process a single advertisement and related cameras
def process_single_advertisement(ad_data, opt, imgsz, device, model, emotions, mongo_db, mongo_collection, stride, half):
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"Current Date and Time: {current_time}")

    ad_id = ad_data['_id']
    ad_title = ad_data['title']
    schedule_time = ad_data['scheduleDateTime']
    end_time = ad_data['endScheduleDateTime']

    print("Advertisement Details:")
    print(f"ID: {ad_id}")
    print(f"Title: {ad_title}")
    print(f"Schedule Time: {schedule_time}")
    print(f"End Time: {end_time}")

    camera_name = ad_data['cameras'][0]['name']
    camera_ip = get_camera_ip_by_name(mongo_db['cameras'], camera_name)

    if camera_ip:
        print(f"Advertisement {ad_title} Started | Check Emotion  Count: {camera_ip}")
        print(f"Wait...")

        # Reset emotion counts for each new advertisement
        running_total_emotion_counts = reset_emotion_counts(EMOTIONS)

        # Start the thread without waiting for it to finish
        ad_thread = threading.Thread(target=process_camera, args=(
            camera_name, opt, imgsz, device, model, EMOTIONS, mongo_db, mongo_collection, stride, half, running_total_emotion_counts, end_time))
        ad_thread.start()

        # Continue with other tasks without waiting for the thread

        # Wait until the scheduled end time of the advertisement
        while datetime.now() < end_time:
            time.sleep(1)

        # Fetch the emotion counts after the thread has completed
        emotion_counts = running_total_emotion_counts.copy()  # Copy the counts to avoid potential race conditions

        # Print and send emotion count summary after the ad ends
        print_emotion_count_after_end(
            emotion_counts, end_time, mongo_collection, ad_id, ad_title, camera_name)

        # Wait for the thread to finish
        ad_thread.join()

    else:
        print(f"No camera information found for {camera_name}")

# Function to process advertisements continuously
def process_advertisements_continuous(opt, imgsz, device, model, emotions, mongo_db, mongo_collection, stride, half):
    while True:
        ad_data = get_advertisement_data(mongo_db['adverticements'])
        if ad_data:
            process_single_advertisement(
                ad_data, opt, imgsz, device, model, emotions, mongo_db, mongo_collection, stride, half)
        else:
            print("No available advertisement at the current time.")

# Function to get camera IP by name from MongoDB
def get_camera_ip_by_name(collection, camera_name):
    camera_data = collection.find_one({'name': camera_name})
    return camera_data.get('ip') if camera_data else None

# Function to initialize components such as folders, logging, and the model
def initialize_components():
    create_folder("output")  # Assuming you want to create a folder for output
    set_logging()
    device = select_device("")  # Use an empty string for CPU
    init(device)
    half = device.type != 'cpu'

    model = attempt_load("weights/yolov7-tiny.pt", map_location=device)
    stride = int(model.stride.max())
    imgsz = check_img_size(512, s=stride)  # Set a default img size for initialization
    if half:
        model.half()

    return device, model, imgsz, stride, half

# Function to check termination condition
def termination_condition_met(end_time):
    return datetime.now() >= end_time

# Function to simulate camera processing
def simulate_camera_processing():
    # Placeholder for simulating camera processing
    # Replace this with your actual camera processing logic
    # For example, you can use a test image and generate a random emotion for simulation
    return np.zeros((640, 480, 3)), np.random.choice(EMOTIONS)

# Function to process camera data and update emotion counts
def process_camera(camera_name, opt, imgsz, device, model, emotions, mongo_db, mongo_collection, stride, half, running_total_emotion_counts, end_time):
    while not termination_condition_met(end_time):
        frame, emotion = simulate_camera_processing()

        # Update emotion counts
        if emotion:
            running_total_emotion_counts[emotion] += 1

        # Add any additional logic as needed

        time.sleep(1)  # Adjust the sleep duration as needed

if __name__ == '__main__':
    opt = argparse.Namespace(
        source='0',  # Replace with your desired source
        img_size=512,  # Replace with your desired image size
        conf_thres=0.5,  # Replace with your desired confidence threshold
        iou_thres=0.45,  # Replace with your desired IOU threshold
        device='',  # Leave empty for automatic device selection
        hide_img=False,
        output_path='output1.mp4',
        no_save=False,
        output_width=1500,
        output_height=1000,
        agnostic_nms=False,
        augment=False,
        line_thickness=2,
        hide_conf=False,
        show_fps=False
    )

    with torch.no_grad():
        device, model, imgsz, stride, half = initialize_components()
        try:
            process_advertisements_continuous(opt, imgsz, device, model, EMOTIONS, mongo_db, mongo_collection, stride, half)
        except KeyboardInterrupt:
            print("Process interrupted. Closing MongoDB client.")
            mongo_client.close()
