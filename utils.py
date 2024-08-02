import cv2
import csv
import os
from datetime import datetime

def save_video(frames, output_video_path):
    if not frames:
        raise ValueError("No frames to save.")
    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, 5, (width, height))
    for frame in frames:
        out.write(frame)
    out.release()

def write_to_csv(data):
    drill_type = data['drill_type']
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    filename = f"csv/{drill_type}-{timestamp}.csv"
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        if file.tell() == 0:
            writer.writerow(data.keys())
        writer.writerow(data.values())
