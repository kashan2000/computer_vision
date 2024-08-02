import torch
from model_config import get_coordinates_from_frame

def process_detection(detection_data):
    frame_count, frame = detection_data
    detection_results = get_coordinates_from_frame(frame)
    print(f"Finished processing detection for frame: {frame_count}")
    return detection_results
