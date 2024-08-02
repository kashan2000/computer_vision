import torch
from model_config import get_pose_from_frame

def process_pose(pose_data):
    frame_count, frame = pose_data
    pose_results = get_pose_from_frame(frame)
    print(f"Finished processing pose for frame: {frame_count}")
    return pose_results
