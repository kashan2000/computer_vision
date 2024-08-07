import numpy as np
import cv2
from log import logging
from ultralytics import YOLO
import torch


# Load models
device = 'cuda' if torch.cuda.is_available() else 'cpu'

yolo_model_detection = YOLO("yolov8n.pt").to(device) 
yolo_model_pose = YOLO("yolov8n-pose.pt").to(device)

def get_pose_from_frame(frame):
    """
    Process pose estimation from the given frame using YOLOv8-pose.
    Args:
        frame (np.array): Frame from the video.
    Returns:
        dict: A dictionary containing the pose data.
    """
    try:
        results = {}
        color = (0, 0, 255)  # Red for keypoints

        pose_results = yolo_model_pose(frame)

        for pr in pose_results:
            if pr.keypoints is not None and len(pr.keypoints.xyn[0]) > 0:
                keypoint_mapping = {
                    'nose': 0, 'r_eye': 1, 'l_eye': 2, 'r_ear': 3, 'l_ear': 4,
                    'r_shoulder': 5, 'l_shoulder': 6, 'r_elbow': 7, 'l_elbow': 8,
                    'r_wrist': 9, 'l_wrist': 10, 'r_hip': 11, 'l_hip': 12,
                    'r_knee': 13, 'l_knee': 14, 'r_ankle': 15, 'l_ankle': 16
                }
                for key, idx in keypoint_mapping.items():
                    if pr.keypoints.xyn[0][idx][0] != 0 and pr.keypoints.xyn[0][idx][1] != 0:
                        keypoint_list = pr.keypoints.xyn[0][idx].tolist()
                        results[key] = keypoint_list

        print("Results before setting defaults:", results)

        # Ensure r_ankle and l_ankle are in the results
        if 'r_ankle' not in results:
            results['r_ankle'] = [None, None]
        
        if 'l_ankle' not in results:
            results['l_ankle'] = [None, None]

        return results

    except Exception as e:
        logging.error("An error occurred in processing the video: %s", str(e))
        return {'r_ankle': [None, None], 'l_ankle': [None, None]}

image_path = "C:\\Users\\khank\\OneDrive\\Desktop\\ml.jpg"
frame = cv2.imread(image_path)
pose_results = get_pose_from_frame(frame)
print("Pose results:", pose_results)