"""
This module is dedicated to the configuration of the model.
"""

import numpy as np
from cv2 import rectangle
from log import logging
from ultralytics import YOLO
import torch
import traceback

# Load models
device = 'cuda' if torch.cuda.is_available() else 'cpu'

yolo_model_detection = YOLO("yolov8n.pt").to(device)  # YOLOv8 for object detection
yolo_model_pose = YOLO("yolov8n-pose.pt").to(device)  # YOLOv8 for pose estimation


# Global variables
EPSILON = 0.01
WIDTH = 800
HEIGHT = 800
MIN_CONFIDENCE = 0.5
SPORTS_BALL_CLASS_INDEX = 32

############### Functions ################

def get_coordinates_from_frame(frame):
    """
    Get the coordinates of the objects in the frame.
    Args:
        frame (np.array): Frame from the video.
    Returns:
        tuple: A tuple containing the results and frame.
    """
    print(f"yolo detect shape : {frame.shape}")
    try:
        results = {}
        color = (0, 255, 0)  # Green for bounding boxes

        detect_results = yolo_model_detection(frame)
        for dr in detect_results:
            if dr.boxes is not None and len(dr.boxes) > 0:
                for i, cls in enumerate(dr.boxes.cls):
                    if cls == SPORTS_BALL_CLASS_INDEX:
                        box = dr.boxes.xywhn[i]
                        x, y, w, h = box.tolist()
                        print("ball in results")
                        results['ball'] = [x, y, w, h]

        if 'ball' not in results:
            print("Ball not in results")
            results['ball'] = [None, None, None, None]

        return results
    
    except Exception as e:
        print("An error occurred in object detection: %s", str(e))
        return {'ball': [None, None, None, None]}

def get_pose_from_frame(frame):
    """
    Process pose estimation from the given frame using YOLOv8-pose.
    Args:
        frame (np.array): Frame from the video.
    Returns:
        dict: A dictionary containing the pose data.
    """
    print(f"yolo pose shape : {frame.shape}")
    try:
        results = {}
        color = (0, 0, 255)  # Red for keypoints

        pose_results = yolo_model_pose(frame)

        for pr in pose_results:
            # print(f"pose results>> ${pr.keypoints.xyn[0]}")
            if pr.keypoints is not None and len(pr.keypoints) > 0:
                keypoint_mapping = {
                    'nose': 0, 'r_eye': 1, 'l_eye': 2, 'r_ear': 3, 'l_ear': 4,
                    'r_shoulder': 5, 'l_shoulder': 6, 'r_elbow': 7, 'l_elbow': 8,
                    'r_wrist': 9, 'l_wrist': 10, 'r_hip': 11, 'l_hip': 12,
                    'r_knee': 13, 'l_knee': 14, 'r_ankle': 15, 'l_ankle': 16
                }
                for key, idx in keypoint_mapping.items():
                    # if pr.keypoints.xyn[0][idx][0] != 0 and pr.keypoints.xyn[0][idx][1] != 0:
                    #     keypoint_list = pr.keypoints.xyn[0][idx].tolist()
                    #     results[key] = keypoint_list

                    if len(pr.keypoints.xyn[0]) > 0:
                        print(f"value of idx> {idx}")
                        print(f"length of list is {len(pr.keypoints.xyn[0])}")
                        keypoint_list = pr.keypoints.xyn[0][idx].tolist()
                        confidence = pr.keypoints.conf[0][idx] if hasattr(pr.keypoints, 'conf') else 1.0
                        print(f"Detected {key}: {keypoint_list} with confidence {confidence}")
                        if keypoint_list[0] != 0 and keypoint_list[1] != 0:
                            results[key] = keypoint_list
                        else:
                            print(f"{key} coordinates are zero.")
                    else:
                        print(f"{key} not detected or keypoints array is empty.")        
                        
                # # Calculate neck point
                # if 'r_shoulder' in results and 'l_shoulder' in results:
                #     neck_x = (results['r_shoulder'][0] + results['l_shoulder'][0]) // 2
                #     neck_y = (results['r_shoulder'][1] + results['l_shoulder'][1]) // 2
                #     results['neck'] = [neck_x, neck_y]
                # # Calculate head point
                # if 'nose' in results and 'l_ear' in results and 'r_eye' in results:
                #     l_ear = results['l_ear']
                #     r_eye = results['r_eye']
                #     distance_between_ears = np.linalg.norm(np.array(l_ear) - np.array(r_eye))
                #     head_x = results['nose'][0]
                #     head_y = results['nose'][1] - int(distance_between_ears)
                #     results['head'] = [head_x, head_y]

        # if 'r_ankle' not in results:
        #     results['r_ankle'] = [None, None]
        
        # if 'l_ankle' not in results:
        #     results['l_ankle'] = [None, None]

        print(f"retuning pose result: ${results}")    

        return results

    except Exception as e:
        print("An error occurred in processing the video: %s", str(e))
        traceback.print_exc()
        return {'r_ankle' : [None, None], 'l_ankle' : [None, None]}