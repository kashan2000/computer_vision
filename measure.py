"""
Module which calculates various parameters.
"""

import math
import numpy as np
from log import logging


###################### Functions for calculating parameters ######################

##### Displacement #####

def displacement_joints(initial_pos, final_pos):
    """
    Calculate displacement based on initial and final positions.
    Args:
    - initial_pos (list): Initial position as a list [x, y].
    - final_pos (list): Final position as a list [x, y].
    Returns:
    - displacement (float): Displacement calculated as the Euclidean distance
                            between the initial and final positions.
    """
    try:
        return np.linalg.norm(np.array(final_pos) - np.array(initial_pos))
    except Exception as e:
        logging.error("Error calculating displacement: %s", str(e))
        return None


def displacement_ball(initial_pos, final_pos):
    """
    Calculate displacement based on initial and final positions.
    Args:
    - initial_pos (list): Initial position as a list [x, y, height, width].
    - final_pos (list): Final position as a list [x, y, height, width].
    Returns:
    - displacement (float): Displacement calculated as the Euclidean distance
                            between the initial and final positions.
    """
    try:
        return np.linalg.norm(np.array(final_pos[:2]) - np.array(initial_pos[:2]))
    except Exception as e:
        logging.error("Error calculating displacement: %s", str(e))
        return None


def depth_of_ball(data):
    """
    Calculate the depth of a ball based on its detected bounding box dimensions.
    Args:
    - data (dict): A dictionary containing information about the detected ball.
                    It should have the key 'ball' with a tuple representing the bounding box
                    (x, y, width, height).
    Returns:
    - depth (float): The calculated depth of the ball.
    Note: The function assumes that the actual size of the ball is 22 cm.
    """
    try:
        x, y, w, h = data['ball']
        height = abs(h)
        width = abs(w)
        depth_height = (height / 11899.432442484951) ** (1 / -0.9742053442689245)
        depth_width = (width / 10872.263895242095) ** (1 / -0.9630431825327789)
        depth = (depth_width + depth_height) / 2
        return depth
    except Exception as e:
        logging.error("Error calculating depth of ball: %s", str(e))
        return None


##### Distance #####

def distance(distances, ball_point):
    """
    Calculate distances for the given frame.
    Args:
        distances (dict): Dictionary containing distance names as keys
                          and corresponding keypoints for both sides as values.
        ball_point (str): Reference point on the ball ("center", "left_mid", "right_mid", "mid_top", "mid_bottom").
    Returns:
        dict: Dictionary containing the calculated distances.
    """
    distances_values = {}
    for distance_name, keypoints_dict in distances.items():
        foot_keypoints, ball_keypoints = keypoints_dict
        try:
            coord1 = foot_keypoints[:2]
            x, y, w, h = ball_keypoints
            coord2 = {
                "center": (x, y),
                "left_mid": (x - w / 2, y),
                "right_mid": (x + w / 2, y),
                "mid_top": (x, y - h / 2),
                "mid_bottom": (x, y + h / 2)
            }.get(ball_point)
            
            if coord2 is None:
                continue
            
            dist = np.linalg.norm(np.array(coord1) - np.array(coord2))
            distances_values[distance_name] = dist
        except Exception as e:
            logging.error("Error calculating distance: %s", str(e))
    return distances_values


def distance_joints(joint1, joint2):
    """
    Calculate the distance between two joints.
    Args:
        joint1 (tuple): Coordinates of the first joint.
        joint2 (tuple): Coordinates of the second joint.
    Returns:
        float: The calculated distance between the two joints.
    """
    try:
        return np.linalg.norm(np.array(joint1) - np.array(joint2))
    except Exception as e:
        logging.error("Error calculating distance: %s", str(e))
        return None


##### Linear Velocity #####

def velocity(distance, time):
    """
    Calculate velocities for the given distances and time.
    Args:
        distance (float): Distance value.
        time (float): Time interval.
    Returns:
        float: The calculated velocity.
    """
    if time is None or time == 0:
        logging.error("Non-zero time interval must be provided for velocity calculation.")
        return None
    try:
        return distance / time
    except Exception as e:
        logging.error("Error calculating velocity: %s", str(e))
        return None


##### Linear Acceleration #####

def acceleration(initial_velocity, final_velocity, interval):
    """
    Calculate acceleration using provided initial and final velocities and time interval.
    Args:
        initial_velocity (float): Initial velocity.
        final_velocity (float): Final velocity.
        interval (float): Time interval.
    Returns:
        float: Calculated acceleration.
    """
    if interval is None or interval == 0:
        logging.error("Non-zero time interval must be provided for acceleration calculation.")
        return None

    try:
        velocity_change = final_velocity - initial_velocity
        return velocity_change / interval
    except Exception as e:
        logging.error("Error calculating acceleration: %s", str(e))
        return None


##### Angle #####

def angles(angle):
    """
    Calculate angles for the given frame.
    Args:
        angles (dict): Dictionary containing angle names as keys
                       and corresponding keypoints for both sides as values.
    Returns:
        dict: Dictionary containing the calculated angles in degrees.
    """
    angles_deg = {}
    for angle_name, keypoints_tuple in angle.items():
        try:
            keypoints = keypoints_tuple[1]
            vector1 = np.array(keypoints_tuple[0]) - np.array(keypoints)
            vector2 = np.array(keypoints_tuple[2]) - np.array(keypoints)
            dot_product = np.dot(vector1, vector2)
            magnitude1 = np.linalg.norm(vector1)
            magnitude2 = np.linalg.norm(vector2)
            if magnitude1 * magnitude2 == 0:
                calculated_angle = None
            else:
                cosine_theta = dot_product / (magnitude1 * magnitude2)
                theta = np.arccos(np.clip(cosine_theta, -1.0, 1.0))
                calculated_angle = np.degrees(theta)
            angles_deg[angle_name] = calculated_angle
        except Exception as e:
            logging.error("Error calculating angle: %s", str(e))
    return angles_deg


##### Angular Velocity #####

def angular_velocities_angle(angle1, angle2, time):
    """
    Calculate angular velocity based on angular displacement and time.
    Args:
        angle1 (float): Initial angle in radians.
        angle2 (float): Final angle in radians.
        time (float): Time interval.
    Returns:
        float: Angular velocity.
    """
    try:
        return (angle2 - angle1) / time
    except Exception as e:
        logging.error("Error calculating angular velocity: %s", str(e))
        return None


def angular_velocities_linear_velocity(velocity, radius):
    """
    Calculate angular velocity based on linear velocity and radius.
    Args:
        velocity (float): Linear velocity.
        radius (float): Radius.
    Returns:
        float: Angular velocity.
    """
    try:
        if radius == 0:
            raise ValueError("Radius cannot be zero.")
        return velocity / radius
    except Exception as e:
        logging.error("Error calculating angular velocity: %s", str(e))
        return None


def angular_velocities_frequency(frequency):
    """
    Calculate angular velocity based on frequency.
    Args:
        frequency (float): Frequency in count/sec.
    Returns:
        float: Angular velocity.
    """
    try:
        return 2 * math.pi * frequency
    except Exception as e:
        logging.error("Error calculating angular velocity: %s", str(e))
        return None


##### Angular Acceleration #####

def angular_accelerations_angular_velocity(angular_velocity_1, angular_velocity_2, time):
    """
    Calculate angular acceleration based on two angular velocities and time.
    Args:
        angular_velocity_1 (float): Initial angular velocity.
        angular_velocity_2 (float): Final angular velocity.
        time (float): Time interval.
    Returns:
        float: Angular acceleration.
    """
    try:
        return (angular_velocity_2 - angular_velocity_1) / time
    except Exception as e:
        logging.error("Error calculating angular acceleration: %s", str(e))
        return None


def angular_accelerations_linear_acceleration(angular_acceleration, radius):
    """
    Calculate angular acceleration based on tangential acceleration and radius.
    Args:
        angular_acceleration (float): Tangential acceleration.
        radius (float): Radius.
    Returns:
        float: Angular acceleration.
    """
    try:
        if radius <= 0:
            raise ValueError("Radius cannot be zero or negative.")
        return angular_acceleration / radius
    except Exception as e:
        logging.error("Error calculating angular acceleration: %s", str(e))
        return None


##### Force #####

def forces(acceleration, mass):
    """
    Calculate force using provided acceleration and mass.
    Args:
        acceleration (float): Acceleration.
        mass (float): Mass.
    Returns:
        float: Calculated force.
    """
    try:
        return mass * acceleration
    except Exception as e:
        logging.error("Error calculating force: %s", str(e))
        return None


##### Stride Frequency #####

def stride_frequency(count, time):
    """
    Calculate stride frequency based on the number of strides and total time.
    Args:
        count (int): Number of strides taken.
        time (float): Total time taken for the strides (in minutes).
    Returns:
        float: Stride frequency calculated as strides per minute.
    """
    return (count / time) * 60


##### Gaze #####

def look_side(r_shoulder, neck, nose):
    """
    Determine the direction of gaze based on the angle between keypoints.
    Args:
        r_shoulder (tuple): Coordinates of the right shoulder.
        neck (tuple): Coordinates of the neck (midpoint).
        nose (tuple): Coordinates of the nose point.
    Returns:
        str: Direction of gaze ('right', 'left', or 'straight').
    """
    vector1 = np.array(neck) - np.array(r_shoulder)
    vector2 = np.array(neck) - np.array(nose)
    dot_product = np.dot(vector1, vector2)
    magnitude1 = np.linalg.norm(vector1)
    magnitude2 = np.linalg.norm(vector2)
    angle = np.degrees(np.arccos(dot_product / (magnitude1 * magnitude2)))
    
    if angle < 90:
        return "right"
    elif angle > 90:
        return "left"
    else:
        return "straight"


##### Head Tilt #####

def head_tilt(r_shoulder, l_shoulder, neck, r_ear, l_ear):
    """
    Determine the direction of head tilt based on the angle between keypoints.
    Args:
        r_shoulder (tuple): Coordinates of the right shoulder.
        l_shoulder (tuple): Coordinates of the left shoulder.
        neck (tuple): Coordinates of the neck.
        r_ear (tuple): Coordinates of the right ear.
        l_ear (tuple): Coordinates of the left ear.
    Returns:
        str: Direction of head tilt ('right', 'left', or 'straight').
    """
    if None in (r_shoulder, l_shoulder, neck, r_ear, l_ear):
        return "Cannot determine"
    
    vector1 = np.array(neck) - np.array(r_shoulder)
    vector2 = np.array(neck) - np.array(l_shoulder)
    vector3 = np.array(neck) - np.array(r_ear)
    vector4 = np.array(neck) - np.array(l_ear)
    
    dot_product_r = np.dot(vector1, vector3)
    dot_product_l = np.dot(vector2, vector4)
    
    magnitude1 = np.linalg.norm(vector1)
    magnitude2 = np.linalg.norm(vector2)
    magnitude3 = np.linalg.norm(vector3)
    magnitude4 = np.linalg.norm(vector4)
    
    r_angle = np.degrees(np.arccos(dot_product_r / (magnitude1 * magnitude3)))
    l_angle = np.degrees(np.arccos(dot_product_l / (magnitude2 * magnitude4)))
    
    if r_angle < l_angle:
        return "right"
    elif r_angle > l_angle:
        return "left"
    else:
        return "straight"
