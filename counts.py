import csv
import os
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math

########## Global Variables ##########

# Ball coordinates
PREV_BALL_X = None
PREV_BALL_Y = None
PREV_BALL_H = None
PREV_BALL_W = None
PREV_BALL_AREA = None

# Right & Left Leg coordinates
PREV_TRIGGER_ANKLE = None

PREV_RIGHT_ANKLE_X = None
PREV_RIGHT_ANKLE_Y = None

PREV_LEFT_ANKLE_X = None
PREV_LEFT_ANKLE_Y = None

PREV_RIGHT_BIG_TOE_X = None
PREV_LEFT_BIG_TOE_X = None

# Other common variables
PREV_DIRECTION = -1
FRAME_THRESHOLD = 12
PREV_TRIGGER_FRAME = 0  # Initialize to a value that allows the first frame to be counted


# Common CONSTANTS
LEFT_TRIGGER_COUNT = 0
RIGHT_TRIGGER_COUNT = 0

LEFT_TRIGGER = False
RIGHT_TRIGGER = False

BALL_MOVEMENT = False

RIGHT_RISING = None
LEFT_RISING = None

PREV_COUNT_LEFT = 0
PREV_COUNT_RIGHT = 0
PREV_COUNT = 0
DELTA_Y = 0.05
########## Drill Specific Constants ##########

# Side View Drills
MIN_BALL_MOVEMENT_X_TOE_TAPS = 5
MIN_ANKLE_MOVEMENT_X_PUSH_PULL = 100
DIST_BALL_ANKLE = 0
ACTIVE_ANKLE = None
PREV_ACTIVE_ANKLE_X = None
PREV_TRIGGER_ACTION = ""

# Front View Drills
MIN_BALL_MOVEMENT_X_INSIDE_TAPS = 50
MIN_BALL_MOVEMENT_X_INSIDE_OUTSIDE_TAPS = 5
PREV_TRIGGER_X = None
MIN_ANKLE_MOVEMENT_X_SCISSORS = 20
MIN_MOVEMENT_X_V_PUSH_PULL = 50
MIN_DISTANCE_THRESHOLD = None

plot_data = []

######################################### SIDE VIEW DRILLS #########################################

def toe_tap_trigger(data, frame_count):
    global PREV_BALL_X, PREV_RIGHT_ANKLE_X, PREV_LEFT_ANKLE_X, PREV_RIGHT_ANKLE_Y, PREV_LEFT_ANKLE_Y, ACTIVE_ANKLE, PREV_TRIGGER_ANKLE, PREV_COUNT_LEFT, PREV_COUNT_RIGHT
    global LEFT_TRIGGER, RIGHT_TRIGGER, PREV_TRIGGER_FRAME, FRAME_THRESHOLD

    ball_x, ball_y, ball_w, ball_h = data["detection"]["ball"]

    if ball_x is None or ball_y is None or ball_w is None or ball_h is None:
        # Ball not detected, so we skip this frame
        return 0, False

    right_ankle_x = data["pose"]["r_ankle"][0]
    right_ankle_y = data["pose"]["r_ankle"][1]
    left_ankle_x = data["pose"]["l_ankle"][0]
    left_ankle_y = data["pose"]["l_ankle"][1]

    if None in (right_ankle_x, right_ankle_y, left_ankle_x, left_ankle_y):
        # Ankle points not detected so we skip this frame
        return 0, False

    trigger = False
    count = 0

    # Initialize variables if None
    if PREV_BALL_X is None:
        PREV_BALL_X = ball_x
    if PREV_COUNT_LEFT == 0:
        PREV_COUNT_LEFT = count
    if PREV_COUNT_RIGHT == 0:
        PREV_COUNT_RIGHT = count    
    if PREV_RIGHT_ANKLE_X is None:
        PREV_RIGHT_ANKLE_X = right_ankle_x
    if PREV_LEFT_ANKLE_X is None:
        PREV_LEFT_ANKLE_X = left_ankle_x
    if PREV_RIGHT_ANKLE_Y is None:
        PREV_RIGHT_ANKLE_Y = right_ankle_y
    if PREV_LEFT_ANKLE_Y is None:
        PREV_LEFT_ANKLE_Y = left_ankle_y

    # MIN_DISTANCE_THRESHOLD = 0.70 * (ball_w + ball_h)
    MIN_DISTANCE_THRESHOLD =  0.90 (ball_w + ball_h)

    # Determine the active ankle through y condition
    if left_ankle_y < right_ankle_y:
        ACTIVE_ANKLE = "left"
    else:
        ACTIVE_ANKLE = "right"

    # Calculate the distances between the ball and the ankles
    left_ankle_distance = np.sqrt((left_ankle_x - ball_x) ** 2 + (left_ankle_y - ball_y) ** 2)
    right_ankle_distance = np.sqrt((right_ankle_x - ball_x) ** 2 + (right_ankle_y - ball_y) ** 2)

    # Check for active ankle movement to touch the ball
    if ACTIVE_ANKLE == "left":
        if left_ankle_distance <= MIN_DISTANCE_THRESHOLD and left_ankle_y > PREV_LEFT_ANKLE_Y:
            if left_ankle_y < ball_y and frame_count - PREV_TRIGGER_FRAME >= FRAME_THRESHOLD:
                LEFT_TRIGGER = True
            else:
                LEFT_TRIGGER = False
        else:
            LEFT_TRIGGER = False
            
    elif ACTIVE_ANKLE == "right":
        if right_ankle_distance <= MIN_DISTANCE_THRESHOLD and right_ankle_y > PREV_RIGHT_ANKLE_Y:
            if right_ankle_y < ball_y and frame_count - PREV_TRIGGER_FRAME >= FRAME_THRESHOLD:
                RIGHT_TRIGGER = True
            else:
                RIGHT_TRIGGER = False
        else:
            RIGHT_TRIGGER = False

    # Conditions to reset triggers
    if LEFT_TRIGGER:
        trigger = LEFT_TRIGGER
        if ACTIVE_ANKLE != PREV_TRIGGER_ANKLE:
            PREV_COUNT_LEFT += 1
            PREV_TRIGGER_FRAME = frame_count
            LEFT_TRIGGER = False
            PREV_TRIGGER_ANKLE = "left"

    if RIGHT_TRIGGER:
        trigger = RIGHT_TRIGGER
        if ACTIVE_ANKLE != PREV_TRIGGER_ANKLE:
            PREV_COUNT_RIGHT += 1
            PREV_TRIGGER_FRAME = frame_count
            RIGHT_TRIGGER = False
            PREV_TRIGGER_ANKLE = "right"

    PREV_BALL_X = ball_x
    PREV_RIGHT_ANKLE_X = right_ankle_x
    PREV_RIGHT_ANKLE_Y = right_ankle_y
    PREV_LEFT_ANKLE_X = left_ankle_x
    PREV_LEFT_ANKLE_Y = left_ankle_y
    count = min(PREV_COUNT_LEFT, PREV_COUNT_RIGHT)

    return count, trigger



def push_pull_trigger(data, frame_count):
    global PREV_BALL_X, ACTIVE_ANKLE, PREV_DIRECTION, PREV_LEFT_ANKLE_X, PREV_RIGHT_ANKLE_X, PREV_LEFT_ANKLE_Y, PREV_RIGHT_ANKLE_Y
    global PREV_TRIGGER_ACTION, PREV_COUNT, PREV_TRIGGER_FRAME, FRAME_THRESHOLD, MIN_DISTANCE_THRESHOLD

    ball_x, ball_y, ball_w, ball_h = data["detection"]["ball"]

    if ball_x is None or ball_y is None or ball_w is None or ball_h is None:
        # Ball not detected, so we skip this frame
        return 0, False

    right_ankle_x = data["pose"]["r_ankle"][0]
    right_ankle_y = data["pose"]["r_ankle"][1]
    left_ankle_x = data["pose"]["l_ankle"][0]
    left_ankle_y = data["pose"]["l_ankle"][1]

    if None in (right_ankle_x, right_ankle_y, left_ankle_x, left_ankle_y):
        # Ankle points not detected so we skip this frame
        return 0, False

    trigger = False
    count = 0
    action = ""

    MIN_DISTANCE_THRESHOLD = 0.70 * (ball_w + ball_h)

    if PREV_BALL_X is None:
        PREV_BALL_X = ball_x  
    if PREV_RIGHT_ANKLE_X is None:
        PREV_RIGHT_ANKLE_X = right_ankle_x
    if PREV_LEFT_ANKLE_X is None:
        PREV_LEFT_ANKLE_X = left_ankle_x
    if PREV_RIGHT_ANKLE_Y is None:
        PREV_RIGHT_ANKLE_Y = right_ankle_y
    if PREV_LEFT_ANKLE_Y is None:
        PREV_LEFT_ANKLE_Y = left_ankle_y
    if PREV_COUNT == 0:
        PREV_COUNT = count
    if PREV_TRIGGER_ACTION == "":
        PREV_TRIGGER_ACTION = action

    ACTIVE_ANKLE = 'right' if right_ankle_y < left_ankle_y else 'left'

    direction = 1 if ball_x > PREV_BALL_X else -1  # this is the direction of ball movement, not foot movement

    # Calculate the distances between the ball and the ankles
    left_ankle_distance = np.sqrt((left_ankle_x - ball_x) ** 2 + (left_ankle_y - ball_y) ** 2)
    right_ankle_distance = np.sqrt((right_ankle_x - ball_x) ** 2 + (right_ankle_y - ball_y) ** 2)

    inter_ankle_distance = abs(left_ankle_x - right_ankle_x)

    if ACTIVE_ANKLE == "left":
        if direction != PREV_DIRECTION and left_ankle_distance <= MIN_DISTANCE_THRESHOLD and frame_count - PREV_TRIGGER_FRAME >= FRAME_THRESHOLD:
            #action = "push"
            trigger = True

        elif direction != PREV_DIRECTION and left_ankle_distance <= MIN_DISTANCE_THRESHOLD and frame_count - PREV_TRIGGER_FRAME >= FRAME_THRESHOLD:
            #action = "pull"
            trigger = True

    elif ACTIVE_ANKLE == "right":
        if direction != PREV_DIRECTION and right_ankle_distance <= MIN_DISTANCE_THRESHOLD and frame_count - PREV_TRIGGER_FRAME >= FRAME_THRESHOLD:
            #action = "push"
            trigger = True

        elif direction != PREV_DIRECTION and right_ankle_distance <= MIN_DISTANCE_THRESHOLD and frame_count - PREV_TRIGGER_FRAME >= FRAME_THRESHOLD:
            #action = "pull"
            trigger = True

    if trigger:
        PREV_COUNT += 1
        #PREV_TRIGGER_ACTION = action
        PREV_DIRECTION = direction
        PREV_TRIGGER_FRAME = frame_count
        trigger = False

    PREV_BALL_X = ball_x
    PREV_RIGHT_ANKLE_X = right_ankle_x
    PREV_LEFT_ANKLE_X = left_ankle_x
    PREV_RIGHT_ANKLE_Y = right_ankle_y
    PREV_LEFT_ANKLE_Y = left_ankle_y

    count = PREV_COUNT // 2
    return count, trigger

######################################### FRONT VIEW DRILLS #########################################

def push_pull_left_trigger(data, frame_count):
    global PREV_BALL_AREA, ACTIVE_ANKLE, PREV_DIRECTION, PREV_LEFT_ANKLE_X, PREV_RIGHT_ANKLE_X, PREV_LEFT_ANKLE_Y, PREV_RIGHT_ANKLE_Y, PREV_TRIGGER_ACTION
    global PREV_TRIGGER_ACTION, PREV_COUNT, PREV_TRIGGER_FRAME, FRAME_THRESHOLD, MIN_DISTANCE_THRESHOLD

    ball_x, ball_y, ball_w, ball_h = data["detection"]["ball"]

    if ball_x is None or ball_y is None or ball_w is None or ball_h is None:
        # Ball not detected, so we skip this frame
        return 0, False

    left_ankle_x = data["pose"]["l_ankle"][0]
    left_ankle_y = data["pose"]["l_ankle"][1]

    if None in (left_ankle_x, left_ankle_y):
        # Ankle points not detected so we skip this frame
        return 0, False
    
    trigger = False
    count = 0
    ball_area = ball_w * ball_h
    action = ""

    if PREV_BALL_AREA is None:
        PREV_BALL_AREA = ball_area 
    if PREV_LEFT_ANKLE_X is None:
        PREV_LEFT_ANKLE_X = left_ankle_x
    if PREV_LEFT_ANKLE_Y is None:
        PREV_LEFT_ANKLE_Y = left_ankle_y
    if PREV_COUNT == 0:
        PREV_COUNT = count
    if PREV_TRIGGER_ACTION == "":
        PREV_TRIGGER_ACTION = action
    
    ACTIVE_ANKLE = "left"
    FRAME_THRESHOLD = 12
    PREV_TRIGGER_FRAME = -FRAME_THRESHOLD

    direction = 1 if ball_area > PREV_BALL_AREA else -1


    if direction != PREV_DIRECTION and left_ankle_y > (ball_y - 0.5*ball_h) and frame_count - PREV_TRIGGER_FRAME >= FRAME_THRESHOLD:
        trigger = True
        action = "push"
    
    elif direction != PREV_DIRECTION and left_ankle_y < (ball_y - 0.5*ball_h) and frame_count - PREV_TRIGGER_FRAME >= FRAME_THRESHOLD:
        trigger = True
        action = "pull"
    

    if trigger and action != PREV_TRIGGER_ACTION:
        PREV_COUNT+=1
        PREV_TRIGGER_ACTION = action
        PREV_DIRECTION = direction
        PREV_TRIGGER_FRAME = frame_count
        trigger = False
    
    PREV_BALL_AREA = ball_area
    PREV_LEFT_ANKLE_X = left_ankle_x
    PREV_LEFT_ANKLE_Y = left_ankle_y
    count = PREV_COUNT // 2

    return count, trigger

def push_pull_right_trigger(data, frame_count):
    global PREV_BALL_AREA, ACTIVE_ANKLE, PREV_DIRECTION, PREV_LEFT_ANKLE_X, PREV_RIGHT_ANKLE_X, PREV_LEFT_ANKLE_Y, PREV_RIGHT_ANKLE_Y, PREV_TRIGGER_ACTION
    global PREV_TRIGGER_ACTION, PREV_COUNT, PREV_TRIGGER_FRAME, FRAME_THRESHOLD, MIN_DISTANCE_THRESHOLD

    ball_x, ball_y, ball_w, ball_h = data["detection"]["ball"]

    if ball_x is None or ball_y is None or ball_w is None or ball_h is None:
        # Ball not detected, so we skip this frame
        return 0, False

    right_ankle_x = data["pose"]["r_ankle"][0]
    right_ankle_y = data["pose"]["r_ankle"][1]

    if None in (right_ankle_x, right_ankle_y):
        # Ankle points not detected so we skip this frame
        return 0, False
    
    trigger = False
    count = 0
    ball_area = ball_w * ball_h
    action = ""

    if PREV_BALL_AREA is None:
        PREV_BALL_AREA = ball_area 
    if PREV_RIGHT_ANKLE_X is None:
        PREV_RIGHT_ANKLE_X = right_ankle_x
    if PREV_RIGHT_ANKLE_Y is None:
        PREV_RIGHT_ANKLE_Y = right_ankle_y
    if PREV_COUNT == 0:
        PREV_COUNT = count
    if PREV_TRIGGER_ACTION == "":
        PREV_TRIGGER_ACTION = action
    
    ACTIVE_ANKLE = "right"
    FRAME_THRESHOLD = 12
    PREV_TRIGGER_FRAME = -FRAME_THRESHOLD

    direction = 1 if ball_area > PREV_BALL_AREA else -1


    if direction != PREV_DIRECTION and right_ankle_y > (ball_y - 3*ball_h/4) and frame_count - PREV_TRIGGER_FRAME >= FRAME_THRESHOLD:
        trigger = True
        action = "push"
    
    elif direction != PREV_DIRECTION and right_ankle_y < (ball_y - 3*ball_h/4) and frame_count - PREV_TRIGGER_FRAME >= FRAME_THRESHOLD:
        trigger = True
        action = "pull"
    

    if trigger and action != PREV_TRIGGER_ACTION:
        PREV_COUNT+=1
        PREV_TRIGGER_ACTION = action
        PREV_DIRECTION = direction
        PREV_TRIGGER_FRAME = frame_count
        trigger = False
    
    PREV_BALL_AREA = ball_area
    PREV_RIGHT_ANKLE_X = right_ankle_x
    PREV_RIGHT_ANKLE_Y = right_ankle_y
    count = PREV_COUNT // 2

    return count, trigger


def v_push_pull_trigger(data, frame_count):
    global PREV_BALL_X, ACTIVE_ANKLE, PREV_DIRECTION, PREV_LEFT_ANKLE_X, PREV_RIGHT_ANKLE_X, PREV_LEFT_ANKLE_Y, PREV_RIGHT_ANKLE_Y
    global PREV_TRIGGER_ACTION, PREV_COUNT, PREV_TRIGGER_FRAME, FRAME_THRESHOLD, MIN_DISTANCE_THRESHOLD

    ball_x, ball_y, ball_w, ball_h = data["detection"]["ball"]

    if ball_x is None or ball_y is None or ball_w is None or ball_h is None:
        # Ball not detected, so we skip this frame
        return 0, False

    right_ankle_x = data["pose"]["r_ankle"][0]
    right_ankle_y = data["pose"]["r_ankle"][1]
    left_ankle_x = data["pose"]["l_ankle"][0]
    left_ankle_y = data["pose"]["l_ankle"][1]

    if None in (right_ankle_x, right_ankle_y, left_ankle_x, left_ankle_y):
        # Ankle points not detected so we skip this frame
        return 0, False
    
    trigger = False
    count = 0
    action = ""

    MIN_DISTANCE_THRESHOLD = 0.70 * (ball_w + ball_h)

    if PREV_BALL_X is None:
        PREV_BALL_X = ball_x  
    if PREV_RIGHT_ANKLE_X is None:
        PREV_RIGHT_ANKLE_X = right_ankle_x
    if PREV_LEFT_ANKLE_X is None:
        PREV_LEFT_ANKLE_X = left_ankle_x
    if PREV_RIGHT_ANKLE_Y is None:
        PREV_RIGHT_ANKLE_Y = right_ankle_y
    if PREV_LEFT_ANKLE_Y is None:
        PREV_LEFT_ANKLE_Y = left_ankle_y
    if PREV_COUNT == 0:
        PREV_COUNT = count
    if PREV_TRIGGER_ACTION == "":
        PREV_TRIGGER_ACTION = action

    FRAME_THRESHOLD = 6
    PREV_TRIGGER_FRAME = -FRAME_THRESHOLD

    # Calculate the distances between the ball and the ankles
    left_ankle_distance = np.sqrt((left_ankle_x - ball_x) ** 2 + (left_ankle_y - ball_y) ** 2)
    right_ankle_distance = np.sqrt((right_ankle_x - ball_x) ** 2 + (right_ankle_y - ball_y) ** 2)

    ACTIVE_ANKLE = 'right' if abs(right_ankle_x - ball_x) <= abs(left_ankle_x - ball_x) else 'left'

    direction = 1 if ball_x > PREV_BALL_X else -1  # this is the direction of ball movement, not foot movement

    if ACTIVE_ANKLE == "left":
        if direction != PREV_DIRECTION and left_ankle_y < (ball_y - ball_h/2) and frame_count - PREV_TRIGGER_FRAME >= FRAME_THRESHOLD:
            action = "lpull"
            trigger = True

        elif PREV_TRIGGER_ACTION == "lpull" and left_ankle_y > (ball_y - ball_h/2) and frame_count - PREV_TRIGGER_FRAME >= FRAME_THRESHOLD:
            action = "lpush"
            trigger = True

    elif ACTIVE_ANKLE == "right":
        if direction != PREV_DIRECTION and right_ankle_y < (ball_y - ball_h/2) and frame_count - PREV_TRIGGER_FRAME >= FRAME_THRESHOLD:
            action = "rpull"
            trigger = True

        elif PREV_TRIGGER_ACTION == "rpull" and right_ankle_y > (ball_y - ball_h/2) and frame_count - PREV_TRIGGER_FRAME >= FRAME_THRESHOLD:
            action = "rpush"
            trigger = True

    if trigger and action == "lpull" and PREV_TRIGGER_ACTION != "lpull":
        PREV_COUNT+=1
        PREV_TRIGGER_ACTION = action
        PREV_DIRECTION = direction
        PREV_TRIGGER_FRAME = frame_count
        trigger = False

    if trigger and action == "lpush" and PREV_TRIGGER_ACTION != "lpush":
        PREV_COUNT+=1
        PREV_TRIGGER_ACTION = action
        PREV_DIRECTION = direction
        PREV_TRIGGER_FRAME = frame_count
        trigger = False

    if trigger and action == "rpull" and PREV_TRIGGER_ACTION != "rpull":
        PREV_COUNT+=1
        PREV_TRIGGER_ACTION = action
        PREV_DIRECTION = direction
        PREV_TRIGGER_FRAME = frame_count
        trigger = False

    if trigger and action == "rpush" and PREV_TRIGGER_ACTION != "rpush":
        PREV_COUNT+=1
        PREV_TRIGGER_ACTION = action
        PREV_DIRECTION = direction
        PREV_TRIGGER_FRAME = frame_count
        trigger = False
    
    PREV_BALL_X = ball_x
    PREV_RIGHT_ANKLE_X = right_ankle_x
    PREV_LEFT_ANKLE_X = left_ankle_x
    PREV_RIGHT_ANKLE_Y = right_ankle_y
    PREV_LEFT_ANKLE_Y = left_ankle_y

    count = PREV_COUNT // 4
    return count, trigger


def roll_across_trigger(data, frame_count):
    global PREV_BALL_X, ACTIVE_ANKLE, PREV_ACTIVE_ANKLE_X, PREV_DIRECTION, PREV_TRIGGER_X, PREV_TRIGGER_ACTION
    global PREV_COUNT, PREV_TRIGGER_FRAME, FRAME_THRESHOLD, MIN_DISTANCE_THRESHOLD

    ball_x, ball_y, ball_w, ball_h = data["detection"]["ball"]

    if ball_x is None or ball_y is None or ball_w is None or ball_h is None:
        # Ball not detected, so we skip this frame
        return 0, False
    
    right_ankle_x = data["pose"]["r_ankle"][0]
    right_ankle_y = data["pose"]["r_ankle"][1]
    left_ankle_x = data["pose"]["l_ankle"][0]
    left_ankle_y = data["pose"]["l_ankle"][1]

    if None in (right_ankle_x, right_ankle_y, left_ankle_x, left_ankle_y):
        # Ankle points not detected so we skip this frame
        return 0, False
    
    trigger = False
    count = 0
    action = ""

    FRAME_THRESHOLD = 6

    MIN_DISTANCE_THRESHOLD = 0.6 * (ball_w + ball_h)

    if PREV_BALL_X is None:
        PREV_BALL_X = ball_x  
    if ACTIVE_ANKLE is None:
        ACTIVE_ANKLE = "right" if right_ankle_y < left_ankle_y else "left"
    if PREV_TRIGGER_X is None:
        PREV_TRIGGER_X = 0
    if PREV_COUNT == 0:
        PREV_COUNT = count
    if PREV_TRIGGER_FRAME == 0:
        PREV_TRIGGER_FRAME = -FRAME_THRESHOLD
    if PREV_TRIGGER_ACTION == "":
        PREV_TRIGGER_ACTION = action
    

    left_ankle_distance = abs(left_ankle_x - ball_x)
    right_ankle_distance = abs(right_ankle_x - ball_x)
    inter_ankle_distance = abs(left_ankle_x - right_ankle_x)

    ACTIVE_ANKLE = "left" if left_ankle_distance < right_ankle_distance else "right"

    direction = 1 if ball_x > PREV_BALL_X else -1

    if ACTIVE_ANKLE == "right":
        if direction != PREV_DIRECTION and PREV_TRIGGER_ACTION != "rroll" and right_ankle_y < (ball_y - ball_h/2) and right_ankle_y < left_ankle_y and abs(ball_x - PREV_TRIGGER_X) > ball_w and frame_count - PREV_TRIGGER_FRAME >= FRAME_THRESHOLD:
            trigger = True
            action = "rroll"
        elif right_ankle_x < PREV_TRIGGER_X and inter_ankle_distance < ball_w and PREV_TRIGGER_ACTION == "rroll" and frame_count - PREV_TRIGGER_FRAME >= FRAME_THRESHOLD:
            trigger = True
            action = "rcross"
            
        else:
            trigger = False
    elif ACTIVE_ANKLE == "left":
        if direction != PREV_DIRECTION and PREV_TRIGGER_ACTION != "lroll" and left_ankle_y < (ball_y - ball_h/2) and left_ankle_y < right_ankle_y and abs(ball_x - PREV_TRIGGER_X) > ball_w and frame_count - PREV_TRIGGER_FRAME >= FRAME_THRESHOLD:
            trigger = True
            action = "lroll"
        elif left_ankle_x > PREV_TRIGGER_X and inter_ankle_distance < ball_w and PREV_TRIGGER_ACTION == "lroll" and frame_count - PREV_TRIGGER_FRAME >= FRAME_THRESHOLD:
            trigger = True
            action = "lcross"
        else:
            trigger = False

    if trigger:
        if action == "rroll" or action == "lroll":
            PREV_TRIGGER_X = ball_x
        elif action == "lcross":
            PREV_TRIGGER_X = left_ankle_x
        else:
            PREV_TRIGGER_X = right_ankle_x
        PREV_COUNT+=0.25
        PREV_TRIGGER_ACTION = action
        PREV_DIRECTION = direction
        PREV_TRIGGER_FRAME = frame_count
        trigger = False
    
    PREV_BALL_X = ball_x

    count = PREV_COUNT - 0.25
    return count, trigger


def inside_tap_trigger(data, frame_count):
    global PREV_BALL_X, PREV_DIRECTION, PREV_COUNT, MIN_DISTANCE_THRESHOLD, ACTIVE_ANKLE, PREV_TRIGGER_FRAME, FRAME_THRESHOLD, PREV_TRIGGER_X
    global PREV_TRIGGER_ANKLE
    ball_x, ball_y, ball_w, ball_h = data["detection"]["ball"]

    if ball_x is None or ball_y is None or ball_w is None or ball_h is None:
        # Ball not detected, so we skip this frame
        return 0, False

    right_ankle_x = data["pose"]["r_ankle"][0]
    right_ankle_y = data["pose"]["r_ankle"][1]
    left_ankle_x = data["pose"]["l_ankle"][0]
    left_ankle_y = data["pose"]["l_ankle"][1]

    if None in (right_ankle_x, right_ankle_y, left_ankle_x, left_ankle_y):
        # Ankle points not detected so we skip this frame
        return 0, False
    
    MIN_DISTANCE_THRESHOLD = 0.6 * (ball_w + ball_h)
    FRAME_THRESHOLD = 6
    trigger = False
    count = 0

    if PREV_BALL_X is None:
        PREV_BALL_X = ball_x  
    if PREV_COUNT == 0:
        PREV_COUNT = count
    if PREV_TRIGGER_X is None:
        PREV_TRIGGER_X = 0
    if PREV_TRIGGER_FRAME <= 0:
        PREV_TRIGGER_FRAME = -FRAME_THRESHOLD

    left_ankle_distance = abs(left_ankle_x - ball_x)
    right_ankle_distance = abs(right_ankle_x - ball_x)

    ACTIVE_ANKLE = "left" if left_ankle_distance < right_ankle_distance else "right"
    direction = 1 if ball_x > PREV_BALL_X else -1

    if ACTIVE_ANKLE == "right":
        if direction != PREV_DIRECTION and PREV_TRIGGER_ANKLE != "right" and right_ankle_y > ball_y - ball_h:
            trigger = True
        else:
            trigger = False
    elif ACTIVE_ANKLE == "left":
        if direction != PREV_DIRECTION and PREV_TRIGGER_ANKLE != "left" and left_ankle_y > ball_y - ball_h:
            trigger = True
        else:
            trigger = False

    if trigger:
        if ACTIVE_ANKLE == "right":
            PREV_TRIGGER_ANKLE = "right"
        else:
            PREV_TRIGGER_ANKLE = "left"
        PREV_COUNT+=0.5
        PREV_DIRECTION = direction
        PREV_TRIGGER_FRAME = frame_count
        trigger = False
    
    PREV_BALL_X = ball_x
    count = PREV_COUNT - 0.5

    return count, trigger

def inside_outside_left_trigger(data, frame_count):
    global PREV_BALL_X, PREV_DIRECTION, PREV_COUNT, MIN_DISTANCE_THRESHOLD, PREV_TRIGGER_FRAME, FRAME_THRESHOLD, PREV_TRIGGER_X, PREV_TRIGGER_ACTION

    ball_x, ball_y, ball_w, ball_h = data["detection"]["ball"]

    if ball_x is None or ball_y is None or ball_w is None or ball_h is None:
        # Ball not detected, so we skip this frame
        return 0, False

    left_ankle_x = data["pose"]["l_ankle"][0]
    left_ankle_y = data["pose"]["l_ankle"][1]

    if None in (left_ankle_x, left_ankle_y):
        # Ankle points not detected so we skip this frame
        return 0, False
    
    MIN_DISTANCE_THRESHOLD = 0.6 * (ball_w + ball_h)
    FRAME_THRESHOLD = 6
    trigger = False
    count = 0

    if PREV_BALL_X is None:
        PREV_BALL_X = ball_x  
    if PREV_COUNT == 0:
        PREV_COUNT = count
    if PREV_TRIGGER_FRAME <= 0:
        PREV_TRIGGER_FRAME = -FRAME_THRESHOLD
    if PREV_TRIGGER_X is None:
        PREV_TRIGGER_X = ball_x

    direction = 1 if ball_x > PREV_BALL_X else -1
    action = "in" if left_ankle_x < ball_x else "out"

    left_ankle_distance = abs(left_ankle_x - ball_x)
  
    if direction != PREV_DIRECTION and left_ankle_distance < ball_w and action != PREV_TRIGGER_ACTION:
        trigger = True
    
    if trigger:
        PREV_COUNT+=0.5
        direction = PREV_DIRECTION
        PREV_TRIGGER_ACTION = action
    
    count = PREV_COUNT
    PREV_BALL_X = ball_x

    return count, trigger
    

def inside_outside_right_trigger(data, frame_count):
    global PREV_BALL_X, PREV_DIRECTION, PREV_COUNT, MIN_DISTANCE_THRESHOLD, PREV_TRIGGER_FRAME, FRAME_THRESHOLD, PREV_TRIGGER_X, PREV_TRIGGER_ACTION
    global BALL_MOVEMENT

    ball_x, ball_y, ball_w, ball_h = data["detection"]["ball"]

    if ball_x is None or ball_y is None or ball_w is None or ball_h is None:
        # Ball not detected, so we skip this frame
        return 0, False

    right_ankle_x = data["pose"]["r_ankle"][0]
    right_ankle_y = data["pose"]["r_ankle"][1]

    if None in (right_ankle_x, right_ankle_y):
        # Ankle points not detected so we skip this frame
        return 0, False
    
    MIN_DISTANCE_THRESHOLD = 0.6 * (ball_w + ball_h)
    FRAME_THRESHOLD = 6
    trigger = False
    count = 0

    if PREV_BALL_X is None:
        PREV_BALL_X = ball_x  
    if PREV_COUNT == 0:
        PREV_COUNT = count
    if PREV_TRIGGER_FRAME <= 0:
        PREV_TRIGGER_FRAME = -FRAME_THRESHOLD
    if PREV_TRIGGER_X is None:
        PREV_TRIGGER_X = ball_x

    direction = 1 if ball_x > PREV_BALL_X else -1
    action = "in" if right_ankle_x > ball_x else "out"
    BALL_MOVEMENT = True if abs(ball_x - PREV_BALL_X) > ball_w/40 else False

    right_ankle_distance = abs(right_ankle_x - ball_x)
  
    if direction != PREV_DIRECTION and right_ankle_distance < ball_w and action!=PREV_TRIGGER_ACTION and BALL_MOVEMENT:
        trigger = True
    
    if trigger:
        PREV_COUNT+=0.5
        PREV_DIRECTION = direction
        PREV_TRIGGER_ACTION = action
    
    count = PREV_COUNT - 0.5
    PREV_BALL_X = ball_x

    return count, trigger
# def scissors_trigger(data, frame_count):
#     global PREV_BALL_X, PREV_RIGHT_ANKLE_X, PREV_LEFT_ANKLE_X, LEFT_TRIGGER_COUNT, RIGHT_TRIGGER_COUNT
#     ball_x = data["detection"]["ball"][0]
#     ball_w = data["detection"]["ball"][2]
#     left_ankle_x = data["pose"]["l_ankle"][0]
#     right_ankle_x = data["pose"]['r_ankle'][0]
#     left_big_toe_x = data["pose"]["left_big_toe"][0]
#     right_big_toe_x = data["pose"]["right_big_toe"][0]
#     trigger = False

#     if None in (PREV_BALL_X, PREV_LEFT_ANKLE_X, PREV_RIGHT_ANKLE_X):
#         PREV_BALL_X = ball_x
#         PREV_LEFT_ANKLE_X = left_ankle_x
#         PREV_RIGHT_ANKLE_X = right_ankle_x

#     ball_position = right_ankle_x < ball_x < left_ankle_x

#     if abs(right_ankle_x - left_ankle_x) > abs(PREV_RIGHT_ANKLE_X - PREV_LEFT_ANKLE_X):
#         if (ball_x - ball_w) < right_big_toe_x < (ball_x + ball_w) and abs(PREV_RIGHT_ANKLE_X - right_ankle_x) > MIN_ANKLE_MOVEMENT_X_SCISSORS:
#             trigger = True
#             RIGHT_TRIGGER_COUNT += 1
#         if (ball_x - ball_w) < left_big_toe_x < (ball_x + ball_w) and abs(PREV_LEFT_ANKLE_X - left_ankle_x) > MIN_ANKLE_MOVEMENT_X_SCISSORS:
#             trigger = True
#             LEFT_TRIGGER_COUNT += 1

#     timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
#     write_to_csv(f'csv/scissors-{timestamp}.csv', 
#                  ['frame_count', 'Prev ball', 'ball_x', 'ball_w', 'left_big_toe', 'right_big_toe', 'left_ankle', 'right_ankle', 'trigger', 'LEFT_TRIGGER_COUNT', 'RIGHT_TRIGGER_COUNT'],
#                  [frame_count, PREV_BALL_X, ball_x, ball_w, left_big_toe_x, right_big_toe_x, left_ankle_x, right_ankle_x, trigger, LEFT_TRIGGER_COUNT, RIGHT_TRIGGER_COUNT])

#     PREV_BALL_X = ball_x
#     PREV_LEFT_ANKLE_X = left_ankle_x
#     PREV_RIGHT_ANKLE_X = right_ankle_x

#     count = LEFT_TRIGGER_COUNT if LEFT_TRIGGER_COUNT == RIGHT_TRIGGER_COUNT else 0
#     return count, trigger


# Utility functions for CSV logging
def write_to_csv(filename, column_names, row):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, mode='a', newline='', encoding="utf-8") as file:
        writer = csv.writer(file)
        if file.tell() == 0:
            writer.writerow(column_names)
        writer.writerow(row)

def save_plots(plot_data, output_dir="graphs"):
    os.makedirs(output_dir, exist_ok=True)
    df = pd.DataFrame(plot_data)

    # Plot for X positions
    plt.figure(figsize=(10, 6))
    plt.plot(df["frame_count"], df["ball_x"], label="Ball X", color="blue")
    plt.plot(df["frame_count"], df["right_ankle_x"], label="Right ankle X", color="green", linestyle="dotted")
    plt.plot(df["frame_count"], df["left_ankle_x"], label="Left ankle X", color="red", linestyle="dotted")
    plt.xlabel("Frame Count")
    plt.ylabel("Position (X)")
    plt.title("Position of Ball and Feet over Frame Count")
    plt.legend()
    plt.savefig(os.path.join(output_dir, "position_x_plot.png"))
    plt.close()

    # Plot for Y positions
    plt.figure(figsize=(10, 6))
    plt.plot(df["frame_count"], df["ball_y"], label="Ball Y", color="blue")
    plt.plot(df["frame_count"], df["right_ankle_y"], label="Right ankle Y", color="green", linestyle="dotted")
    plt.plot(df["frame_count"], df["left_ankle_y"], label="Left ankle Y", color="red", linestyle="dotted")
    plt.xlabel("Frame Count")
    plt.ylabel("Position (Y)")
    plt.title("Position of Ball and Feet over Frame Count")
    plt.legend()
    plt.savefig(os.path.join(output_dir, "position_y_plot.png"))
    plt.close()

    print(f"Plots saved to {output_dir}")

