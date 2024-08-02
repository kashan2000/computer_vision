import cv2
import queue
import time
import os
from datetime import datetime

def display_frames(trigger_queue, stop_flag_trigger, stop_flag_display, drill_type):
    current_count = 0
    current_trigger = False

    # Get the present working directory
    pwd = os.getcwd()
    analyzed_videos_dir = os.path.join(pwd, "analyzed_videos")
    # Create the directory if it does not exist
    if not os.path.exists(analyzed_videos_dir):
        os.makedirs(analyzed_videos_dir)
    # Generate the video file name based on the drill type and current date and time
    current_time = datetime.now().strftime("%Y%m%d%H%M%S")
    video_file_name = f"{drill_type}_{current_time}.mp4"
    video_path = os.path.join(analyzed_videos_dir, video_file_name)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = None
    cv2.namedWindow('Output', cv2.WINDOW_NORMAL)

    while not stop_flag_trigger.is_set() or not trigger_queue.empty():
        try:
            trigger_data = trigger_queue.get(timeout=1)
            frame_count = trigger_data[0]
            frame = trigger_data[1]
            current_count = trigger_data[2]
            current_trigger = trigger_data[3]
            merged_data = trigger_data[4]

            detection_data = merged_data['detection']
            pose_data = merged_data['pose']

            # Overlay bounding box if detection data is available
            if detection_data and detection_data['ball'][0] is not None:
                ball_x, ball_y, ball_w, ball_h = detection_data["ball"]
                start_point = (int((ball_x - ball_w / 2) * frame.shape[1]), int((ball_y - ball_h / 2) * frame.shape[0]))
                end_point = (int((ball_x + ball_w / 2) * frame.shape[1]), int((ball_y + ball_h / 2) * frame.shape[0]))
                cv2.rectangle(frame, start_point, end_point, (0, 255, 0), 2)

                # Draw circular threshold boundary
                center_point = (int(ball_x * frame.shape[1]), int(ball_y * frame.shape[0]))
                radius = int(0.70 * (ball_w + ball_h) / 2 * frame.shape[1])
                cv2.circle(frame, center_point, radius, (255, 0, 0), 2)

            # Overlay keypoints if pose data is available
            if pose_data:
                if 'r_ankle' in pose_data and pose_data['r_ankle'] is not None:
                    right_ankle_x, right_ankle_y = pose_data['r_ankle']
                    if right_ankle_x is not None and right_ankle_y is not None:
                        center = (int(right_ankle_x * frame.shape[1]), int(right_ankle_y * frame.shape[0]))
                        cv2.circle(frame, center, 5, (0, 0, 255), -1) # red
                        label = "R"
                        cv2.putText(frame, label, (center[0] + 10, center[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                if 'l_ankle' in pose_data and pose_data['l_ankle'] is not None:
                    left_ankle_x, left_ankle_y = pose_data['l_ankle']
                    if left_ankle_x is not None and left_ankle_y is not None:
                        center = (int(left_ankle_x * frame.shape[1]), int(left_ankle_y * frame.shape[0]))
                        cv2.circle(frame, center, 5, (0, 0, 255), -1)
                        label = "L"
                        cv2.putText(frame, label, (center[0] + 10, center[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)


            # Overlay text
            cv2.putText(frame, f"F: {frame_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 2)
            cv2.putText(frame, f"C: {current_count}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 2)
            cv2.putText(frame, f"T: {current_trigger}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 2)

            if out is None:
                height, width, layers = frame.shape
                out = cv2.VideoWriter(video_path, fourcc, 30, (width, height))

            out.write(frame)

            cv2.imshow('Output', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        except queue.Empty:
            continue

    if out is not None:
        out.release()   

    cv2.destroyAllWindows()
    print("Finished displaying frames")
    stop_flag_display.set()
