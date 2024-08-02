import cv2
from cv2 import VideoCapture, CAP_PROP_FRAME_WIDTH, CAP_PROP_FRAME_HEIGHT
from log import logging
from cv2 import putText, waitKey, imshow, destroyAllWindows
import time
import numpy as np
import base64

SKIP_FRAMES = 3
# mobile_address = "http://192.168.1.8:8080/video"

def capture_frames(frame,frame_count, capture_to_detection_queue, capture_to_pose_queue, stop_flag_capture):
    
    # frame_count = 0
    # frame_interval = 1 / 20.0  # 30 frames per second

    # cap = cv2.VideoCapture(video_path)
    # cap.set(CAP_PROP_FRAME_WIDTH, 1280)
    # cap.set(CAP_PROP_FRAME_HEIGHT, 720)
    # cap.set(cv2.CAP_PROP_FPS, 20)
    #cap.open(mobile_address)

    # while cap.isOpened():
        # start_time = time.time()
        # ret, frame = cap.read()
        # if not ret:
        #     logging.error("Error capturing video.")
        #     break
      # Decode the base64 string to bytes
    frame_bytes = base64.b64decode(frame)
    frame_np = np.frombuffer(frame_bytes, dtype=np.uint8)

    print(f"Array size: {frame_np.size}")
    
    channels = 3
    total_pixels = frame_np.size // channels
    side_length = int(total_pixels**0.5)
    
    expected_size = side_length * side_length * channels
    print(f"Expected size: {expected_size}")
    print(f"Calculated dimensions: {side_length}x{side_length}x{channels}")

    # if abs(frame_np.size - expected_size) < channels:
    frame_np = frame_np[:expected_size]
    frame_np = frame_np.reshape((side_length, side_length, channels))
    # else:
    #     frame_width = 1280
    #     frame_height = 720
    #     expected_size = frame_width * frame_height * channels
    #     if frame_np.size == expected_size:
    #         frame_np = frame_np.reshape((frame_height, frame_width, channels))
    #     else:
    #         print("no option returning")
    #         return {"error": "The size of the provided frame does not match the expected dimensions."}

    data = (frame_count, frame_np)
        # if frame_count % SKIP_FRAMES == 0:
    capture_to_detection_queue.put(data)
    capture_to_pose_queue.put(data)
        
     
            #capture_to_display_queue.put(data)

        # elapsed_time = time.time() - start_time
        # time_to_sleep = frame_interval - elapsed_time
        # if time_to_sleep > 0:
        #     time.sleep(time_to_sleep)

        # print(f"Captured frame {frame_count}")

        # frame_count += 1

        # if waitKey(1) & 0xFF == ord('q'):
        #     stop_flag_capture.set()

    # cap.release()
    # destroyAllWindows()
    # print("Finished capturing frames")