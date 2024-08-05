import torch
import time
import numpy as np
import base64
import cv2
from PIL import Image
import asyncio
from fastapi import FastAPI, WebSocket
from pydantic import BaseModel
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from detection import process_detection
from pose import process_pose
from trigger import get_trigger
from starlette.websockets import WebSocketDisconnect
import logging

app = FastAPI()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FrameData(BaseModel):
    frame: str  # Base64 encoded JPEG
    frame_count: int
    drill_type: str
    device: str = "cpu"
    width: int
    height: int

# Global counter
frame_counter = 0

detection_results_dict = {}
pose_results_dict = {}
trigger_results_dict = {}

async def process_frame(frame_data: FrameData):
    global frame_counter
    frame = frame_data.frame
    frame_count = frame_data.frame_count
    drill_type = frame_data.drill_type
    device = frame_data.device
    expected_width = frame_data.width
    expected_height = frame_data.height

 
    torch.device("cuda" if device == "cuda" else "cpu")
    
    start_time = time.time()
    
    try:
        # Decode base64 JPEG
        frame_bytes = base64.b64decode(frame)
        
        # Convert bytes to numpy array
        frame_np = cv2.imdecode(np.frombuffer(frame_bytes, np.uint8), cv2.IMREAD_COLOR)
        
        if frame_np is None:
            raise ValueError("Failed to decode image")
        
        # Convert BGR to RGB
        frame_np = cv2.cvtColor(frame_np, cv2.COLOR_BGR2RGB)

        height, width, channels = frame_np.shape

        logger.info(f"Image dimensions: {width}x{height}x{channels}")
        logger.info(f"Expected dimensions: {expected_width}x{expected_height}")

        if width != expected_width or height != expected_height:
            logger.warning(f"Image dimensions do not match expected dimensions")
            frame_np = cv2.resize(frame_np, (expected_width, expected_height))
            logger.info(f"Resized image to {expected_width}x{expected_height}")

        # For debugging: Save the image to disk to verify its correctness
        img = Image.fromarray(frame_np, 'RGB') 
        img.save(f'debug_image.jpg')
        # logger.info(f'Saved image to debug_image_{frame_count}.jpg in RGB format')

        data = (frame_count, frame_np)

        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = {
                executor.submit(process_detection, data): 'detection',
                executor.submit(process_pose, data): 'pose'
            }

            for future in as_completed(futures):
                task_type = futures[future]
                if task_type == 'detection':
                    detection_results_dict[frame_count] = future.result()
                elif task_type == 'pose':
                    pose_results_dict[frame_count] = future.result()

            trigger_result = get_trigger(
                drill_type, 
                detection_results_dict, 
                pose_results_dict, 
                frame_count
            )

        end_time = time.time()
        print(f"Request processing time: {end_time - start_time:.2f} seconds")

        frame_counter += 1

        return {
            "message": "Image processed successfully", 
            "count": frame_count, 
            "trigger_result": 0   
        }
    except Exception as e:
        logger.error(f"Error processing frame: {e}")
        return {
            "message": f"Error processing frame: {str(e)}",
            "count": frame_count,
        }

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            frame_data = FrameData(**json.loads(data))
            result = await process_frame(frame_data)
            await websocket.send_text(json.dumps(result))
    except WebSocketDisconnect:
        print("WebSocket disconnected")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # Attempt to close the connection if it's still open
        try:
            print("closing connection ")
            await websocket.close()
        except RuntimeError:
            pass

 
# @app.websocket("/ws")
# async def websocket_endpoint(websocket: WebSocket):
#     await websocket.accept()
#     try:
#         while True:
#             data = await websocket.receive_text()
#             frame_data = FrameData(**json.loads(data))
#             result = await process_frame(frame_data)
#             await websocket.send_text(json.dumps(result))
#     except Exception as e:
#         print(f"WebSocket connection closed: {e}")
#         await websocket.close()       