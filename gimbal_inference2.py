
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst

from ultralytics import YOLO
from bytetracker_utils import BYTETracker
import logging
from pathlib import Path
import numpy as np
import cv2
import torch 
from torchreid.reid.utils import  FeatureExtractor
import pyrealsense2 as rs
import struct
import socket
import pickle
import random
import math
import warnings
from v11utils import plot_one_box ,plot_target ,calculate_iou
import time
import threading
import queue  # Added for thread-safe queue
from collections import deque
from liveinference_utlis import RtspServer

# Suppress warnings
logging.getLogger('ultralytics').setLevel(logging.ERROR)
warnings.filterwarnings('ignore')
print("LIB IMPORTED")

# Initialize GStreamer
Gst.init(None)

rtsp_frames = RtspServer()
rtsp_thread = threading.Thread(target=rtsp_frames.run_rtsp, args=("8554", "10.10.10.147"))
rtsp_thread.start()



# Intilaize Feature extractor 
class ReIdExtractor:
    def  __init__(self, device = "cuda"):
        self.extractor = FeatureExtractor(
            model_name='osnet_ain_x1_0',
            model_path= "/home/bhavit/Desktop/Ghost_ultralytics_v2/osnet_ain_ms_d_c.pth.tar",
            device = device
        )

    @torch.no_grad()
    def extract_feature(self ,image_patch):
        if image_patch is None or image_patch.size == 0 :
            return f"EMPTY------{[]}"
        features = self.extractor(image_patch)
        return features.cpu().numpy()


feature_extraction = ReIdExtractor()

# Paths for model and labels
PT_FILE = "/home/bhavit/Desktop/Ghost_ultralytics_v2/best_fixed_np.pt"
LABELS_NAMES = "/home/bhavit/Desktop/Ghost_ultralytics_v2/labels.names"
TRACK = True
RED = (0, 0, 255) 
GREEN = (0,255,0)
YELLOW = (0 ,255,255)
BLUE = (255 ,0, 0)

# Load class names
names = [label.strip() for label in open(LABELS_NAMES)]
print(names)

# Initialize the YOLO model
device = 'cuda:0'
model = YOLO(PT_FILE).to(device=device)

print("MODEL LOADED-------", model.device)
fp16 = True

# Global variables
clicked_cord = None  # Stores user's click coordinates
selected_track_id = None  # Stores selected object ID
frames_without_target = 0  # Counter for frames where target is missing
MAX_MISSING_FRAMES = 300  # Number of frames to wait before resetting tracking
points_queue = queue.Queue()  # Thread-safe queue for click coordinates

# Threaded function to receive points with debounce
def receive_points_thread():
    MCAST_GRP = '224.1.1.1'
    MCAST_PORT = 5004
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(('', MCAST_PORT))
    mreq = struct.pack("4sl", socket.inet_aton(MCAST_GRP), socket.INADDR_ANY)
    sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
    
    last_point = None  # Track the last received point for debouncing
    
    while True:
        data = sock.recv(10240)
        X_Y_points = pickle.loads(data)
        
        # Only queue the point if it's different from the last one
        if X_Y_points != last_point:
            points_queue.put(X_Y_points)
            last_point = X_Y_points
            print("New click received:", X_Y_points)

# Start the point receiver thread
point_thread = threading.Thread(target=receive_points_thread, daemon=True)
point_thread.start()


# MCAST for send fov
group = '224.1.1.1'
port = 5005

# 2-hop restriction in network
ttl = 2

sock = socket.socket(socket.AF_INET,
                    socket.SOCK_DGRAM,
                    socket.IPPROTO_UDP)
sock.setsockopt(socket.IPPROTO_IP,
                socket.IP_MULTICAST_TTL,ttl)


# sending fov points to  the visualize
def send_fov_points(cordinates):
    center_points = cordinates

    
    if len(center_points) != 0:
        data = pickle.dumps(center_points)
        sock.sendto(data, (group, port))



def setup_realsense():
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 30)
    pipeline.start(config)
    print("PIPELINE SETTED UP")
    return pipeline

def get_frames(pipeline):
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    if not color_frame:
        return None
    color_image = np.asanyarray(color_frame.get_data())
    return color_image



def detect():
    global clicked_cord, selected_track_id, frames_without_target 


    frame_count = 0
    tracker = BYTETracker()  # Optionally configure with parameters
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
    pipeline = setup_realsense()
    start_time = time.time()
    send_center_points = True
    fixed_bbox = deque(maxlen=1)
    draw_box = True
    store_feature = None
    target_id = None

    while True:
        frame = get_frames(pipeline)
        # ret, frame = cap.read()
        if frame is None:
            print("No frames to read...")
            break

        # Check for new click from the queue
        try:
            clicked_cord = points_queue.get_nowait()
            print("New click store_featurefrom queue:", clicked_cord)
        except queue.Empty:
            pass  # No new click available

        frame_count += 1
        results = model(frame, device=device)
        DET = results[0]

        if len(DET) != 0:
            track_dets = []
            for i in range(len(results[0].boxes)):
                box = results[0].boxes[i]
                clsID = int(box.cls.cpu().numpy()[0])
                conf = box.conf.cpu().numpy()[0]
                conf = float(f'{conf:.2f}')
                bb = box.xyxy.cpu().numpy()  # Shape (1,4)
                x1, y1, x2, y2 = bb[0]  # Unpack the coordinates
                track_dets.append([x1, y1, x2, y2, conf, clsID])

            tracker_detection = np.array(track_dets)
            tracker_detection = tracker.update(tracker_detection)  # Update tracker
            # print(tracker_detection)


            if clicked_cord is not None:
                click_x, click_y = clicked_cord
                min_distance = float('inf')
                for det in tracker_detection:
                    x1, y1, x2, y2, track_id = det[:5]
                    if x1 <= click_x <= x2 and y1 <= click_y <= y2:
                        center_x = (x1 + x2) / 2
                        center_y = (y1 + y2) / 2
                        distance = np.sqrt((click_x - center_x) ** 2 + (click_y - center_y) ** 2)
                        if distance < min_distance:
                            min_distance = distance
                            store_feature = frame[int(y1):int(y2), int(x1):int(x2)]
                            selected_track_id = int(track_id)
                clicked_cord = None  # Reset after processing
                
            # Check if selected object is present in current frame
            target_found = False  
            
            # Draw bounding boxes
            for det in tracker_detection:
                # print(det)
                x1, y1, x2, y2, track_id,class_id, conf_score = det
                class_id = int(class_id)
                class_names = names[int(class_id)]
                label = f"{int(track_id)} {conf_score:.2f} {class_names}"
                # print("DETECTIONS", det)

                # print(f"Selected Track_id {selected_track_id}")
                
                # If we have a selected track ID
                if selected_track_id is not None:
                    # If this detection matches our selected track ID
                    if int(track_id) == selected_track_id:
                        target_found = True
                        cx = int((x1 + x2) / 2)
                        cy = int((y1 + y2) / 2)

                        if draw_box:
                            fixed_bbox.append((x1,y1,x2,y2))
                        draw_box = False


                        # print("PLotting target ----------------------->" ,selected_track_id)
                        plot_one_box([x1, y1, x2, y2], frame, label=label, color=RED, line_thickness=2)

                        # unpacking fixed box 
                        fx1 ,fy1 ,fx2 ,fy2 = fixed_bbox[0]
                        iou_score = calculate_iou(x1,y1,x2,y2,fx1,fy1,fx2,fy2)
                        # print(f" IOU  Score : {iou_score}")

                        if send_center_points:
                            send_fov_points((cx,cy))
                            # print("Send once")
                        
                        send_center_points = False

                        if iou_score < 0.7:
                            send_center_points = True
                            draw_box = True 
                        
                else:
                    plot_one_box([x1, y1, x2, y2], frame, label=label, color=colors[class_id], line_thickness=2)
                        

                if frames_without_target > 10:
                        print("Finding match --------------")
                        prev_target = cv2.resize(store_feature,(224,244))
                        print(prev_target.shape)
                        current_target = frame[int(y1):int(y2),int(x1):int(x2)]
                        print(current_target.shape)

                        if current_target.size !=0:
                            current_target = cv2.resize(current_target, (224, 224))
                            prev_target_feat = feature_extraction.extract_feature(prev_target).flatten()
                            current_target_feat = feature_extraction.extract_feature(current_target).flatten()

                            similarity = np.dot(prev_target_feat, current_target_feat) / (
                                np.linalg.norm(prev_target_feat) * np.linalg.norm(current_target_feat)
                            )
                            
                            print(f"REVIVING BYTETRACK ---------{x1,y1,x2,y2}---{int(track_id)}---{similarity}" )

                            if similarity > 0.8:
                                selected_track_id = int(track_id)

                            else:
                                frames_without_target += 1

                

            # Update counter for missing target
            if selected_track_id is not None and not target_found:
                frames_without_target += 1
                print("Tartget Lost-----",frames_without_target)
            else:
                frames_without_target = 0
            
            # If target is missing for too many frames, reset tracking
            if frames_without_target >= MAX_MISSING_FRAMES:
                selected_track_id = None
                frames_without_target = 0

        else:
            # No detections at all
            if selected_track_id is not None:
                frames_without_target += 1
                if frames_without_target >= MAX_MISSING_FRAMES:
                    selected_track_id = None
                    frames_without_target = 0

        # Add status text
        status_text = "Tracking All Objects"
        if selected_track_id is not None:
            status_text = f"Tracking Object ID: {selected_track_id}"
        
        # Add FPS counter and status
        elapsed_time = time.time() - start_time
        fps = frame_count / elapsed_time
        rtsp_frames.add_frame(frame)

detect()
























