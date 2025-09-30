import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst

from ultralytics import YOLO
from bytetracker_utils import BYTETracker
import logging
from pathlib import Path
import numpy as np
import cv2
import pyrealsense2 as rs
import struct
import socket
import pickle
import random
import math
import warnings
from v10utils import plot_one_box, plot_target, calculate_iou
import time
import threading
import queue 
from collections import deque
from liveinference_utlis import RtspServer
import torch
import torch.nn.functional as F
from torchreid.utils import FeatureExtractor

# Suppress warnings
logging.getLogger('ultralytics').setLevel(logging.ERROR)
warnings.filterwarnings('ignore')
print("LIB IMPORTED")

class OSNetFeatureExtractor:
    def __init__(self, model_path='./osnet_ain_ms_d_c.pth.tar', device='cuda'):
        self.extractor = FeatureExtractor(
            model_name='osnet_ain_x1_0',
            model_path=model_path,
            device=device
        )
        self.initial_feature = None
        self.latest_feature = None
        self.alpha = 0.8  # Feature update rate
        self.similarity_threshold = 0.18  # Cosine similarity threshold
        
    def extract_features(self, crops):
        """Extract features from a list of cropped images"""
        features = self.extractor(crops)
        return F.normalize(features, p=2, dim=1)
    
    def initialize_target(self, crops):
        """Initialize target features from multiple crops of the same target"""
        features = self.extract_features(crops)
        mean_feature = torch.mean(features, dim=0)
        self.initial_feature = F.normalize(mean_feature.unsqueeze(0), p=2, dim=1).squeeze(0)
        self.latest_feature = self.initial_feature.clone()
        return True
    
    def update_features(self, crop):
        """Update the latest feature with new detection"""
        if self.latest_feature is None:
            return False
        features = self.extract_features([crop])
        self.latest_feature = (self.alpha * self.latest_feature) + ((1 - self.alpha) * features[0])
        self.latest_feature = F.normalize(self.latest_feature.unsqueeze(0), p=2, dim=1).squeeze(0)
        return True
    
    def compute_similarity(self, crops, track_ids):
        """Compute similarity with current crops and return matching track IDs"""
        if self.initial_feature is None or self.latest_feature is None:
            return [], []
            
        features = self.extract_features(crops)
        init_matches = []
        latest_matches = []
        
        for idx, feature in enumerate(features):
            cos_sim_init = F.cosine_similarity(self.initial_feature.unsqueeze(0), feature.unsqueeze(0), dim=1)
            cos_sim_latest = F.cosine_similarity(self.latest_feature.unsqueeze(0), feature.unsqueeze(0), dim=1)
            
            dist_init = float(1 - cos_sim_init.item())
            dist_latest = float(1 - cos_sim_latest.item())
            
            if dist_init < self.similarity_threshold:
                init_matches.append(track_ids[idx])
            if dist_latest < self.similarity_threshold:
                latest_matches.append(track_ids[idx])
                
        return init_matches, latest_matches

# Initialize GStreamer
Gst.init(None)

rtsp_frames = RtspServer()
rtsp_thread = threading.Thread(target=rtsp_frames.run_rtsp, args=("8554", "10.10.10.155"))
rtsp_thread.start()

# Paths for model and labels
PT_FILE = "./yolov11_tuned.pt"
LABELS_NAMES = "./label.names"
TRACK = True
RED = (0, 0, 255)
GREEN = (0,255,0)
YELLOW = (0 ,255,255)
BLUE = (255 ,0, 0)
# Load class names
names = [label.strip() for label in open(LABELS_NAMES)]

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

# Initialize OSNet
reid = OSNetFeatureExtractor()
feature_collection_frames = 0
max_feature_collection_frames = 60
target_crops = []
target_lost_count = 0
max_lost_count = 30

# Initialize tracking variables
alpha = 0.8
init_track = False
init_match_count = 0
Latest_match_count = 0
byteT_obj_lst = []
initial_obj_feature = None
obj_lost = None
pred_with_init_feature = []
pred_with_latest_feature = []
count = 0
c_init_feature = 0
st_time = time.perf_counter()
list_crped_imgs = []
view_img = np.ones((500, 800, 3), np.uint8) * 255
find_track_id = 0
lost_count = 0

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



# Intializing A8 mini 
frame_capture_pipeline = (
            'rtspsrc location=rtsp://192.168.144.25:8554/main.264 protocols=udp latency=0 ! '  # RTSP source
            'rtph264depay ! '                                                  # RTP depayloading
            'h264parse ! '                                                     # H.264 parsing
            'nvv4l2decoder ! '                                                 # NVIDIA hardware decoder
            'nvvidconv ! '                                                     # NVIDIA video converter
            'video/x-raw,format=RGBA,width=1920,height=1080 ! '                # Set desired resolution and RGBA format
            'appsink emit-signals=False sync=false max-buffers=1 drop=true name=sink'  # Appsink for frame grabbing
        )

pipeline = Gst.parse_launch(frame_capture_pipeline)
appsink = pipeline.get_by_name('sink')
appsink.set_property('emit_signals', False)
appsink.set_property('sync', False)

# Start the pipeline
pipeline.set_state(Gst.State.PLAYING)



def grab_frame(appsink):
    sample = appsink.emit('pull-sample')  # Pull the sample from appsink
    if sample:
        buf = sample.get_buffer()  
        caps = sample.get_caps() 
        width = caps.get_structure(0).get_value('width')
        height = caps.get_structure(0).get_value('height')
        data = buf.extract_dup(0, buf.get_size())

        # Convert raw data to a NumPy array (4 bytes per pixel for RGBA)
        rgba_frame = np.frombuffer(data, dtype=np.uint8).reshape((height, width, 4))
        rgb_frame = cv2.cvtColor(rgba_frame, cv2.COLOR_RGBA2BGR)

        return rgb_frame
    
    return None


def detect():
    global clicked_cord, selected_track_id, frames_without_target, feature_collection_frames, target_crops
    global init_track, init_match_count, Latest_match_count, byteT_obj_lst, initial_obj_feature, obj_lost
    global pred_with_init_feature, pred_with_latest_feature, count, c_init_feature, list_crped_imgs, find_track_id, lost_count

    frame_count = 0
    tracker = BYTETracker()
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
    start_time = time.time()
    st_time = time.perf_counter()
    prev_area = None
    scale_factor = 0
    send_center_points = True
    fixed_bbox = deque(maxlen=1)
    draw_box = True
    center = (960, 540)

    while True:
        count += 1
        frame = grab_frame(appsink)
        if frame is None:
            print("No frames to read...")
            break

        frame1 = frame.copy()
        
        # Check points queue for new target selection
        try:
            clicked_cord = points_queue.get_nowait()
            print("New click from queue:", clicked_cord)
        except queue.Empty:
            pass

        results = model.track(frame, conf=0.5, persist=True, verbose=False)

        if time.perf_counter() - st_time >= 1:
            print(f"FPS.......{count}")
            count = 0
            st_time = time.perf_counter()

        if results[0].boxes.is_track:
            for result in results:
                obj = result.boxes
                fram_byte_ids = obj.id.tolist()
                crpped_objs = [frame1[int(xyxy[1]):int(xyxy[3]), int(xyxy[0]):int(xyxy[2])] for xyxy in obj.xyxy]

                for cls, xyxy, obj_id in zip(obj.cls, obj.xyxy, obj.id):
                    obj_id = int(obj_id.tolist())
                    clas_ind = int(cls.item())
                    x_min, y_min = int(xyxy[0]), int(xyxy[1])
                    x_max, y_max = int(xyxy[2]), int(xyxy[3])
                    crp_obj = frame1[y_min:y_max, x_min:x_max]

                    # Object selection via click
                    if clicked_cord is not None:
                        click_x, click_y = clicked_cord
                        if x_min < click_x < x_max and y_min < click_y < y_max:
                            byteT_obj_lst = [obj_id]
                            print("byteTrack_object ID", byteT_obj_lst[0])
                            init_track = True
                            obj_lost = False
                            clicked_cord = None

                    # Handle target tracking and feature collection
                    if obj_id in byteT_obj_lst and crp_obj.size > 0 and len(byteT_obj_lst) > 0:
                        c_init_feature += 1
                        if c_init_feature < 60 and (c_init_feature % 10 == 0 or c_init_feature == 1):
                            crp_obj_resized = crp_obj
                            list_crped_imgs.append(crp_obj_resized)

                        if draw_box:
                            fixed_bbox.append((x_min, y_min, x_max, y_max))
                        draw_box = False

                        fx1, fy1, fx2, fy2 = fixed_bbox[0]
                        iou_score = calculate_iou(x_min, y_min, x_max, y_max, fx1, fy1, fx2, fy2)

                        if send_center_points:
                            cx = int((x_min + x_max) / 2)
                            cy = int((y_min + y_max) / 2)
                            send_fov_points((cx, cy))
                            print(f"send once ------ IOU Score : {iou_score} \n")
                        send_center_points = False

                        if iou_score < 0.60:
                            send_center_points = True
                            draw_box = True

                        # Display tracking
                        label = f"{obj_id} {1.0:.2f} {names[clas_ind]}"
                        plot_one_box([x_min, y_min, x_max, y_max], frame, label=label, color=RED, line_thickness=2)
                        target_center_x = int((x_min + x_max) / 2)
                        target_center_y = int((y_min + y_max) / 2)
                        cv2.line(frame, center, (target_center_x, target_center_y), YELLOW, 2)

                        if c_init_feature == 60:
                            features = reid.extractor(list_crped_imgs)
                            print("initial feature saved, length", len(list_crped_imgs))
                            for ind, img in enumerate(list_crped_imgs):
                                img = cv2.resize(img, (100, 250))
                                view_img[0:250, (100 * ind):100 * (ind + 1)] = img
                            cv2.imshow('View', view_img)
                            cv2.waitKey(1)

                            stack_feature = F.normalize(features, p=2, dim=1)
                            mean_feature = torch.mean(stack_feature, dim=0).cpu()
                            initial_obj_feature = F.normalize(mean_feature.unsqueeze(0), p=2, dim=1).squeeze(0)
                            lattest_obj_feature = initial_obj_feature.clone()
                            list_crped_imgs.clear()

                        elif c_init_feature >= 60 and c_init_feature % 10 == 0:
                            crp_obj_resized = crp_obj
                            img = cv2.resize(crp_obj_resized, (150, 250))
                            view_img[250:500, 200:350] = img
                            cv2.imshow('View', view_img)
                            cv2.waitKey(1)

                            features = reid.extractor([crp_obj_resized])
                            features = F.normalize(features, p=2, dim=1).cpu()
                            lattest_obj_feature = (alpha * lattest_obj_feature) + ((1 - alpha) * features[0])
                            lattest_obj_feature = F.normalize(lattest_obj_feature.unsqueeze(0), p=2, dim=1).squeeze(0)

                    elif initial_obj_feature is not None and byteT_obj_lst[0] not in fram_byte_ids:
                        lost_count += 1
                        if lost_count > 30:
                            obj_lost = True

                    # Display other objects
                    if obj_id not in byteT_obj_lst:
                        label = f"{obj_id} {1.0:.2f} {names[clas_ind]}"
                        plot_one_box([x_min, y_min, x_max, y_max], frame, label=label, color=colors[clas_ind], line_thickness=2)

            # Target recovery logic
            if obj_lost:
                plot_target(frame, "Target Lost", (1500, 55))
                find_track_id += 1
                if find_track_id % 10 == 0:
                    features = reid.extractor([obj for obj in crpped_objs]).cpu()
                    features = F.normalize(features, p=2, dim=1)
                    for ind_, feature in enumerate(features):
                        cos_sim_init = F.cosine_similarity(initial_obj_feature.unsqueeze(0), feature.unsqueeze(0), dim=1)
                        cos_sim_latest = F.cosine_similarity(lattest_obj_feature.unsqueeze(0), feature.unsqueeze(0), dim=1)
                        dist_init = float(1 - cos_sim_init.item())
                        dist_latest = float(1 - cos_sim_latest.item())
                        if dist_init < 0.18:
                            pred_with_init_feature.append(int(fram_byte_ids[ind_]))
                            init_match_count += 1
                        if dist_latest < 0.18:
                            pred_with_latest_feature.append(int(fram_byte_ids[ind_]))
                            Latest_match_count += 1

                    if init_match_count >= 4:
                        byteT_obj_lst[0] = max(set(pred_with_init_feature), key=pred_with_init_feature.count)
                        obj_lost = False
                        print("INITIAL MATCH, NEW TRACK ID", byteT_obj_lst[0])
                        init_match_count, Latest_match_count = 0, 0
                        find_track_id = 0
                        pred_with_init_feature, pred_with_latest_feature = [], []
                        send_center_points = True
                        draw_box = True
                    elif Latest_match_count >= 4:
                        byteT_obj_lst[0] = max(set(pred_with_latest_feature), key=pred_with_latest_feature.count)
                        obj_lost = False
                        print("LATEST MATCH, NEW TRACK ID", byteT_obj_lst[0])
                        init_match_count, Latest_match_count = 0, 0
                        find_track_id = 0
                        pred_with_init_feature, pred_with_latest_feature = [], []
                        send_center_points = True
                        draw_box = True

        elif init_track:
            plot_target(frame, "Target Lost", (1500, 55))

        # Add FPS counter
        elapsed_time = time.time() - start_time
        fps = frame_count / elapsed_time
        cv2.putText(frame, f"FPS: {fps:.1f}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # pushing frame to server
        rtsp_frames.add_frame(frame)
        frame_count += 1  # Add frame counter increment

if __name__ == "__main__":
    detect()
























