
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst

import warnings
from ultralytics import YOLO
from bytetracker_utils import BYTETracker
import logging
from pathlib import Path
import pyrealsense2 as rs
import numpy as np
import cv2
import struct
import socket
import pickle
import random
from v10utils import plot_one_box ,plot_target
import time
import threading
import queue  # Added for thread-safe queue
from liveinference_utlis import RtspServer ,get_host_ip

# Suppress warnings
warnings.filterwarnings('ignore',module='numpy')
logging.getLogger('ultralytics').setLevel(logging.ERROR)
print("LIB IMPORTED")



class Ultralytics_Inference():
    def __init__(self ,pt_file,
                 label_names,
                 track =True,
                 ):
        

        # Intialize the RTSP Server 
        self.host_ip = get_host_ip()
        self.rtsp_frames = RtspServer()
        self.rtsp_thread = threading.Thread(
            target=self.rtsp_frames.run_rtsp, 
            args=("8554", self.host_ip)
        )
        self.rtsp_thread.start()
        # Initialize the YOLO model
        self.device = 'cuda:0'
        self.model = YOLO(pt_file).to(device=self.device)
        print("MODEL LOADED-------", self.model.device)
        self.fp16 = True

        self.clicked_cord = None  # Stores user's click coordinates
        self.selected_track_id = None  # Stores selected object ID
        self.frames_without_target = 0  # Counter for frames where target is missing
        self.MAX_MISSING_FRAMES = 300  # Number of frames to wait before resetting tracking
        self.points_queue = queue.Queue()  # Thread-safe queue for click coordinates
        self.track_id_set = set()
        self.STX_DATA = {
                    "stx_id_vision": []
                    }
        

        # MCAST for send fov
        self.group = '224.1.1.1'
        self.port = 5005

        # 2-hop restriction in network
        self.ttl = 2

        self.sock = socket.socket(socket.AF_INET,
                            socket.SOCK_DGRAM,
                            socket.IPPROTO_UDP)
        self.sock.setsockopt(socket.IPPROTO_IP,
                        socket.IP_MULTICAST_TTL,
                        self.ttl)
        

        
        # Initialize GStreamer
        Gst.init(None)
                
        # Load class names
        self.names = [label.strip() for label in open(label_names)]
        
        # Running thread for recieving points
        point_thread = threading.Thread(target= self.receive_points_thread, daemon=True)
        point_thread.start()
        


    # Function to setup realsense
    def setup_realsense(self):
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 30)
        pipeline.start(config)
        print("PIPELINE SETTED UP")
        return pipeline
    

    def get_frames(self,pipeline):
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            return None
        color_image = np.asanyarray(color_frame.get_data())
        return color_image


    

        # Threaded function to receive points with debounce
    def receive_points_thread(self):
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
                self.points_queue.put(X_Y_points)
                last_point = X_Y_points
                print("New click received:", X_Y_points)


    def send_fov_points(self,cordinates):
        self.center_points = cordinates
        print(self.center_points)

        
        if len(self.center_points) != 0:
            data = pickle.dumps(self.center_points)
            self.sock.sendto(data, (self.group, self.port))

    def detect(self):
        self.frame_count = 0
        center = (960, 540)

        # setup  releasense 
        self.pipeline = self.setup_realsense()
        
        # Initialize ByteTracker
        tracker = BYTETracker() 

        colors = [[random.randint(0, 255) for _ in range(3)] for _ in self.names]
        start_time = time.time()

        while True:
            frame = self.get_frames(self.pipeline)
            if frame is None:
                print("No frames to read...")
                break

            # Check for new click from the queue
            try:
                self.clicked_cord = self.points_queue.get_nowait()
                print("New click from queue:", self.clicked_cord)
            except queue.Empty:
                pass  # No new click available

            self.frame_count += 1
            results = self.model(frame, device=self.device)
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



                if self.clicked_cord is not None:
                    click_x, click_y = self.clicked_cord
                    min_distance = float('inf')
                    for det in tracker_detection:
                        # Assume first 5 values are x1,y1,x2,y2,track_id
                        x1, y1, x2, y2, track_id = det[:5]
                        if x1 <= click_x <= x2 and y1 <= click_y <= y2:
                            center_x = (x1 + x2) / 2
                            center_y = (y1 + y2) / 2
                            distance = np.sqrt((click_x - center_x) ** 2 + (click_y - center_y) ** 2)
                            if distance < min_distance:
                                min_distance = distance
                                self.selected_track_id = int(track_id)
                    self.clicked_cord = None  # Reset after processing
                    
                # Check if selected object is present in current frame
                self.target_found = False


                current_id = {int(det[4]) for det in tracker_detection}
            

                # Draw bounding boxes
                for det in tracker_detection:
                    x1, y1, x2, y2, track_id,class_id, conf_score = det
                    class_id = int(class_id)
                    class_names = self.names[int(class_id)]
                    label = f"{int(track_id)} {conf_score:.2f} {class_names}"



                    detection_data = {
                    "detection_id": track_id,
                    "detection_type": class_names,
                    "time_stamp": time.strftime("%H:%M:%S", time.localtime())
                    }
                    
                    id = int(track_id)

                    if id not in self.track_id_set:
                        self.STX_DATA['stx_id_vision'].append(detection_data)
                        self.track_id_set.add(id)
                        print(self.STX_DATA, "\n\n")

                    
                    # If we have a selected track ID
                    if self.selected_track_id is not None:
                        if int(track_id) == self.selected_track_id:
                            self.target_found = True
                            cx = (x1 + x2) / 2
                            cy = (y1 + y2) / 2
                            self.send_fov_points((cx,cy))
                            plot_one_box([x1, y1, x2, y2], frame, label=label, color=(0,0,255), line_thickness=2)
                            # Draw a line from center to target
                            target_center_x = int((x1 + x2) / 2)
                            target_center_y = int((y1 + y2) / 2)
                            cv2.line(frame, center, (target_center_x, target_center_y), (0, 255, 255), 2)
                
                    else:
                        # No selected object, draw all detections
                        plot_one_box([x1, y1, x2, y2], frame, label=label, color=colors[class_id], line_thickness=2)

                previous_id = {data['detection_id'] for data in self.STX_DATA['stx_id_vision']}
                lost_id = previous_id - current_id
                if lost_id:
                    self.STX_DATA["stx_id_vision"] = [data for data in self.STX_DATA["stx_id_vision"] if data["detection_id"] in current_id]
                    self.track_id_set.intersection_update(current_id)


                # Update counter for missing target
                if self.selected_track_id is not None and not self.target_found:
                    self.frames_without_target += 1
                else:
                    self.frames_without_target = 0
                
                # If target is missing for too many frames, reset tracking
                if self.frames_without_target >= self.MAX_MISSING_FRAMES:
                    self.selected_track_id = None
                    self.frames_without_target = 0
            else:
                # No detections at all
                if self.selected_track_id is not None:
                    self.frames_without_target += 1
                    if self.frames_without_target >= self.MAX_MISSING_FRAMES:
                        # print(f"No detections for {MAX_MISSING_FRAMES} frames, resuming normal tracking...")
                        self.selected_track_id = None
                        self.frames_without_target = 0

            
            # Add FPS counter and status
            elapsed_time = time.time() - start_time
            fps = self.frame_count / elapsed_time
            
            # pushing frame to server
            self.rtsp_frames.add_frame(frame)


pt_file = "./yolo11n.pt"
labels_file = "./coco.names"

run_inference = Ultralytics_Inference(pt_file,labels_file)
run_inference.detect()























