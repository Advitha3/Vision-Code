import sys
sys.path.append("../../../yolov7/")
import pyrealsense2 as rs
import numpy as np
import cv2
import torch
from pathlib import Path
from ultralytics import YOLO
from torchreid.utils import FeatureExtractor
import time
import os

import torch.nn.functional as F
import torch.backends.cudnn as cudnn
# from utils.datasets import letterbox

# Initialize YOLO model
model = YOLO("./yolo11n_latest.pt")
names = model.names
print("names", names)

# Initialize OSNet Re-ID
extractor = FeatureExtractor(
    model_name='osnet_ain_x1_0',
    model_path='./osnet_ain_ms_d_c.pth.tar',
    device='cuda'
)

# Configure RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

# Global variables for mouse callback
click_x, click_y = None, None
select_object = False

def on_mouse(event, x, y, flags, param):
    global click_x, click_y, select_object
    if event == cv2.EVENT_LBUTTONDOWN:
        click_x, click_y = x, y
        select_object = True

# Set up window and callback
cv2.namedWindow('Frame')
cv2.setMouseCallback('Frame', on_mouse)

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

try:
    while True:
        count += 1
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        frame = np.asanyarray(color_frame.get_data())  # Keep at 640x480
        frame1 = frame.copy()
        # results = model.track(frame, conf=0.5, persist=True, verbose=False, tracker="./custom_bytetrack.yaml")
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
                    if select_object and click_x is not None and click_y is not None:
                        if x_min < click_x < x_max and y_min < click_y < y_max:
                            byteT_obj_lst = [obj_id]
                            print("byteTrack_object ID", byteT_obj_lst[0])
                            init_track = True
                            obj_lost = False
                            select_object = False
                            click_x, click_y = None, None

                    if obj_id in byteT_obj_lst and crp_obj.size > 0 and len(byteT_obj_lst) > 0:
                        c_init_feature += 1
                        if c_init_feature < 60 and (c_init_feature % 10 == 0 or c_init_feature == 1):
                            # crp_obj_resized = cv2.resize(crp_obj, (128, 256))
                            crp_obj_resized = crp_obj
                            list_crped_imgs.append(crp_obj_resized)

                        frame = cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 0, 255), 10)
                        frame = cv2.putText(frame, 'Target', (x_min + 25, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 191, 255), 4)

                        if c_init_feature == 60:
                            features = extractor(list_crped_imgs)
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
                            list_crped_imgs.clear()  # Clear to save memory

                        elif c_init_feature >= 60 and c_init_feature % 10 == 0:
                            # crp_obj_resized = cv2.resize(crp_obj, (128, 256))
                            crp_obj_resized = crp_obj
                            img = cv2.resize(crp_obj_resized, (150, 250))
                            view_img[250:500, 200:350] = img
                            cv2.imshow('View', view_img)
                            cv2.waitKey(1)

                            features = extractor([crp_obj_resized])
                            features = F.normalize(features, p=2, dim=1).cpu()
                            lattest_obj_feature = (alpha * lattest_obj_feature) + ((1 - alpha) * features[0])
                            lattest_obj_feature = F.normalize(lattest_obj_feature.unsqueeze(0), p=2, dim=1).squeeze(0)

                    elif initial_obj_feature is not None and byteT_obj_lst[0] not in fram_byte_ids:
                        lost_count += 1
                        if lost_count > 30:
                            obj_lost = True

                    frame = cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 255, 255), 2)

            if obj_lost:
                frame = cv2.putText(frame, "Target Lost", (1500, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.9, (0, 0, 255), 5)
                find_track_id += 1
                if find_track_id % 10 == 0:  # Reduced frequency
                    # features = extractor([cv2.resize(obj, (128, 256)) for obj in crpped_objs]).cpu()
                    features = extractor([obj for obj in crpped_objs]).cpu()
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
                    elif Latest_match_count >= 4:
                        byteT_obj_lst[0] = max(set(pred_with_latest_feature), key=pred_with_latest_feature.count)
                        obj_lost = False
                        print("LATEST MATCH, NEW TRACK ID", byteT_obj_lst[0])
                        init_match_count, Latest_match_count = 0, 0
                        find_track_id = 0
                        pred_with_init_feature, pred_with_latest_feature = [], []

        elif init_track:
            frame = cv2.putText(frame, "Target Lost", (1500, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.9, (0, 0, 255), 5)

        # letterbox frame maintaining aspect ratio and padding 640 x 480
        # frame = letterbox(frame, new_shape=(640, 480))[0]
        cv2.imshow('Frame', frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()