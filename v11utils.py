import cv2
import random


def plot_one_box(x, img, color=None, label=None, line_thickness=3):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)



def  plot_target(image):
    center = ((1920//2),(1080//2))
    radius = 60
    red = (0,0,255)
    thickness = 2 

    vertical_start = (960 ,450)
    vertical_end = (960 ,630)

    horizontal_start = (860 ,540)
    horizontal_end = (1060,540)

    cv2.line(image ,horizontal_start ,horizontal_end ,color=red ,thickness=thickness)
    cv2.line(image, vertical_start ,vertical_end ,color=red ,thickness=thickness)
    cv2.circle(image, center, radius, red, thickness)
    return image




def calculate_iou(x1, y1, x2, y2, fx1, fy1, fx2, fy2):
 
    x1_inter = max(x1, fx1)
    y1_inter = max(y1, fy1)
    x2_inter = min(x2, fx2)  
    y2_inter = min(y2, fy2) 

   
    width = max(0, x2_inter - x1_inter)  
    height = max(0, y2_inter - y1_inter)  

    # Compute intersection area
    area_inter = width * height

    # Compute areas of both bounding boxes
    area1_bbox = (x2 - x1) * (y2 - y1)
    area2_bbox = (fx2 - fx1) * (fy2 - fy1)

    area_union = area1_bbox + area2_bbox - area_inter


    iou = area_inter / area_union if area_union > 0 else 0
    return iou
