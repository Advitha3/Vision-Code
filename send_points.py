import socket
import struct
import pickle
import math
from relative_commands import SiyiMount

MCAST_GRP = "224.1.1.1"
MCAST_PORT = 5005

# Camera and screen settings
width = 1920
height = 1080
circle_center_x = width // 2
circle_center_y = height // 2
circle_radius = 60


cam = SiyiMount()
# print("Moved to initial  position")
cam.move_relative(0.0, 0.0) 
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
sock.bind(("", MCAST_PORT))
mreq = struct.pack("4sl", socket.inet_aton(MCAST_GRP), socket.INADDR_ANY)
sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)

def points_for_gimbal():

    while True:
        data = sock.recv(10240)
        data = pickle.loads(data)  
        x_points, y_points = data  

        distance = math.sqrt((x_points - circle_center_x) ** 2 + (y_points - circle_center_y) ** 2)

        if distance > circle_radius:
            print("controlling gimble")
            pitch, yaw = cam.calculate_gimbal_angles(x_points, y_points, width, height)
            cam.move_relative(pitch, yaw)
            print(distance)
            print()
            # print(f"MOVING CAM to Pitch: Distance {pitch}, Yaw: {yaw} ,{distance}")
        
        if distance < circle_radius:
            print("Target locked - No gimbal movement required")

if __name__=="__main__":
    points_for_gimbal()

