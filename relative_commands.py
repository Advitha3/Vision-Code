import socket
import struct
import time

 
class SiyiMount:
    # Command constants
    GET_ATTITUDE = bytes.fromhex("55 66 01 00 00 00 00 0d e8 05")
    CENTER = bytes.fromhex("55 66 01 01 00 00 00 08 01 d1 12")
 
    def __init__(self, ip="192.168.144.25", port=37260):
        self.ip = ip
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.settimeout(5)
        self.current_yaw = 0
        self.current_pitch = 0
 
    def __del__(self):
        if self.sock:
            self.sock.close()
 
    def send_command(self, command):
        try:
            self.sock.sendto(command, (self.ip, self.port))
            return self.sock.recvfrom(1024)[0]
        except socket.timeout:
            return None
 
    def get_attitude(self):
        response = self.send_command(self.GET_ATTITUDE)
        if response and len(response) >= 14:
            #print("INSIDE RESPONSE")
            yaw, pitch, roll = struct.unpack('<hhh', response[8:14])
            #print("self_yaw",yaw , pitch)
            self.current_yaw = -yaw * 0.1
            self.current_pitch = pitch * 0.1
            return self.current_yaw, self.current_pitch, roll * 0.1
        print("NO RESPONSE")
        return None
    def get_attitudee(self):
        response = self.send_command(self.GET_ATTITUDE)
        if response and len(response) >= 22:
            print("Full response (hex):", response.hex(' '))
            
            # Try different positions for the attitude data
            # The payload seems to start after byte 7
            # The data might be packed in a different order or format
            
            # Let's try these positions (looking at the entire payload)
            values = []
            for i in range(8, 22, 2):
                if i+2 <= len(response):
                    val = struct.unpack('<h', response[i:i+2])[0]
                    values.append(val)
            
            print("All possible values at 2-byte intervals:", values)
            
            # Based on your expected output, we need to find which values
            # when multiplied by an appropriate factor give us 45 degrees
            
            # Let's use the traditional positions but with adjusted scaling
            yaw_raw = struct.unpack('<h', response[8:10])[0]
            pitch_raw = struct.unpack('<h', response[10:12])[0]
            
            # Try different scaling factors or transformations
            # If 45 degrees is represented as 450 in raw form (×10)
            self.current_yaw = -yaw_raw * 0.1
            self.current_pitch = pitch_raw* 0.1
            
            # Additional diagnostic information
            print(f"Raw values - yaw: {yaw_raw} pitch: {pitch_raw}")
            print(f"Scaled values - yaw: {self.current_yaw:.1f}° pitch: {self.current_pitch:.1f}°")
            
            # If these still don't match expected values, we'll need to
            # look elsewhere in the packet or try different scaling
            
            return self.current_yaw, self.current_pitch, 0  # Placeholder for roll
        
        print("NO RESPONSE")
        return None
    # def get_attitude(self):
    #     response = self.send_command(self.GET_ATTITUDE)
    #     if response and len(response) >= 14:
    #         print("INSIDE RESPONSE")
    #         yaw, pitch, roll = struct.unpack('<hhh', response[8:14])
    #         print("self_yaw",yaw , pitch)
    #         self.current_yaw = -yaw * 0.1
    #         self.current_pitch = pitch * 0.1
    #         return yaw, pitch, roll
    #     print("NO RESPONSE")
    #     return None
    
    # def update_info():
    #     attitude = self.get_attitude() 
    #     print("hi")
    #     if attitude:
    #         yaw, pitch, roll = attitude
    #         print(f"Attitude: Yaw={yaw}, Pitch={pitch}, Roll={roll}")
    #     else:
    #         print("Error: Could not retrieve gimbal attitude.")
    #     return yaw, pitch, roll
 
    def center(self):
        response = self.send_command(self.CENTER)
        if response:
            self.current_yaw = 0
            self.current_pitch = 0
            return True
        return False
 
    def move_relative(self, rel_pitch_deg, rel_yaw_deg):
        self.get_attitudee()
        new_yaw = self.current_yaw + rel_yaw_deg
        new_pitch = self.current_pitch + rel_pitch_deg
        
        new_yaw = max(min(new_yaw, 180), -180)
        new_pitch = max(min(new_pitch, 90), -90)
        
        yaw_cdeg = int(-new_yaw * 10)
        pitch_cdeg = int(new_pitch * 10)
        
        command = struct.pack('<BBBHHBHH', 0x55, 0x66, 0x01, 0x0004, 0x0000, 0x0E,
                              yaw_cdeg & 0xFFFF, pitch_cdeg & 0xFFFF)
        command += struct.pack('<H', self.crc16_ccitt(command))
        
        response = self.send_command(command)
        print("sending command to change position",response)
        if response:
            self.current_yaw = new_yaw
            self.current_pitch = new_pitch
            return True
        return False
    
    def calculate_gimbal_angles(self, x, y, frame_width, frame_height):
        # Unpack values
        frame_x = frame_width / 2
        frame_y = frame_height / 2
        obj_x, obj_y = x, y
        width, height = (frame_width, frame_height)
        hfov, vfov = (81, 65.60)  # Check the camera specs # Hari changing 65.60 to 45.69
 
        # Calculate offsets
        offset_x = obj_x - frame_x
        offset_y =(frame_y - obj_y ) # Invert y-axis
 
        # Calculate yaw and pitch angles
        yaw = (offset_x / (width / 2)) * (hfov / 2)
        pitch = (offset_y / (height / 2)) * (vfov / 2)
 
        #print(yaw, pitch)
        return pitch, yaw
 
    @staticmethod
    def crc16_ccitt(data):
        crc = 0
        for byte in data:
            crc ^= byte << 8
            for _ in range(8):
                crc = (crc << 1) ^ 0x1021 if crc & 0x8000 else crc << 1
        return crc & 0xFFFF
    