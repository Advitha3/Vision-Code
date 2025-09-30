import gi
gi.require_version('Gst', '1.0')
gi.require_version('GstRtspServer', '1.0')
import cv2
from collections import deque
import time
import socket
import json
from  collections import deque
import random
from gi.repository import Gst, GstRtspServer, GLib
import torch
import warnings
warnings.filterwarnings('ignore')
print("GSTREAMER LIB IMPORTED")

fps = 30
count = 0
duration = 1 / fps * Gst.SECOND
frame_queue = deque(maxlen=1)

loop = GLib.MainLoop()
Gst.init(None)


class RtspServer(GstRtspServer.RTSPMediaFactory):
    def __init__(self):
        GstRtspServer.RTSPMediaFactory.__init__(self)
        self.number_frames = 0
        self.fps = fps
        self.duration = 1 / self.fps * Gst.SECOND

        #cpu pipeline
        self.launch_string = ('appsrc name=source is-live=true block=true format=GST_FORMAT_TIME ' \
                             'caps=video/x-raw,format=BGR,width=1920,height=1080,framerate={}/1 ' \
                             '! videoconvert ! video/x-raw,format=I420 ' \
                             '! x264enc bitrate=5000000 speed-preset=ultrafast tune=zerolatency ' \
                             '! rtph264pay config-interval=1 name=pay0 pt=96'.format(self.fps)
        )



        # gpu pipeline
        # self.launch_string = (
        #                         'appsrc name=source is-live=true block=true format=GST_FORMAT_TIME '
        #                         'caps=video/x-raw,format=BGR,width=1920,height=1080,framerate={}/1 '
        #                         '! videoconvert ! video/x-raw,format=NV12 '
        #                         '! nvvidconv ! video/x-raw(memory:NVMM),format=NV12 '
        #                         '! nvv4l2h264enc bitrate=4000000 '
        #                         '! rtph264pay config-interval=1 name=pay0 pt=96'.format(self.fps)
        #                     )


    def do_create_element(self, url):
        print ("Element created: " + self.launch_string)
        return Gst.parse_launch(self.launch_string)

    def do_configure(self, rtsp_media):
        self.number_frames = 0
        appsrc = rtsp_media.get_element().get_child_by_name('source')
        appsrc.connect('need-data', self.on_need_data)

    def on_need_data(self, src, length):
        global count
        if len(frame_queue) > 0:
            frame = frame_queue[0]
            data = frame.tobytes()
            buf = Gst.Buffer.new_allocate(None, len(data), None)
            buf.fill(0, data)
            buf.duration = self.duration
            timestamp = self.number_frames * self.duration
            buf.pts = buf.dts = int(timestamp)
            buf.offset = timestamp
            self.number_frames += 1
            retval = src.emit('push-buffer', buf)
            count +=1
            # print("pushed to buffer" ,count)
            if retval != Gst.FlowReturn.OK:
                print(f"Push buffer failed: {retval}")
            else:
                pass
                # print("DATA PUSHED SUCCESSFULLY")
        else:
            print("No frame in queue")

    def add_frame(self, frame):
        # height ,width,_= frame.shape
        # print(f"HEIGHT{height} WIDTH {width}")
        frame_queue.append(frame)


    

    def run_rtsp(self ,port, ip):
        rtspServer = GstRtspServer.RTSPServer()
        rtspServer.set_address("0.0.0.0")
        factory = RtspServer()
        factory.set_shared(True)  # Should handle multiple client streams
        mountPoints = rtspServer.get_mount_points()
        print("MOUNT POINTS", mountPoints)
        mountPoints.add_factory("/stream1", factory)
        rtspServer.set_service(port)
        rtspServer.set_address(ip)
        rtspServer.attach(None)
        print(f"RTSP Server setup on {ip}:{port}")
        loop.run()


def get_host_ip():
    """Retrieve the host IP address."""
    try:
        # Create a dummy socket to determine the primary IP
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))  # Use a public DNS server to determine the interface
        ip_address = s.getsockname()[0]
        s.close()
        return ip_address
    except Exception as e:
        print(f"Error determining host IP: {e}")
        return "127.0.0.1"  # Fallback to localhost  





