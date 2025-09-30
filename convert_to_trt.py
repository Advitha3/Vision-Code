from ultralytics import YOLO

# Load the YOLO11 model
model = YOLO("/home/nvidia/ultra_test/yolov11_tuned.pt")

# Export the model to TensorRT format
model.export(half=True ,format="engine")  # creates 'yolo11n.engine'
