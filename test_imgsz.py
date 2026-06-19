import cv2
from ultralytics import YOLO

model = YOLO("yolo26n.pt") # Actually, maybe it's yolo11n.pt or whatever local model is present
print(model.model.names)
