# Driver in which we can cobble a solution together

from ultralytics import YOLO
import cv2
import torch
import numpy as np
from ObjectDetector import ObjectDetector

# Load the model
model_path = 'models/Optimized_Resized_cl_1.onnx'
model = YOLO(model_path)

# Load the video
video_path = 'assets/Sample_Video2.mp4'
video = cv2.VideoCapture(video_path)

threshold = 0.75
frame_nmr = -1

# Grabbing these just in case
width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(video.get(cv2.CAP_PROP_FPS))


od = ObjectDetector(model=model)


while True:
    ret, frame = video.read()
    frame_nmr = frame_nmr + 1

    if not ret:
        break
    
    frame_bb_drawn = frame.copy()

    detections = od.getDetections(frame, 192) #<---Use imgsz=192 if using a "Resized" model; use 640 for "Fullsized" model
    
    print("Pytorch version: " + torch.__version__)
    print("Video is " + str(fps) + " frames per second")

    for detection in detections.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = map(int, detection)
        if detection[4] > threshold:
            cv2.rectangle(frame_bb_drawn, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    testframe = od.getCroppedDetections(frame, 192)[0] #<--testing helper class functions
    cv2.imshow("testframe", testframe)

    cv2.imshow('Frame', frame_bb_drawn)

    if cv2.waitKey(2) == ord('q'):
        break

video.release()


