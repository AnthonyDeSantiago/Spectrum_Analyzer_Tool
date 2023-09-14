# Driver in which we can cobble a solution together

from ultralytics import YOLO
import cv2
import torch
import numpy as np
from ObjectDetector import ObjectDetector
from ImageProcessor import ImageProcessor

# Load the models
model_path = 'models/Optimized_Resized_cl_1.onnx'
model_SpecScreen = YOLO(model_path, task='detect') #<---Model specifically for finding spectrum analyzer screen

# Load the video
video_path = 'assets/Sample_Video2.mp4'
video = cv2.VideoCapture(video_path)

# Specify thresholds
threshold = 0.75


# Grabbing these just in case
width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(video.get(cv2.CAP_PROP_FPS))

# Instantiate object detectors here
detector_SpecScreen = ObjectDetector(model=model_SpecScreen, imgz=192)

# A frame count so we can perform actions every ith, jth, kth, etc frame
frame_nmr = -1
detect_screen_freq = fps #<--Detect screen once every second

bb_screen = []

while True:
    ret, frame = video.read()
    frame_nmr = frame_nmr + 1

    if not ret:
        break
    
    #---------------------------------
    # Isolate Spectrum Analyzer Screen
    #---------------------------------
    if frame_nmr % detect_screen_freq == 0: #<------Only locate screen once every 30 frames
        box = detector_SpecScreen.getBoundingBoxes(frame)[0]
        x1, y1, x2, y2, conf, class_id = map(int, box)
        bb_screen = [x1, y1, x2, y2]

    isolated_spec_screen = frame[bb_screen[1]: bb_screen[3], bb_screen[0]: bb_screen[2]]

    cv2.imshow("Test isolated_spec_screen", isolated_spec_screen)
    #------------------------------------
    # Grab Important Settings from Screen
    #------------------------------------

    # Get Reference Level
    # Get Center Frequency
    # Get Span
    # Get Decibal increments? lol dont know what its called



    #------------------------------------
    # Isolate Measured signal
    #------------------------------------

    #isolated_signal = ImageProcessor.isolateSignal(isolated_spec_screen)
    

    #------------------------------------
    # Isolate Grid
    #------------------------------------

    #isolated_grid = ImageProcessor.isolateGrid(isolated_spec_screen)


    #------------------------------------
    # Grab Data
    #------------------------------------

    # get timestamp
    # get center frequency
    # get minimum amplitude
    # get maximum amplitude
    # get average amplitude


    #------------------------------------
    # Output Data to a CSV file
    #------------------------------------

    if cv2.waitKey(2) == ord('q'):
        break

video.release()


