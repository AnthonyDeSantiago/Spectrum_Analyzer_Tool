# Driver in which we can cobble a solution together

from ultralytics import YOLO
import cv2
import torch
import numpy as np
from .ObjectDetector import ObjectDetector
from .ImageProcessor import ImageProcessor
from os import path
#from Util import util

# Load the models
model_path_screen_finder = path.abspath(path.join(path.dirname(__file__),'models/Optimized_Resized_cl_1.onnx'))
model_path_grid_finder = path.abspath(path.join(path.dirname(__file__),'models/Grid_LowRes_1-4_224.onnx'))
model_SpecScreen = YOLO(model_path_screen_finder, task='detect') #<---Model specifically for finding spectrum analyzer screen
model_Grid = YOLO(model_path_grid_finder, task='detect')

# Create an ImageProcessor Object
processor = ImageProcessor()


# Load the video
video_path = path.abspath(path.join(path.dirname(__file__),'assets/Sample_Video2.mp4'))
video = cv2.VideoCapture(video_path)

# Specify thresholds
threshold = 0.75
width = 470
hieght = 588

# Grabbing these just in case
width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(video.get(cv2.CAP_PROP_FPS))


# Instantiate object detectors here
detector_SpecScreen = ObjectDetector(model=model_SpecScreen, imgz=192)
detector_grid = ObjectDetector(model=model_Grid, imgz=224)

# A frame count so we can perform actions every ith, jth, kth, etc frame
frame_nmr = -1
detect_screen_freq = fps #<--Detect screen once every second
detect_gridCrop_freq = fps
sequence_length = 8

bb_screen = []
grid_crop = []

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
    
    # Testing image processor
    
    

    #------------------------------------
    # Isolate Grid
    #------------------------------------
    if frame_nmr % 100 == 0: #<gratuitiously grabbing it every frame
        box = detector_grid.getBoundingBoxes(frame)[0]
        x1, y1, x2, y2, conf, class_id = map(int, box)
        grid_crop = [x1, y1, x2, y2]
        processor.frame_deque.clear()

    isolated_grid_crop = frame[grid_crop[1]: grid_crop[3], grid_crop[0]: grid_crop[2]]


    cv2.imshow("Test grid crop", isolated_grid_crop)
    isolated_grid_crop = cv2.cvtColor(isolated_grid_crop, cv2.COLOR_BGR2GRAY)
    shape = isolated_grid_crop.shape

    processor.append(isolated_grid_crop)
    background = processor.getBackground()

    resultants = processor.performDifferencing(background)
    sums = processor.performSumming(resultants)

    contours, _ = cv2.findContours(sums, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    

    for contour in contours:
        if cv2.contourArea(contour) < 250:
            (x, y, w, h) = cv2.boundingRect(contour)
            x = x + grid_crop[0]
            print(grid_crop)
            y = y + grid_crop[1]
        
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)



    


    cv2.imshow("Test image processor", processor.getBackground())

    cv2.imshow("final frame", frame)

    if cv2.waitKey(2) == ord('q'):
        break

video.release()


