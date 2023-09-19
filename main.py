# Driver in which we can cobble a solution together

from ultralytics import YOLO
import cv2
import torch
import numpy as np
from ObjectDetector import ObjectDetector
from ImageProcessor import ImageProcessor
#from Util import util

# Load the models
model_path_screen_finder = 'models/Optimized_Resized_cl_1.onnx'
model_path_grid_finder = 'models/Grid_LowRes_1-4_224.onnx'
model_SpecScreen = YOLO(model_path_screen_finder, task='detect') #<---Model specifically for finding spectrum analyzer screen
model_Grid = YOLO(model_path_grid_finder, task='detect')


# Load the video
video_path = 'assets/Sample_Video2.mp4'
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
    #------------------------------------
    # Grab Important Settings from Screen
    #------------------------------------

    # reference_level = util.getReferenceLevel(isolated_spec_screen)
    # center_frequency = util.getCenter(isolated_spec_screen)
    # span = util.getSpan(isolated_spec_screen)


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
    if frame_nmr % 1 == 0: #<gratuitiously grabbing it every frame
        box = detector_grid.getBoundingBoxes(frame)[0]
        x1, y1, x2, y2, conf, class_id = map(int, box)
        grid_crop = [x1, y1, x2, y2]

    isolated_grid_crop = frame[grid_crop[1]: grid_crop[3], grid_crop[0]: grid_crop[2]]
    print(isolated_grid_crop.shape)

    cv2.imshow("Test grid crop", isolated_grid_crop)




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


