import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from os import path

video = cv2.VideoCapture(path.abspath(path.join(path.dirname(__file__),'assets/Sample_Video.mp4')))
real_capture = cv2.VideoCapture(path.abspath(path.join(path.dirname(__file__),'assets/Sample_Video.mp4')))

# Skip locating the screen for now
y1 = 235 #y1
y2 = 930 #y2
x1 = 700 #x1
x2 = 1582 #x2


while True:
    ret, frame = video.read()
    ret1, frame1 = real_capture.read()

    if not ret:
        break

    frame = frame[y1:y2, x1:x2] #<-- Skip locating the screen for now

    # Convert to Gray Scale
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Attempt to remove noise by blurring with Gaussian filter
    img = cv2.GaussianBlur(img, (5, 5), 0)

    # Process using Canny
    low_threshold = 20
    high_threshold = 40
    edges = cv2.Canny(img, low_threshold, high_threshold)
    cv2.imshow("Edges", edges)


    # Apply Laplace function
    # frame = cv2.Laplacian(frame, cv2.CV_16S, ksize=5)

    # # Convert back to uint8
    # #frame = cv2.convertScaleAbs(frame)

    # # Equalize Histogram
    # #frame = cv2.equalizeHist(frame)

    # # Let me see the result of all the filtering
    # #cv2.imshow("Filtered Frame", frame)


    # # Create a mask to filter out a range of values
    # # lower = 180
    # # upper = 300
    # # mask = cv2.inRange(frame, lower, upper)
    # # frame = cv2.bitwise_and(frame, frame, mask=mask)

    # # # Let me see the result after I apply the mask
    # # cv2.imshow("Masked Frame", frame)
    
    
    
    corners = cv2.goodFeaturesToTrack(edges, 121, 0.07, 50)
    corners = np.int0(corners)

    for corner in corners:
         x, y = corner.ravel()
         x = x + x1 #<-- Skip locating the screen for now
         y = y + y1 #<-- Skip locating the screen for now
         cv2.circle(frame1, (x, y), 10, (255, 255, 255), -1)


    cv2.imshow('frame', frame1)

    if cv2.waitKey(2) == ord('q'):
        break

video.release()

# Some utility functions
