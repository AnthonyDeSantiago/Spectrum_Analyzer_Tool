import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from os import path
# from sklearn.cluster import KMeans


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
    
    # Show me the imported video
    cv2.imshow("frame", frame)

    # Convert to Gray Scale
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Attempt to remove noise by blurring with Gaussian filter
    img = cv2.GaussianBlur(img, (5, 5), 0)

    # Process using Canny
    low_threshold = 20
    high_threshold = 40
    edges = cv2.Canny(img, low_threshold, high_threshold)
    
    # Run Hough on edge detected image
    rho = 1
    theta = np.pi / 180
    threshold = 200
    min_line_length = 600
    max_line_gap = 200
    line_image = np.copy(frame1) * 0

    lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]), min_line_length, max_line_gap)

    # Draw lines
    for line in lines:
         for x_1, y_1, x_2, y_2 in line:
             x_1 = x_1 + x1
             x_2 = x_2 + x1
             y_1 = y_1 + y1
             y_2 = y_2 + y1
             cv2.line(line_image, (x_1,y_1), (x_2,y_2), (0, 0, 255), 2)

    # Show me just just the lines drawn
    cv2.imshow("lines", line_image)
    
    # Show me processed image using Canny
    cv2.imshow("Canny Edges", edges)

    # Create image with lines drawn on original image
    line_edges = cv2.addWeighted(frame1, 1, line_image, 1, 0)

    # Show me lines drawn on the original video
    cv2.imshow("Line Edges", line_edges)


    if cv2.waitKey(2) == ord('q'):
        break

video.release()
