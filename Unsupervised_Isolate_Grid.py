#UNDOCUMENTED CODE FROM MEETING TONIGHT -- WILL COMMENT OUT LATER 
#STILL REQUIRES MORE DATA CLEANING + CODE TWEAKING TO WORK WELL

import numpy as np
import cv2

#FOR SAMPLE VIDEO 2
cap = cv2.VideoCapture('/assets/Sample_Video2.mp4')

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
size = (frame_width, frame_height)

#Writers with 0 @ end write in grayscale (paramter value required to write grayscale to VideoWriter)
writer_gray = cv2.VideoWriter('/assets/Sample_Video2_gray.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, size, 0)
writer_blur = cv2.VideoWriter('/assets/Sample_Video2_blur.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, size, 0)
writer_thresh = cv2.VideoWriter('/assets/Sample_Video2_thresh.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, size, 0)
writer_mask = cv2.VideoWriter('/assets/Sample_Video2_mask.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, size, 0)
writer_out = cv2.VideoWriter('/assets/Sample_Video2_out.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, size, 0)
writer_blur1 = cv2.VideoWriter('/assets/Sample_Video2_blur1.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, size, 0)
writer_thresh1 = cv2.VideoWriter('/assets/Sample_Video2_thresh1.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, size, 0)
writer_final = cv2.VideoWriter('/assets/Sample_Video2_final.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, size)

"""
#FOR SAMPLE VIDEO
cap = cv2.VideoCapture('/assets/Sample_Video.mp4')

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
size = (frame_width, frame_height)

writer_gray = cv2.VideoWriter('/assets/Sample_Video_gray.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, size, 0)
writer_blur = cv2.VideoWriter('/assets/Sample_Video_blur.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, size, 0)
writer_thresh = cv2.VideoWriter('/assets/Sample_Video_thresh.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, size, 0)
writer_mask = cv2.VideoWriter('/assets/Sample_Video_mask.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, size, 0)
writer_out = cv2.VideoWriter('/assets/Sample_Video_out.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, size, 0)
writer_blur1 = cv2.VideoWriter('/assets/Sample_Video_blur1.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, size, 0)
writer_thresh1 = cv2.VideoWriter('/assets/Sample_Video_thresh1.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, size, 0)
writer_final = cv2.VideoWriter('/assets/Sample_Video_final.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, size)
"""

while True:
    ret, image = cap.read()
  
    if not ret:
        break

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gray = cv2.resize(gray, size)
    writer_gray.write(gray)

    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    blur = cv2.resize(blur, size)
    writer_blur.write(blur)

    ret2,thresh = cv2.threshold(blur, 50,350, cv2.THRESH_OTSU)
    thresh = cv2.resize(thresh, size)
    writer_thresh.write(thresh)

    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    max_area = 0
    c = 0
    for i in contours:
        area = cv2.contourArea(i)
        if area > 1000:
            if area > max_area:
                max_area = area
                best_cnt = i
                image = cv2.drawContours(image, contours, c, (0, 255, 0), 3)
        c += 1

    mask = np.zeros((gray.shape), np.uint8)

    cv2.drawContours(mask, [best_cnt], 0, 255, -1)
    cv2.drawContours(mask, [best_cnt], 0, 0, 2)
    writer_mask.write(mask)

    out = np.zeros_like(gray)
    out[mask == 255] = gray[mask == 255]
    out = cv2.resize(out, size)
    writer_out.write(out)

    blur = cv2.GaussianBlur(out, (5, 5), 0)
    writer_blur1.write(blur)
    thresh = cv2.adaptiveThreshold(blur, 255, 1, 1, 11, 2)
  
    writer_thresh1.write(thresh)
    contours, _= cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    c = 0
    for i in contours:
        area = cv2.contourArea(i)
        if area > 1000 / 2:
            cv2.drawContours(image, contours, c, (0, 255, 0), 3)
        c += 1

    writer_final.write(image)

cap.release()
writer_gray.release()
writer_blur.release()
writer_thresh.release()
writer_mask.release()
writer_out.release()
writer_blur1.release()
writer_thresh1.release()
writer_final.release()
cv2.destroyAllWindows()
