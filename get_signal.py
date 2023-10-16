import numpy as np
import cv2
import csv

# CAN UPDATE THIS IN THE FUTURE TO CHOOSE WHERE ON USERS MACHINE THE OUTPUT GOES
assetDirAdd = 'assets/' 

######################################################################
# ANALYSIS MODULE TO GET THE SIGNAL USING CV2
#####################################################################

class GetSignalWithCV2:
    def __init__(self, frames, consecutive_frames=4, median_background_image=''):
        self.frameset = frames
        self.consecutive_frames = int(consecutive_frames)
        self.median_bg = median_background_image
        self.boxes = []
    
    ### ISOLATE THE SIGNAL FROM THE SPECTRUM ANALYZER
    def get_signal(self):
        out = []

        # Get static background, convert to grayscale
        background_raw = self.median_bg
        background = background_raw #cv2.cvtColor(background_raw, cv2.COLOR_BGR2GRAY)

        frame_count = 0

        while frame_count < len(self.frameset):
            if self.frameset[frame_count:frame_count + self.consecutive_frames] != [ ]:
                frame = self.frameset[frame_count]
                orig_frame = frame
                gray = orig_frame #cv2.cvtColor(orig_frame, cv2.COLOR_BGR2GRAY)

                if frame_count % self.consecutive_frames == 0 or frame_count == 1:
                    frame_diff_list = []
                
                frame_diff = cv2.absdiff(gray, background) # <-- find the difference between current and base
                ret,thres = cv2.threshold(frame_diff, 50, 255, cv2.THRESH_BINARY) # <-- thresholding to convert frame to binary
                dilate_frame = cv2.dilate(thres, None, iterations=2) # <-- dilate frame to get more white area to help w/ contouring
                #kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)) 
                #open_frame = cv2.morphologyEx(thres, cv2.MORPH_OPEN, kernel)
                frame_diff_list.append(dilate_frame) # <-- append the final result into the `frame_diff_list`

                # if reached `consecutive_frame` number of frames
                if len(frame_diff_list) == self.consecutive_frames:
                    sum_frames = sum(frame_diff_list) # add all frames in diff list
                    
                    #TBD: EXPERIMENT WITH CHAIN TYPE FURTHER
                    contours, hierarchy = cv2.findContours(sum_frames, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
 
                    # For Testing - draw contours
                    sum_frames = cv2.cvtColor(sum_frames, cv2.COLOR_GRAY2BGR)
                    
                    for i, cnt in enumerate(contours):
                        cv2.drawContours(sum_frames, contours, i, (0, 0, 255), 3)                      

                    ##################################################################
                    # Contour Processing
                    ##################################################################
                    contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)
                    contours = contours[:int(len(contours)-(len(contours)*.01))] #<-- throwing out approximately 1% of smallest contours
                    #give sorted contours to agglomerative clustering, with a threshold distance of 2% of the width of the image
                    contours = agglomerative_cluster(contours, (orig_frame.shape[1])*0.02) 

                    for contour in contours:
                        # noise detection: only continue if contour area is less than 500...
                        #if cv2.contourArea(contour) > 1: continue
                        # get the xmin, ymin, width, and height coordinates from the contours
                        (x, y, w, h) = cv2.boundingRect(contour)
                        # add boundingRect to lists of coords, along with the second of the video the rect is associated with
                        self.boxes.append([(frame_count + 1)/self.consecutive_frames, x, y, x + w, y + h, w, h])
                        # draw the bounding boxes
                        cv2.rectangle(sum_frames, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        
                        cv2.imshow("sum frame", sum_frames)
                        cv2.imshow("orig frame", orig_frame)
                        cv2.waitKey(100)
                        
                        #for frame in frame_diff_list:
                        #    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                        #    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        #    out.append(frame)
                    
                    if cv2.waitKey(0) & 0xFF == ord('q'):
                        break

            else:
                break

            frame_count += 1

        # WRITE BOUNDING BOX INFORMATION TO CSV
        print("Printing contours to CSV")

        header = ['frame number', 'x1', 'y1', 'x2', 'y2', 'width', 'height']

        with open(assetDirAdd + 'out.csv', 'w', encoding='UTF8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)

            for box in self.boxes:
                writer.writerow(box)
        
        print("\nget_signal run was successful\n")


def agglomerative_cluster(contours, threshold_distance=30.0):
    current_contours = contours
    while len(current_contours) > 1:
        min_distance = None
        min_coordinate = None

        for x in range(len(current_contours)-1):
            for y in range(x+1, len(current_contours)):
                x1, y1, w1, h1 = cv2.boundingRect(current_contours[x])
                c_x1 = x1 + w1/2
                c_y1 = y1 + h1/2

                x2, y2, w2, h2 = cv2.boundingRect(current_contours[y])
                c_x2 = x2 + w2/2
                c_y2 = y2 + h2/2

                distance = max(abs(c_x1 - c_x2) - (w1 + w2)/2, abs(c_y1 - c_y2) - (h1 + h2)/2)

                if min_distance is None:
                    min_distance = distance
                    min_coordinate = (x, y)
                elif distance < min_distance:
                    min_distance = distance
                    min_coordinate = (x, y)

        if min_distance < threshold_distance:
            index1, index2 = min_coordinate
            current_contours[index1] = np.concatenate((current_contours[index1], current_contours[index2]), axis=0)
            del current_contours[index2]
        else: 
            break

    return current_contours