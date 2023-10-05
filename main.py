# Driver in which we can cobble a solution together

from ultralytics import YOLO
import cv2
import numpy as np
from ObjectDetector import ObjectDetector

def script_main(filePath, reference_level=0, center_frequency=1, span=100):

    # Load the models
    model_path_screen_finder = 'models/Optimized_Resized_cl_1.onnx'
    model_path_grid_finder = 'models/Grid_LowRes_1-4_224.onnx'
    model_SpecScreen = YOLO(model_path_screen_finder, task='detect') #<---Model specifically for finding spectrum analyzer screen
    model_Grid = YOLO(model_path_grid_finder, task='detect')


    # Load the video
    video_path = filePath
    video = cv2.VideoCapture(video_path)


    # Grabbing these just in case
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video.get(cv2.CAP_PROP_FPS))


    # Instantiate object detectors here
    detector_grid = ObjectDetector(model=model_Grid, imgz=224)

    # A frame count so we can perform actions every ith, jth, kth, etc frame
    frame_nmr = 0

    # The frequency we want to perform actions
    detect_gridCrop_freq = fps * 60 * 20 # <-- Arbitrarily choosing once every 20 mins for updating grid bounding box
    append_gridCrop_freq = fps # <-- Gonna append to a list a frame once per second of video
    process_frames_freq = fps * 60 * 20 # <-- Set it to once every 20 mins for now

    
    grid_crop = []
    ret = True


    frames = [] # <-- This will hold the frames 

    while ret:
        
        
        #------------------------------------
        # Isolate Grid
        #------------------------------------
      
        if frame_nmr % append_gridCrop_freq == 0: #<-- Check if its time to append a new frame to frames list

            video.set(cv2.CAP_PROP_POS_FRAMES, frame_nmr)
            ret, frame = video.read()

            if ret:
                if frame_nmr % detect_gridCrop_freq == 0: #<-- Check if it's time to update the bounding boxes
                    box = detector_grid.getBoundingBoxes(frame)[0]
                    x1, y1, x2, y2, conf, class_id = map(int, box)
                    grid_crop = [x1, y1, x2, y2]

                isolated_grid_crop = frame[grid_crop[1]: grid_crop[3], grid_crop[0]: grid_crop[2]]
                frames.append(isolated_grid_crop)
        
        # For Testing Purposes - Remove
        cv2.imshow("Isolated grid crop", isolated_grid_crop)

        
        #------------------------------------
        # Perform Signal Processing Stuff Here
        #------------------------------------
        
        if frame_nmr % process_frames_freq == 0:
            print('')






        # For Testing Purposes - Remove
        if cv2.waitKey(2) == ord('q'):
            break

        frame_nmr = frame_nmr + 1



    video.release()
    cv2.destroyAllWindows()



