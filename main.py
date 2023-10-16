# Driver in which we can cobble a solution together

from ultralytics import YOLO
import cv2
import numpy as np
from ObjectDetector import ObjectDetector
from get_signal import GetSignalWithCV2
import time

def script_main(filePath='', reference_level=0.0, center_frequency=1.0, span=100):
    total_start = time.time()    
    # Load the models
    model_path_screen_finder = 'models/Optimized_Resized_cl_1.onnx'
    model_path_grid_finder = 'models/Grid_LowRes_1-4_224.onnx'
    model_SpecScreen = YOLO(model_path_screen_finder, task='detect') # <---Model specifically for finding spectrum analyzer screen
    model_Grid = YOLO(model_path_grid_finder, task='detect')

    # Load the video
    video_path = filePath
    video = cv2.VideoCapture(video_path)

    ###################################################################################################
    # SET UP VIDEO FOR ANALYSIS
    ###################################################################################################

    # Signal analysis modifiers -- impacts video processing
    consecutive_frame_count = 5 # <-- should be able to evenly divide FPS with no remainder

    # Variables
    #width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    #height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video.get(cv2.CAP_PROP_FPS))
    approx_frame_count = video.get(cv2.CAP_PROP_FRAME_COUNT)

    # The frequency we want to perform actions
    detect_gridCrop_freq = fps * 60 * 20 # <-- Arbitrarily choosing once every 20 mins for updating grid bounding box
    append_gridCrop_freq = fps // consecutive_frame_count # <-- # frames per second to grab, # = consecutive_frame_count
    process_frames_freq = fps * 60 * 20 # <-- Set it to once every 20 mins for now

    # Setting up batch count for the whole video, including tail batch handling
    frame_batch_total = 1

    if approx_frame_count > process_frames_freq:    
        frame_batch_total = int(approx_frame_count // process_frames_freq)
        frame_batch_remainder = approx_frame_count % process_frames_freq

        if frame_batch_remainder > (fps * 60 * 5): frame_batch_total += 1 # <-- If >5 mins will remain in final segm, add 1 to segm total
        else: frame_batch_total = 1
    
        print("Number of frame batches: " + str(frame_batch_total))
    
    # Instantiate object detectors here
    detector_grid = ObjectDetector(model=model_Grid, imgz=224)

    frames = [] # <-- This will hold the frames 
        
    #--------------------------------------------------------------------------------------------------
    # Isolate Grid
    #--------------------------------------------------------------------------------------------------
    
    print("\nGrid isolation starting -------------------------------------------\n")
    start = time.time()

    # A frame count so we can perform actions every ith, jth, kth, etc frame
    frame_nmr = 0

    grid_crop = []
    ret = True
        
    while ret:
        if frame_nmr % append_gridCrop_freq == 0: #<-- Check if its time to append a new frame to frames list

            video.set(cv2.CAP_PROP_POS_FRAMES, frame_nmr)
            ret, frame = video.read()

            if ret:
                if frame_nmr % detect_gridCrop_freq == 0: #<-- Check if it's time to update the bounding boxes
                    box = detector_grid.getBoundingBoxes(frame)[0]
                    x1, y1, x2, y2, conf, class_id = map(int, box)
                    grid_crop = [x1, y1, x2, y2]

                isolated_grid_crop = frame[grid_crop[1]: grid_crop[3], grid_crop[0]: grid_crop[2]]

                #------------------------------------------
                # Image cleaning & pre-processing
                 
                gray = cv2.cvtColor(isolated_grid_crop, cv2.COLOR_BGR2GRAY)
                #----------

                frames.append(gray)

        frame_nmr += 1

    end = time.time()
    print("\n>>> Grid isolation took " + str(end-start) + "s\n")
    

    #--------------------------------------------------------------------------------------------------
    # Get Static Background Image (as close to just the grid as possible)
    #--------------------------------------------------------------------------------------------------
    print("\nRetrieving background image... ------------------------------------")
    start = time.time()

    # Setting up background processing
    background_frames = []
    background_sample_size = 500
    background_frame_indices_raw = np.random.uniform(0, len(frames)-1, background_sample_size)
    background_frame_indices = background_frame_indices_raw.astype(int)

    # Get background frame array
    for index in background_frame_indices:
        background_frames.append(frames[index]) 
    
    # Calculate the median image
    median_background_image = np.median(background_frames, axis=0).astype(np.uint8)
    
    end = time.time()
    print("\n>>> Background image retrieval took " + str(end-start) + "s\n")
    
    #For testing purposes - can remove
    #cv2.imshow("background median", median_background_image)
    #cv2.waitKey(5000)

    ###################################################################################################
    # VIDEO ANALYSIS RUN
    ###################################################################################################
    
    #Running through one batch at a time -- TBD, MAKE THIS FUNCTIONAL
    #for i in range(0, frame_batch_total):
        #----------------------------------------------------------------------------------------------
        # Perform Signal Analysis 
        #----------------------------------------------------------------------------------------------
    print('\nSignal analysis starting ------------------------------------------\n')
    start = time.time()
        
    #run signal analysis (includes appending results to CSV)
    signalSample = GetSignalWithCV2(frames, consecutive_frame_count, median_background_image)
    signalSample.get_signal()

    end = time.time()
    print(">>> Signal processing took " + str(end-start) + "s\n")
    #---------
    frames = [] #<-----Clearing out 'frames' list to start adding fresh frames for the next 20 min segment

        
    video.release()
    cv2.destroyAllWindows()
    total_end = time.time()

    total_time = total_end - total_start
    approx_video_seconds = approx_frame_count / fps
    tps = total_time / approx_video_seconds

    print("\n-------------------------------------------------------------------\n")

    print("\nTOTAL VIDEO PROCESSING TIME >>> " + str(total_time) + "s")
    print("\tApproximate video length >>> " + str(approx_video_seconds) + "s")
    print("\tTime cost per second of video >>> " + str(tps) + "s/s")
    print("\tVideo processed " + str((abs(total_time - approx_video_seconds)/approx_video_seconds)*100) + "% faster than real time\n")



