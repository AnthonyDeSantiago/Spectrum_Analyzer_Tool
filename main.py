# Driver in which we can cobble a solution together

from ultralytics import YOLO
import cv2
import numpy as np
from ObjectDetector import ObjectDetector
from get_signal import GetSignalWithCV2
import time
from decord import VideoReader
from decord import cpu, gpu

filePath=''
reference_level=0.0
center_frequency=1.0
span=100
IOC = 0.0

# def script_main(filePath='', reference_level=0.0, center_frequency=1.0, span=100):

def script_main():
    total_start = time.time()    
    # Load the models
    model_path_screen_finder = 'models/Optimized_Resized_cl_1.onnx'
    model_path_grid_finder = 'models/Grid_LowRes_1-4_224.onnx'
    model_SpecScreen = YOLO(model_path_screen_finder, task='detect') # <---Model specifically for finding spectrum analyzer screen
    model_Grid = YOLO(model_path_grid_finder, task='detect')

    # Load the video
    video_path = filePath
    video = VideoReader(filePath, ctx=cpu(0))

    ###################################################################################################
    # SET UP VIDEO FOR ANALYSIS
    ###################################################################################################

    # Signal analysis modifiers -- impacts video processing
    consecutive_frame_count = 5 # <-- should be able to evenly divide FPS with no remainder

    # Variables
    fps = int(video.get_avg_fps())
    approx_frame_count = len(video)

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
    # USING DECORD
    #--------------------------------------------------------------------------------------------------
    print("\nGrid isolation using decord starting ------------------------------\n")
    start = time.time()

    start_f = 0
    end_f = len(video)
    every_f = append_gridCrop_freq

    frames_list = list(range(start_f, end_f, every_f))

    grid_crop = []

    if every_f > 25 and len(frames_list) < 1000:  # this is faster for every > 25 frames and can fit in memory
        frames = video.get_batch(frames_list).asnumpy()

        for index, frame in zip(frames_list, frames):  # lets loop through the frames until the end
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            if index % (detect_gridCrop_freq / append_gridCrop_freq) == 0: #<-- Check if it's time to update the bounding boxes
                box = detector_grid.getBoundingBoxes(frame)[0]
                x1, y1, x2, y2, conf, class_id = map(int, box)
                grid_crop = [x1, y1, x2, y2]

            isolated_grid_crop = frame[grid_crop[1]: grid_crop[3], grid_crop[0]: grid_crop[2]]

            #------------------------------------------
            # Image cleaning & pre-processing
            gray = cv2.cvtColor(isolated_grid_crop, cv2.COLOR_BGR2GRAY)
            #----------

            frames.append(gray)

    else:  # this is faster for every <25 and consumes small memory
        for index in range(start_f, end_f):  # lets loop through the frames until the end
            frame = video[index]  # read an image from the capture
            frame = cv2.cvtColor(frame.asnumpy(), cv2.COLOR_RGB2BGR)

            if index % every_f == 0:  # if this is a frame we want to write out based on the 'every' argument
                if index % (detect_gridCrop_freq / append_gridCrop_freq) == 0: #<-- Check if it's time to update the bounding boxes
                    box = detector_grid.getBoundingBoxes(frame)[0]
                    x1, y1, x2, y2, conf, class_id = map(int, box)
                    grid_crop = [x1, y1, x2, y2]

                isolated_grid_crop = frame[grid_crop[1]: grid_crop[3], grid_crop[0]: grid_crop[2]]

                #------------------------------------------
                # Image cleaning & pre-processing
                gray = cv2.cvtColor(isolated_grid_crop, cv2.COLOR_BGR2GRAY)
                #----------

                frames.append(gray)

    

    end = time.time()
    print("\n>>> Grid isolation using decord took " + str(end-start) + "s\n")

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



