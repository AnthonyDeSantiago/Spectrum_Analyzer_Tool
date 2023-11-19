# Driver in which we can cobble a solution together

from ultralytics import YOLO
import cv2
import numpy as np
from .ObjectDetector import ObjectDetector
from .get_signal import GetSignalWithCV2
import os
import multiprocessing
import csv
import time
from . import interface
from decord import VideoReader
from decord import cpu, gpu
from os import path

filePath=''
center_frequency=1.0
reference_level=0.0
span=100
IOC=-10.0
max_power = 0.0
min_power = 0.0

#def script_main(filePath='', reference_level=0.0, center_frequency=1.0, span=100, IOC=-10.0):


###################################################################################################
# TRAINED ML MODEL APPROACH
###################################################################################################

def script_trained_ml_approach():
    print("Script_main adjusted called")
    # Load the models
    model_g_s = path.abspath(path.join(path.dirname(__file__),'models/192_300Epochs_AllVideos.onnx'))
    # model_g_s = path.abspath(path.join(path.dirname(__file__),'CreateDataSet/runs/detect/train12/weights/best.onnx'))
    model_Grid = YOLO(model_g_s, task='detect')


    # Load the video
    video_path = filePath
    video = cv2.VideoCapture(video_path)


    # Grabbing these just in case
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video.get(cv2.CAP_PROP_FPS))


    # Instantiate object detectors here
    detector_grid = ObjectDetector(model=model_Grid, imgz=192)

    # A frame count so we can perform actions every ith, jth, kth, etc frame
    frame_nmr = 0

    # The frequency we want to perform actions
    read_freq = fps# <-- Gonna append to a list a frame once per second of video

    ret = True

    show = True

    lb_freq = center_frequency - ((span / 1000)/2)
    ub_freq = center_frequency + ((span / 1000)/2)

    lb_power = 0
    ub_power = 100

    start_time = time.time()
    with open('output.csv', 'w', newline='') as csvfile:
        fieldnames = ['Timestamp', 'Estimated Center Frequency (GHz)', 'Estimated Power (dB)']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # Write the header row
        writer.writeheader()

        start_time = time.time()
        while ret:
            if frame_nmr % read_freq == 0:
                video.set(cv2.CAP_PROP_POS_FRAMES, frame_nmr)
                ret, frame = video.read()

                if not ret:
                    break

                if ret:
                    if frame_nmr % read_freq == 0:
                        num_classes, x1, y1, x2, y2, conf, class_id, s_x1, s_y1, s_x2, s_y2, s_conf, s_class_id = process_frame(frame, detector_grid)
                        timestamp, estimated_center_frequency, estimated_power = get_signal_properties(frame_nmr, fps, x1, y1, x2, y2, s_x1, s_y1, s_x2, s_y2, lb_freq, ub_freq, lb_power, ub_power)
                        if show and num_classes == 2:
                            draw_hud(frame, x1, y1, x2, y2, s_x1, s_y1, s_x2, s_y2, estimated_center_frequency, estimated_power)
                            if cv2.waitKey(2) == ord('q'):
                                break
                        if num_classes >= 2:
                            
                            # Write the data to the CSV file
                            writer.writerow({'Timestamp': timestamp, 'Estimated Center Frequency (GHz)': estimated_center_frequency, 'Estimated Power (dB)': estimated_power})
                        else:
                            writer.writerow({'Timestamp': timestamp, 'Estimated Center Frequency (GHz)': 0, 'Estimated Power (dB)': 0})

            frame_nmr = frame_nmr + 1

    
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time} seconds")


    video.release()
    cv2.destroyAllWindows()



def process_batch(start_frame, end_frame, video, detector_grid, lb_freq, ub_freq, lb_power, ub_power, range_freq, range_power, read_freq):
    for frame_nmr in range(start_frame, end_frame):
        ret, frame = video.read()
        if not ret:
            return

        if frame_nmr % read_freq == 0:
            num_classes, x1, y1, x2, y2, conf, class_id, s_x1, s_y1, s_x2, s_y2, s_conf, s_class_id = process_frame(frame, detector_grid)
            if num_classes < 2 and num_classes != 0:
                x = 1
            else:
                grid_size_x = x2 - x1
                grid_size_y = y2 - y1

                midpoint = s_x2 - (s_x2 - s_x1) // 2
                estimated_center_frequency = ((midpoint - x1) / grid_size_x) * range_freq + lb_freq
                vertical_line_text = vertical_line_text + str(estimated_center_frequency) + " GHz"

                estimated_power = ((s_y1 - y1) / grid_size_y) * range_power + lb_power
                horizontal_line_text = horizontal_line_text + "{:.{}f}".format(estimated_power, 2) + " dB"


def process_frame(frame, detector):
    boxes = detector.getBoundingBoxes(frame)
    
    num_classes = 0
    x1, y1, x2, y2, conf, class_id = 0, 0, 0, 0, 0, 0 
    s_x1, s_y1, s_x2, s_y2, s_conf, s_class_id = 0, 0, 0, 0, 0, 0 

    if len(boxes) > 0:
        box = boxes[0]
        x1, y1, x2, y2, conf, class_id = map(int, box)
        num_classes = 1

    if len(boxes) > 1:
        box = boxes[1]
        s_x1, s_y1, s_x2, s_y2, s_conf, s_class_id = map(int, box)
        num_classes = 2

    return num_classes, x1, y1, x2, y2, conf, class_id, s_x1, s_y1, s_x2, s_y2, s_conf, s_class_id


def get_signal_properties(frame_nmr, fps, x1, y1, x2, y2, s_x1, s_y1, s_x2, s_y2, lb_freq, ub_freq, lb_power, ub_power):
    grid_size_x = x2 - x1
    grid_size_y = y2 - y1
    range_freq = ub_freq - lb_freq
    range_power = ub_power - lb_power

    midpoint = s_x2 - (s_x2 - s_x1) // 2

    timestamp = frame_nmr / fps
    estimated_center_frequency = ((midpoint - x1) / grid_size_x) * range_freq + lb_freq
    estimated_power = ((s_y1 - y1) / grid_size_y) * range_power + lb_power

    return timestamp, estimated_center_frequency, estimated_power


def draw_hud(frame, x1, y1, x2, y2, s_x1, s_y1, s_x2, s_y2, estimated_center_frequency, estimated_power):
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_color = (100, 255, 100)
    font_scale = 1
    font_thickness = 2

    horizontal_line_text = "Power: "
    vertical_line_text = "Center Freq: "

    vertical_line_text = vertical_line_text + str(estimated_center_frequency) + " GHz"
    horizontal_line_text = horizontal_line_text + "{:.{}f}".format(estimated_power, 2) + " dB"

    midpoint = s_x2 - (s_x2 - s_x1) // 2
    frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    frame = cv2.rectangle(frame, (s_x1, s_y1), (s_x2, s_y2), (0, 255, 0), 2)
    cv2.line(frame, (x1, s_y1), (x2, s_y1), (0, 0, 255), 2)
    cv2.line(frame, (midpoint, y1), (midpoint, y2), (0, 0, 255), 2)
    cv2.putText(frame, horizontal_line_text, (x1 + 500, s_y1 - 10), font, font_scale, text_color, font_thickness)
    cv2.putText(frame, vertical_line_text, (midpoint + 10, 900), font, font_scale, text_color, font_thickness)
    cv2.imshow("Visualizer", frame)

    return frame


def get_cpu_info():
    num_cores = os.cpu_count()

    num_available_cores = multiprocessing.cpu_count()
    print(f"Number of CPU cores: {num_cores}")
    print(f"Number of available CPU cores: {num_available_cores}")

    return num_cores, num_available_cores




###################################################################################################
# ORIGINAL APPROACH
###################################################################################################

def script_main():
    total_start = time.time()    
    # Load the models
    model_path_screen_finder = path.abspath(path.join(path.dirname(__file__),'models/Optimized_Resized_cl_1.onnx'))
    model_path_grid_finder = path.abspath(path.join(path.dirname(__file__),'models/Grid_LowRes_1-4_224.onnx'))
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
    signalSample = GetSignalWithCV2(frames, consecutive_frame_count, median_background_image, reference_level=0.0, center_frequency=1.0, span=100, IOC=-10.0)
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

