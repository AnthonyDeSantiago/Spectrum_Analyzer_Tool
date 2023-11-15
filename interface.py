
import sys
import time
import io
import asyncio
from contextlib import redirect_stdout
import main
from PyQt6.QtWidgets import QMainWindow, QApplication, QFileDialog, QInputDialog, QGraphicsScene, QGraphicsView, QGraphicsPixmapItem
from PyQt6.QtCore import pyqtSlot, Qt
from PyQt6.QtGui import QPixmap, QPainter, QImage
from PyQt6 import uic
from datetime import datetime

import os
import multiprocessing
import csv
from ultralytics import YOLO
import cv2
import numpy as np
from ObjectDetector import ObjectDetector


str_out = io.StringIO()


center_frequency=1.0
reference_level=0.0
span=100
max_power = 0.0
min_power = 0.0



# Load the models
model_g_s = 'models/192_200Epochs_AllVideos.onnx'
model_Grid = YOLO(model_g_s, task='detect')
detector_grid = ObjectDetector(model=model_Grid, imgz=192)

class Main(QMainWindow):

    filePath=''
    test_frame = ""

    center_frequency= None
    reference_level= None
    span= None
    max_power = None
    min_power = None

    img_array1 = ["Capture 1,images\Capture1.jpg"]
    img_array2 = ["Capture 2,images\Capture2.jpg"]
    img_array3 = ["Capture 3,images\Capture3.jpg"]
    image_array = [img_array1,img_array2,img_array3]

    image_index = 0

    def __init__(self):
        super().__init__()

        uic.loadUi("pyqt.ui", self)

        self.pushButton.clicked.connect(self.open_dialog)

        self.centerButton.clicked.connect(self.set_center)
        self.referenceButton.clicked.connect(self.set_reference)
        self.spanButton.clicked.connect(self.set_span)
        self.minPowerButton.clicked.connect(self.set_min_power)
        self.maxPowerButton.clicked.connect(self.set_max_power)

        self.script1Button.clicked.connect(self.call_model1)
        self.script2Button.clicked.connect(self.call_model2)

        self.imgGotoFirst.clicked.connect(self.set_image_first)
        self.imgGotoPrevious.clicked.connect(self.set_image_previous)
        self.imgGotoNext.clicked.connect(self.set_image_next)
        self.imgGotoLast.clicked.connect(self.set_image_last)

        self.resetLog.clicked.connect(self.reset_log)
        self.downloadLog.clicked.connect(self.download_log)

    @pyqtSlot()
    def open_dialog(self):
        fname = QFileDialog.getOpenFileName(
            self,
            "Open File",
            "",
            "Video Files (*.mp4);;",
        )
        print(fname)

        # Make path into a variable
        filePath = fname[0]
        self.filePath = fname[0]
        main.filePath = fname[0]
        
        if filePath != '':
            test_video = cv2.VideoCapture(filePath)
            ret, self.test_frame = test_video.read()
            test_video.release()

            height, width, channel = self.test_frame.shape
            self.test_frame = cv2.resize(self.test_frame, (width // 2, height // 2))

            bytes_per_line = 3 * width // 2
            q_image = QImage(self.test_frame.data, width // 2, height // 2, bytes_per_line, QImage.Format.Format_BGR888)
            scene = QGraphicsScene(0, 0, width // 2, height // 2)
            pixmap_item = QGraphicsPixmapItem(QPixmap.fromImage(q_image))
            scene.addItem(pixmap_item)
            self.graphicsView.setScene(scene)
            self.graphicsView.setRenderHint(QPainter.RenderHint.Antialiasing)
            self.graphicsView.show()

            self.imgTitle.setText("Sample Image of Video")

            output = ("Please select the values for the video selected using the sample image below")
            self.label_2.setText(output)

            self.eventLog.appendPlainText("Video Selected: " + filePath)

            self.centerButton.setEnabled(True)
            self.referenceButton.setEnabled(True)
            self.spanButton.setEnabled(True)
            self.minPowerButton.setEnabled(True)
            self.maxPowerButton.setEnabled(True)

        else:
            output = ("No Video Selected!")
            self.label_2.setText(output)

    def var_check(self):
        if self.center_frequency != None and self.reference_level != None and self.span != None and self.max_power != None and self.min_power != None:
            self.script1Button.setEnabled(True)
            self.script2Button.setEnabled(True)
            self.displayCheckBox.setEnabled(True)
            self.modelSelectText.setText("Select a script to run")
        # else:
        #     self.eventLog.appendPlainText(" Center =" + str(self.center_frequency) + " ref =" + str(self.reference_level) + " span =" + str(self.span) + 
        #                                   " min =" + str(self.min_power) + " max =" + str(self.max_power))

    def set_center(self):
        d, ok = QInputDialog().getDouble(self, "Center Frequency",
                               "Center Frequency in GHz:", 0.0, -10000, 10000, 4,
                                Qt.WindowType.Dialog, 1)

        if ok:
            self.label_6.setText("Center: "+ str(d) + " GHz")
            main.center_frequency = d
            self.center_frequency = d
            self.eventLog.appendPlainText("Center Frequency Entered: " + str(d))
            self.var_check()
        else:
            self.label_6.setText("No Center Selected!")

    def set_reference(self):
        d, ok = QInputDialog().getDouble(self, "Reference Level",
                               "Reference Level in dBm:", 0.0, -10000, 10000, 4,
                                Qt.WindowType.Dialog, 1)

        if ok:
            self.label_7.setText("Reference: "+ str(d) +" dBm")
            main.reference_level = d
            self.reference_level = d
            self.eventLog.appendPlainText("Reference Level Entered: " + str(d))
            self.var_check()
        else:
            self.label_7.setText("No Reference Selected!")

    def set_span(self):
        d, ok = QInputDialog().getDouble(self, "Span",
                               "Span in GHz:", 0.0, -10000, 10000, 4,
                                Qt.WindowType.Dialog, 1)

        if ok:
            self.label_8.setText("Span: "+ str(d) +" GHz")
            main.span = d
            self.span = d
            self.eventLog.appendPlainText("Span Entered: " + str(d))
            self.var_check()
        else:
            self.label_8.setText("No Span Selected!")

    def set_min_power(self):
        d, ok = QInputDialog().getDouble(self, "Minimum Power",
                               "Minimum Power in dB:", 0.0, -10000, 10000, 4,
                                Qt.WindowType.Dialog, 1)

        if ok:
            self.label_9.setText("Minimum Power: "+ str(d) +" dB")
            main.min_power = d
            self.min_power = d
            self.eventLog.appendPlainText("Minimum Power Entered: " + str(d))
            self.var_check()
        else:
            self.label_9.setText("No Minimum Power Selected!")

    def set_max_power(self):
            d, ok = QInputDialog().getDouble(self, "Maximum Power",
                                "Maximum Power in dB:", 0.0, -10000, 10000, 4,
                                    Qt.WindowType.Dialog, 1)

            if ok:
                self.maxPowerOutput.setText("Maximum Power: "+ str(d) +" dB")
                main.max_power = d
                self.max_power = d
                self.eventLog.appendPlainText("Maximum Power Entered: " + str(d))
                self.var_check()
            else:
                self.maxPowerOutput.setText("No Maximum Power Selected!")

    def call_model1(self):
        self.scriptStatus.setText("Processing Video...")
        self.repaint()

        check_for_check = self.displayCheckBox.isChecked()

        with redirect_stdout(str_out):
            # QApplication.setOverrideCursor(pyqtSlot.WaitCursor)
            print(self.filePath)
            self.script_trained_ml_approach(self.filePath, check_for_check)
            # QApplication.restoreOverrideCursor()

        out = str_out.getvalue()
        self.eventLog.appendPlainText(out)
        str_out.truncate(0)

        self.scriptStatus.setText("Video Processed! CSV file has been added to directory")

        self.imgGotoFirst.setEnabled(True)
        self.imgGotoPrevious.setEnabled(True)
        self.imgGotoNext.setEnabled(True)
        self.imgGotoLast.setEnabled(True)

        scene = QGraphicsScene(0, 0, 0, 0)

        Main.image_index = 0
        image_prop = Main.image_array[Main.image_index]
        self.imgTitle.setText(image_prop[0].split(',')[0])
        image_path = image_prop[0].split(',')[1]
        pixmap = QPixmap(image_path)
        pixmapitem = scene.addPixmap(pixmap)
        pixmapitem.setPos(0, 0)

        self.graphicsView.setScene(scene)
        self.graphicsView.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.graphicsView.show()

        image_number = Main.image_index + 1
        total_image_number = len(Main.image_array)
        self.label_5.setText("Image: "+ str(image_number) + " / " + str(total_image_number))

    def call_model2(self):
        self.scriptStatus.setText("Processing Video...")
        self.repaint()

        with redirect_stdout(str_out):
            # QApplication.setOverrideCursor(pyqtSlot.WaitCursor)
            main.script_main()
            # QApplication.restoreOverrideCursor()

        out = str_out.getvalue()
        self.eventLog.appendPlainText(out)
        str_out.truncate(0)

        self.scriptStatus.setText("Video Processed! CSV file has been added to directory")

        self.imgGotoFirst.setEnabled(True)
        self.imgGotoPrevious.setEnabled(True)
        self.imgGotoNext.setEnabled(True)
        self.imgGotoLast.setEnabled(True)

        scene = QGraphicsScene(0, 0, 0, 0)

        Main.image_index = 0
        image_prop = Main.image_array[Main.image_index]
        self.imgTitle.setText(image_prop[0].split(',')[0])
        image_path = image_prop[0].split(',')[1]
        pixmap = QPixmap(image_path)
        pixmapitem = scene.addPixmap(pixmap)
        pixmapitem.setPos(0, 0)

        self.graphicsView.setScene(scene)
        self.graphicsView.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.graphicsView.show()

        image_number = Main.image_index + 1
        total_image_number = len(Main.image_array)
        self.label_5.setText("Image: "+ str(image_number) + " / " + str(total_image_number))


    async def runModels():
        await asyncio.run(main.script_main)
 
    def set_image_first(self):
        scene = QGraphicsScene(0, 0, 0, 0)

        Main.image_index = 0
        image_prop = Main.image_array[Main.image_index]
        self.imgTitle.setText(image_prop[0].split(',')[0])
        image_path = image_prop[0].split(',')[1]
        pixmap = QPixmap(image_path)
        # pixmap.scaled(s[, aspectMode=Qt.IgnoreAspectRatio[, mode=Qt.FastTransformation]])
        pixmapitem = scene.addPixmap(pixmap)
        pixmapitem.setPos(0, 0)

        self.graphicsView.setScene(scene)
        self.graphicsView.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.graphicsView.show()

        image_number = Main.image_index + 1
        total_image_number = len(Main.image_array)
        self.label_5.setText("Image: "+ str(image_number) + " / " + str(total_image_number))
            

    def set_image_previous(self):
        scene = QGraphicsScene(0, 0, 0, 0)

        Main.image_index = Main.image_index - 1 
        
        if (Main.image_index < 0):
            Main.image_index = 0

            image_prop = Main.image_array[Main.image_index]
            self.imgTitle.setText(image_prop[0].split(',')[0])

            image_path = image_prop[0].split(',')[1]
            pixmap = QPixmap(image_path)
            pixmapitem = scene.addPixmap(pixmap)
            pixmapitem.setPos(0, 0)

            self.graphicsView.setScene(scene)
            self.graphicsView.setRenderHint(QPainter.RenderHint.Antialiasing)
            self.graphicsView.show()

            image_number = Main.image_index + 1
            total_image_number = len(Main.image_array)
            self.label_5.setText("Image: "+ str(image_number) + " / " + str(total_image_number))

        else:
            image_prop = Main.image_array[Main.image_index]
            self.imgTitle.setText(image_prop[0].split(',')[0])

            image_path = image_prop[0].split(',')[1]
            pixmap = QPixmap(image_path)
            pixmapitem = scene.addPixmap(pixmap)
            pixmapitem.setPos(0, 0)

            self.graphicsView.setScene(scene)
            self.graphicsView.setRenderHint(QPainter.RenderHint.Antialiasing)
            self.graphicsView.show()

            image_number = Main.image_index + 1
            total_image_number = len(Main.image_array)
            self.label_5.setText("Image: "+ str(image_number) + " / " + str(total_image_number))

    def set_image_next(self):

        if (Main.image_index < len(Main.image_array) - 1):
            scene = QGraphicsScene(0, 0, 0, 0)
            Main.image_index = Main.image_index + 1

            # if (Main.image_index > len(Main.image_array) + 1):
            # image_index = len(Main.image_array) - 1

            image_prop = Main.image_array[Main.image_index]
            self.imgTitle.setText(image_prop[0].split(',')[0])

            image_path = image_prop[0].split(',')[1]
            pixmap = QPixmap(image_path)
            pixmapitem = scene.addPixmap(pixmap)
            pixmapitem.setPos(0, 0)

            self.graphicsView.setScene(scene)
            self.graphicsView.setRenderHint(QPainter.RenderHint.Antialiasing)
            self.graphicsView.show()

            image_number = Main.image_index + 1
            total_image_number = len(Main.image_array)
            self.label_5.setText("Image: "+ str(image_number) + " / " + str(total_image_number))

            # else:
            #     image_prop = Main.image_array[Main.image_index]
            #     self.imgTitle.setText(image_prop[0].split(',')[0])

            #     image_path = image_prop[0].split(',')[1]
            #     pixmap = QPixmap(image_path)
            #     pixmapitem = scene.addPixmap(pixmap)
            #     pixmapitem.setPos(0, 0)

            #     self.graphicsView.setScene(scene)
            #     self.graphicsView.setRenderHint(QPainter.RenderHint.Antialiasing)
            #     self.graphicsView.show()

            #     image_number = Main.image_index + 1
            #     total_image_number = len(Main.image_array)
            #     self.label_5.setText("Image: "+ str(image_number) + " / " + str(total_image_number))

    def set_image_last(self):
        scene = QGraphicsScene(0, 0, 0, 0)

        Main.image_index = len(Main.image_array) - 1

        image_prop = Main.image_array[Main.image_index]
        self.imgTitle.setText(image_prop[0].split(',')[0])

        image_path = image_prop[0].split(',')[1]
        pixmap = QPixmap(image_path)
        pixmapitem = scene.addPixmap(pixmap)
        pixmapitem.setPos(0, 0)

        self.graphicsView.setScene(scene)
        self.graphicsView.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.graphicsView.show()

        image_number = len(Main.image_array)
        total_image_number = len(Main.image_array)
        self.label_5.setText("Image: "+ str(image_number) + " / " + str(total_image_number))

    def reset_log(self):
        self.eventLog.clear()

    def download_log(self):
        time = datetime.now()
        text_file = open("log_"+str(time.month)+"_"+str(time.day)+"_"+str(time.year)+"_"+str(time.hour)+"_"+str(time.minute)+".txt", "w")

        # Take content
        content = self.eventLog.toPlainText()

        # Write content to file
        n = text_file.write(content)

###################################################################
# Script_Trained_ML_Approach
###################################################################
    def script_trained_ml_approach(self, videopath, check_for_check):
        print("script_trained_ml_approach called")
        


        # Load the models
        model_g_s = 'models/192_200Epochs_AllVideos.onnx'
        # model_g_s = 'CreateDataSet/runs/detect/train12/weights/best.onnx'
        model_Grid = YOLO(model_g_s, task='detect')


        # Load the video
        # video_path = filePath
        video = cv2.VideoCapture(videopath)


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

        show = check_for_check

        lb_freq = center_frequency - ((span / 1000)/2)
        ub_freq = center_frequency + ((span / 1000)/2)

        lb_power = 0
        ub_power = 100

        incomming_powers = []

        name_time = datetime.now()
        output_filename = "Out_"+ str(name_time.month) +"_"+ str(name_time.day)+"_"+str(name_time.year) +"_"+ str(name_time.hour)+"_"+ str(name_time.minute)+ "_" + str(name_time.second) + ".csv"

        start_time = time.time()
        with open(output_filename, 'w', newline='') as csvfile:
            fieldnames = ['Timestamp', 'Frequency (GHz)', 'Power (dBm)', 'Min Power (dBm)', 'Max Power (dBm)', 'Avg Power (dBm)']
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
                            if num_classes >= 2:
                                if show:
                                    draw_hud(frame, x1, y1, x2, y2, s_x1, s_y1, s_x2, s_y2, estimated_center_frequency, estimated_power)
                                    if cv2.waitKey(2) == ord('q'):
                                        break
                                incomming_powers.append(estimated_power)
                                # Write the data to the CSV file
                                writer.writerow({'Timestamp': timestamp, 'Frequency (GHz)': estimated_center_frequency, 'Power (dBm)': estimated_power, 'Min Power (dBm)': 0, 'Max Power (dBm)': 0, 'Avg Power (dBm)': 0})
                            else:
                                if len(incomming_powers) > 0:
                                    minumum_power = min(incomming_powers)
                                    max_power = max(incomming_powers)
                                    average_power = sum(incomming_powers) / len(incomming_powers)
                                    writer.writerow({'Timestamp': timestamp, 'Frequency (GHz)': 0, 'Power (dBm)': 0, 'Min Power (dBm)': minumum_power, 'Max Power (dBm)': max_power, 'Avg Power (dBm)': average_power})
                                    incomming_powers = []
                                else:
                                     writer.writerow({'Timestamp': timestamp, 'Frequency (GHz)': 0, 'Power (dBm)': 0, 'Min Power (dBm)': 0, 'Max Power (dBm)': 0, 'Avg Power (dBm)': 0})

                frame_nmr = frame_nmr + 1

        
        end_time = time.time()
        execution_time = end_time - start_time
        total_vid_length = frame_nmr / fps
        print(f"Execution time: {execution_time} seconds")
        print(f"Video length: {total_vid_length} seconds")


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
    estimated_power = lb_power - ((s_y1 - y1) / grid_size_y) * range_power

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
    height, width, channel = frame.shape
    
    frame = cv2.resize(frame, (width // 2, height // 2))
    cv2.imshow("Visualizer", frame)
    return frame


def get_cpu_info():
    num_cores = os.cpu_count()

    num_available_cores = multiprocessing.cpu_count()
    print(f"Number of CPU cores: {num_cores}")
    print(f"Number of available CPU cores: {num_available_cores}")

    return num_cores, num_available_cores

###################################################################
###################################################################



if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_gui = Main()
    main_gui.show()
    sys.exit(app.exec())