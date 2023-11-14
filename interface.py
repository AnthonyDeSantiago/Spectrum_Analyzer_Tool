
import sys
import time
import io
import asyncio
from contextlib import redirect_stdout
import main
from PyQt6.QtWidgets import QMainWindow, QApplication, QFileDialog, QInputDialog, QGraphicsScene, QGraphicsView
from PyQt6.QtCore import pyqtSlot, Qt
from PyQt6.QtGui import QPixmap, QPainter
from PyQt6 import uic


str_out = io.StringIO()

class Main(QMainWindow):

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
        main.filePath = fname[0]  

        output = ("Please select the values for the video selected using the sample image below")
        self.label_2.setText(output)

        self.eventLog.appendPlainText("Video Selected: " + filePath)

        self.centerButton.setEnabled(True)
        self.referenceButton.setEnabled(True)
        self.spanButton.setEnabled(True)
        self.minPowerButton.setEnabled(True)
        self.maxPowerButton.setEnabled(True)

    def set_center(self):
        d, ok = QInputDialog().getDouble(self, "Center Frequency",
                               "Center Frequency in video:", 0.0, -10000, 10000, 4,
                                Qt.WindowType.Dialog, 1)

        if ok:
            self.label_6.setText("Center: "+ str(d) + " GHz")
            main.center_frequency = d
            self.eventLog.appendPlainText("Center Frequency Entered: " + str(d))
        else:
            self.label_6.setText("No Center Selected!")

    def set_reference(self):
        d, ok = QInputDialog().getDouble(self, "Reference Level",
                               "Reference Level in video:", 0.0, -10000, 10000, 4,
                                Qt.WindowType.Dialog, 1)

        if ok:
            self.label_7.setText("Reference: "+ str(d) +" dBm")
            main.reference_level = d
            self.eventLog.appendPlainText("Reference Level Entered: " + str(d))
        else:
            self.label_7.setText("No Reference Selected!")

    def set_span(self):
        d, ok = QInputDialog().getDouble(self, "Span",
                               "Span in video:", 0.0, -10000, 10000, 4,
                                Qt.WindowType.Dialog, 1)

        if ok:
            self.label_8.setText("Span: "+ str(d) +" MHz")
            main.span = d
            self.eventLog.appendPlainText("Span Entered: " + str(d))
        else:
            self.label_8.setText("No Span Selected!")

    def set_min_power(self):
        d, ok = QInputDialog().getDouble(self, "Minimum Power",
                               "Minimum Power in video:", 0.0, -10000, 10000, 4,
                                Qt.WindowType.Dialog, 1)

        if ok:
            self.label_9.setText("Minimum Power: "+ str(d) +" dB/")
            main.min_power = d
            self.eventLog.appendPlainText("Minimum Power Entered: " + str(d))
        else:
            self.label_9.setText("No Minimum Power Selected!")

    def set_max_power(self):
            d, ok = QInputDialog().getDouble(self, "Maximum Power",
                                "Maximum Power in video:", 0.0, -10000, 10000, 4,
                                    Qt.WindowType.Dialog, 1)

            if ok:
                self.maxPowerOutput.setText("Maximum Power: "+ str(d) +" dB/")
                main.max_power = d
                self.eventLog.appendPlainText("Maximum Power Entered: " + str(d))
            else:
                self.maxPowerOutput.setText("No Maximum Power Selected!")

    def call_model1(self):

        self.scriptStatus.setText("Processing Video...")
        self.repaint()

        with redirect_stdout(str_out):
            # QApplication.setOverrideCursor(pyqtSlot.WaitCursor)
            main.script_trained_ml_approach()
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



if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_gui = Main()
    main_gui.show()
    sys.exit(app.exec())