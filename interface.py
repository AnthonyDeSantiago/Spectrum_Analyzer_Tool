
import sys
import io
from contextlib import redirect_stdout
import main
from PyQt6.QtWidgets import QMainWindow, QApplication, QFileDialog, QInputDialog, QGraphicsScene, QGraphicsView
from PyQt6.QtCore import pyqtSlot, Qt
from PyQt6.QtGui import QPixmap, QPainter
from PyQt6 import uic


str_out = io.StringIO()

class Main(QMainWindow):

    img_array1 = ["Capture 1,C:\Spectrum\images\Capture1.jpg"]
    img_array2 = ["Capture 2,C:\Spectrum\images\Capture2.jpg"]
    img_array3 = ["Capture 3,C:\Spectrum\images\Capture3.jpg"]
    image_array = [img_array1,img_array2,img_array3]

    image_index = 0

    def __init__(self):
        super().__init__()

        uic.loadUi("pyqt.ui", self)

        self.pushButton.clicked.connect(self.open_dialog)

        self.pushButton_2.clicked.connect(self.set_center)

        self.pushButton_3.clicked.connect(self.set_reference)

        self.pushButton_4.clicked.connect(self.set_span)

        self.pushButton_5.clicked.connect(self.call_main)

        self.pushButton_6.clicked.connect(self.set_IOP)

        self.imgGotoFirst.clicked.connect(self.set_image_first)
        self.imgGotoPrevious.clicked.connect(self.set_image_previous)
        self.imgGotoNext.clicked.connect(self.set_image_next)
        self.imgGotoLast.clicked.connect(self.set_image_last)



        # Temp code for proof of images-------------------------------
        # scene = QGraphicsScene(0, 0, 400, 200)

        # pixmap = QPixmap("8x9u9cS.jpg")
        # pixmapitem = scene.addPixmap(pixmap)
        # pixmapitem.setPos(10, 10)

        # self.graphicsView.setScene(scene)
        # self.graphicsView.setRenderHint(QPainter.RenderHint.Antialiasing)
        # self.graphicsView.show()
        # Temp code for proof of images-------------------------------

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

        output = ("Please select Values for Video: " + filePath)

        self.label_2.setText(output)

        self.label_2.repaint()

        return filePath

    def set_center(self):
        d, ok = QInputDialog().getDouble(self, "Center:",
                               "Center on screen:", 0.0, -10000, 10000, 4,
                                Qt.WindowType.Dialog, 1)

        if ok:
            self.label_6.setText("Center: "+ str(d) + " GHz")
            main.center_frequency = d
        else:
            self.label_6.setText("No Center Selected!")

    def set_reference(self):
        d, ok = QInputDialog().getDouble(self, "Reference:",
                               "Reference Level on screen:", 0.0, -10000, 10000, 4,
                                Qt.WindowType.Dialog, 1)

        if ok:
            self.label_7.setText("Reference: "+ str(d) +" dBm")
            main.reference_level = d
        else:
            self.label_7.setText("No Reference Selected!")

    def set_span(self):
        d, ok = QInputDialog().getDouble(self, "Span:",
                               "Span on screen:", 0.0, -10000, 10000, 4,
                                Qt.WindowType.Dialog, 1)

        if ok:
            self.label_8.setText("Span: "+ str(d) +" MHz")
            main.span = d
        else:
            self.label_8.setText("No Span Selected!")

    def set_IOP(self):
        d, ok = QInputDialog().getDouble(self, "Increment of Power:",
                               "Increment of Power on screen:", 0.0, -10000, 10000, 4,
                                Qt.WindowType.Dialog, 1)

        if ok:
            self.label_9.setText("Inc of Power: "+ str(d) +" dB/")
            main.IOC = d
        else:
            self.label_9.setText("No Inc of Power Selected!")

    def call_main(self):
        with redirect_stdout(str_out):  
            main.script_main()

        out = str_out.getvalue()
        self.eventLog.appendPlainText(out)
        str_out.truncate(0)

        # self.eventLog.appendPlainText('Clicked on First Button')
        scene = QGraphicsScene(0, 0, 0, 0)

        Main.image_index = 0 
        image_prop = Main.image_array[Main.image_index]
        self.imgTitle.setText(image_prop[0].split(',')[0])
        image_path = image_prop[0].split(',')[1]
        pixmap = QPixmap(image_path)
        # pixmap.scaled(s[, aspectMode=Qt.IgnoreAspectRatio[, mode=Qt.FastTransformation]])
        pixmapitem = scene.addPixmap(pixmap)
        #pixmapitem.setPos(0, 0)

        self.graphicsView.setScene(scene)
        self.graphicsView.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.graphicsView.show()

        image_number = Main.image_index + 1
        total_image_number = len(Main.image_array)
        self.label_5.setText("Image: "+ str(image_number) + " / " + str(total_image_number))



    def set_image_first(self):
        self.eventLog.appendPlainText('Clicked on First Button')
        scene = QGraphicsScene(0, 0, 0, 0)

        Main.image_index = 0 
        image_prop = Main.image_array[Main.image_index]
        self.imgTitle.setText(image_prop[0].split(',')[0])
        image_path = image_prop[0].split(',')[1]
        pixmap = QPixmap(image_path)
        # pixmap.scaled(s[, aspectMode=Qt.IgnoreAspectRatio[, mode=Qt.FastTransformation]])
        pixmapitem = scene.addPixmap(pixmap)
        #pixmapitem.setPos(0, 0)

        self.graphicsView.setScene(scene)
        self.graphicsView.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.graphicsView.show()

        image_number = Main.image_index + 1
        total_image_number = len(Main.image_array)
        self.label_5.setText("Image: "+ str(image_number) + " / " + str(total_image_number))
            

    def set_image_previous(self):
        self.eventLog.appendPlainText('Clicked on Previous Button')
        scene = QGraphicsScene(0, 0, 0, 0)

        Main.image_index = Main.image_index - 1 
        image_prop = Main.image_array[Main.image_index]
        self.imgTitle.setText(image_prop[0].split(',')[0])
        image_path = image_prop[0].split(',')[1]
        pixmap = QPixmap(image_path)
        pixmapitem = scene.addPixmap(pixmap)

        self.graphicsView.setScene(scene)
        self.graphicsView.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.graphicsView.show()

        image_number = Main.image_index + 1
        total_image_number = len(Main.image_array)
        self.label_5.setText("Image: "+ str(image_number) + " / " + str(total_image_number))

    def set_image_next(self):
        self.eventLog.appendPlainText('Clicked on Next Button')
        scene = QGraphicsScene(0, 0, 0, 0)

        Main.image_index = Main.image_index + 1

        image_prop = Main.image_array[Main.image_index]
        self.imgTitle.setText(image_prop[0].split(',')[0])
        image_path = image_prop[0].split(',')[1]
        pixmap = QPixmap(image_path)
        pixmapitem = scene.addPixmap(pixmap)

        self.graphicsView.setScene(scene)
        self.graphicsView.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.graphicsView.show()

        image_number = Main.image_index + 1
        total_image_number = len(Main.image_array)
        self.label_5.setText("Image: "+ str(image_number) + " / " + str(total_image_number))

    def set_image_last(self):
        self.eventLog.appendPlainText('Clicked on Last Button')
        scene = QGraphicsScene(0, 0, 0, 0)

        image_index = len(Main.image_array) - 1
        image_prop = Main.image_array[image_index]
        self.imgTitle.setText(image_prop[0].split(',')[0])
        image_path = image_prop[0].split(',')[1]
        pixmap = QPixmap(image_path)
        pixmapitem = scene.addPixmap(pixmap)

        self.graphicsView.setScene(scene)
        self.graphicsView.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.graphicsView.show()

        image_number = Main.image_index + 1
        total_image_number = len(Main.image_array)
        self.label_5.setText("Image: "+ str(image_number) + " / " + str(total_image_number))






if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_gui = Main()
    main_gui.show()
    sys.exit(app.exec())