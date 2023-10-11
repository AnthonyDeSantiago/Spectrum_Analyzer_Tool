import sys

import main
# from xml.etree.ElementTree import tostringlist
from PyQt6.QtWidgets import QMainWindow, QApplication, QPushButton, QFileDialog, QLabel, QWidget, QInputDialog, QGraphicsWidget
from PyQt6.QtCore import pyqtSlot, Qt
from PyQt6 import uic

class Main(QMainWindow):

    def __init__(self):
        super().__init__()

        uic.loadUi("pyqt.ui", self)

        self.pushButton.clicked.connect(self.open_dialog)

        self.pushButton_2.clicked.connect(self.set_center)

        self.pushButton_3.clicked.connect(self.set_reference)

        self.pushButton_4.clicked.connect(self.set_span)

        self.pushButton_5.clicked.connect(self.call_main)
    
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
        d, ok = QInputDialog().getDouble(self, "Enter the Center:",
                               "Amount:", 0.0, -10000, 10000, 4,
                                Qt.WindowType.Dialog, 1)

        if ok:
            self.label_6.setText("Center: "+ str(d))
            # self.label_6.repaint()
            main.center_frequency = d
        else:
            self.label_6.setText("No Center Selected!")
            # self.label_6.repaint()

    def set_reference(self):
        d, ok = QInputDialog().getDouble(self, "Enter the Reference:",
                               "Amount:", 0.0, -10000, 10000, 4,
                                Qt.WindowType.Dialog, 1)

        if ok:
            self.label_7.setText("Reference: "+ str(d))
            # self.label_7.repaint()
            main.reference_level = d
        else:
            self.label_7.setText("No Reference Selected!")
            self.label_7.repaint()

    def set_span(self):
        d, ok = QInputDialog().getDouble(self, "Enter the Span:",
                               "Amount:", 0.0, -10000, 10000, 4,
                                Qt.WindowType.Dialog, 1)

        if ok:
            self.label_8.setText("Span: "+ str(d))
            # self.label_8.repaint()
            main.span = d
        else:
            self.label_8.setText("No Span Selected!")
            # self.label_8.repaint()

    def call_main(self):
        main.script_main()

        
if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_gui = Main()
    main_gui.show()
    sys.exit(app.exec())