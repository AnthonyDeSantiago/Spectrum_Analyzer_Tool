import sys
import main
# from xml.etree.ElementTree import tostringlist
from PyQt6.QtWidgets import QMainWindow, QApplication, QPushButton, QFileDialog, QLabel, QWidget
from PyQt6.QtCore import pyqtSlot
from PyQt6 import uic

class Main(QMainWindow):
    def __init__(self):
        super().__init__()

        # self.setWindowTitle("Spectrum Analyzer Automator")
        # self.setGeometry(200,200,750,535)

        uic.loadUi("pyqt.ui", self)
        
        btn = QPushButton(self)
        btn.setGeometry(225,200,300,100)
        btn.setText("Select Video")
        # btn.setFont("Times", 14)
        # self.setCentralWidget(btn)
        btn.clicked.connect(self.open_dialog)

        # label = QLabel("Please select the Spectrum Analyzer recording using the button below", self)
        # label.move(0,0)
    
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
        
        # Pass path into script
        main.script_main(filePath)
        

        output = ("Current File Path: " + filePath)

        self.label_2.setText(output)

        self.label_2.repaint()

        
if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_gui = Main()
    main_gui.show()
    sys.exit(app.exec())