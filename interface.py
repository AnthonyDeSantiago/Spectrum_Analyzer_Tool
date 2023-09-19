import sys
import script
from xml.etree.ElementTree import tostringlist
from PyQt6.QtWidgets import QMainWindow, QApplication, QPushButton, QFileDialog
from PyQt6.QtCore import pyqtSlot

class Main(QMainWindow):
    def __init__(self):
        super().__init__()

        btn = QPushButton(self)
        btn.setText("Select Video")
        self.setCentralWidget(btn)
        btn.clicked.connect(self.open_dialog)
    
    @pyqtSlot()
    def open_dialog(self):
        fname = QFileDialog.getOpenFileName(
            self,
            "Open File",
            "",
            "Video Files (*.mp4);; All Files (*);;",
        )
        print(fname)

        # Make path into a variable
        filePath = fname[0]
        
        # Pass path into script
        script.script_main(filePath)
        
if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_gui = Main()
    main_gui.show()
    sys.exit(app.exec())