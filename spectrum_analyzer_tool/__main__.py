import sys
from PyQt6.QtWidgets import QApplication
from .interface import Main

app = QApplication(sys.argv)
main_gui = Main()
main_gui.show()
sys.exit(app.exec())