import sys
from PyQt6.QtWidgets import QApplication
from pages.home import HomeWindow

if __name__ == "__main__":
    app = QApplication(sys.argv)
    home_window = HomeWindow()
    home_window.show()
    sys.exit(app.exec())
