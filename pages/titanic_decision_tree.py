from PyQt6.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QPushButton, QLabel
from PyQt6.QtCore import Qt


class TitanicDecisionTree(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Titanic DT")
        self.setMinimumSize(1280, 720)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QVBoxLayout(central_widget)

        title_label = QLabel("Titanic DT")
        title_label.setAlignment(
            Qt.AlignmentFlag.AlignCenter)
        title_label.setStyleSheet("font-size: 24px; margin-bottom: 20px;")
        main_layout.addWidget(title_label)

        content_label = QLabel("Content of task 3...")
        content_label.setAlignment(
            Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(content_label)

        back_button = QPushButton("Return to main")
        back_button.clicked.connect(self.go_back)
        main_layout.addWidget(back_button)

        self.home_window = None

    def go_back(self):
        self.hide()

        if self.home_window:
            self.home_window.show()

    def closeEvent(self, event):
        if self.home_window:
            self.home_window.show()
        event.accept()
