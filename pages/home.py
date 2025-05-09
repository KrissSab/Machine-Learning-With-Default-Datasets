from PyQt6.QtWidgets import (
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QPushButton,
    QLabel,
    QGridLayout,
)
from PyQt6.QtCore import Qt

from pages.iris import Iris
from pages.spam_email import SpamEmail
from pages.titanic_decision_tree import TitanicDecisionTree
from pages.credit_approval_decision_tree import CreditApprovalDecisionTree
from pages.fashion_decision_tree import FashionDecisionTree
from pages.load_default_prediction import LoadDefaultPrediction


class HomeWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Main Window")
        self.setMinimumSize(1280, 720)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QVBoxLayout(central_widget)

        title_label = QLabel('Select task:')
        title_label.setAlignment(
            Qt.AlignmentFlag.AlignCenter)
        title_label.setStyleSheet("font-size: 24px; margin-bottom: 20px;")
        main_layout.addWidget(title_label)

        grid_layout = QGridLayout()
        main_layout.addLayout(grid_layout)

        self.exercise_windows = [
            Iris(),
            SpamEmail(),
            TitanicDecisionTree(),
            CreditApprovalDecisionTree(),
            FashionDecisionTree(),
            LoadDefaultPrediction()
        ]

        task_names = [
            "Iris",
            "Spam Email",
            "Titanic DT",
            "Credit Approval DT",
            "Fashion DT",
            "Load Default Prediction"
        ]

        for i in range(6):
            button = QPushButton(task_names[i])
            button.setMinimumHeight(50)
            button.clicked.connect(
                lambda checked, idx=i: self.open_exercise(idx))
            row, col = divmod(i, 3)
            grid_layout.addWidget(button, row, col)

    def open_exercise(self, index):
        self.hide()

        self.exercise_windows[index].show()

        self.exercise_windows[index].home_window = self
