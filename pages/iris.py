from PyQt6.QtWidgets import (
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QPushButton,
    QLabel,
    QTextEdit,
    QComboBox,
    QHBoxLayout,
    QGridLayout,
    QSpinBox,
)
from PyQt6.QtCore import Qt
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import seaborn as sns


class Iris(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Iris")
        self.setMinimumSize(1280, 720)

        self.load_data()

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QVBoxLayout(central_widget)

        title_label = QLabel("Iris")
        title_label.setAlignment(
            Qt.AlignmentFlag.AlignCenter)
        title_label.setStyleSheet("font-size: 24px; margin-bottom: 20px;")
        main_layout.addWidget(title_label)

        content_layout = QHBoxLayout()
        main_layout.addLayout(content_layout)

        self.figure = plt.figure(figsize=(10, 8))
        self.canvas = FigureCanvas(self.figure)
        content_layout.addWidget(self.canvas, 2)

        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        content_layout.addWidget(right_panel, 1)

        controls_group = QWidget()
        controls_layout = QGridLayout(controls_group)
        right_layout.addWidget(controls_group)

        k_label = QLabel("Value of k:")
        controls_layout.addWidget(k_label, 0, 0)

        self.k_spinbox = QSpinBox()
        self.k_spinbox.setMinimum(1)
        self.k_spinbox.setMaximum(20)
        self.k_spinbox.setValue(5)
        controls_layout.addWidget(self.k_spinbox, 0, 1)

        metric_label = QLabel("Metric:")
        controls_layout.addWidget(metric_label, 1, 0)

        self.metric_combo = QComboBox()
        self.metric_combo.addItems(["euclidean", "manhattan", "minkowski"])
        controls_layout.addWidget(self.metric_combo, 1, 1)

        features_label = QLabel("Features:")
        controls_layout.addWidget(features_label, 2, 0)

        self.feature_x_combo = QComboBox()
        self.feature_x_combo.addItems(self.iris.feature_names)
        controls_layout.addWidget(self.feature_x_combo, 2, 1)

        self.feature_y_combo = QComboBox()
        self.feature_y_combo.addItems(self.iris.feature_names)
        self.feature_y_combo.setCurrentIndex(1)
        controls_layout.addWidget(self.feature_y_combo, 3, 1)

        buttons_layout = QHBoxLayout()
        right_layout.addLayout(buttons_layout)

        self.train_button = QPushButton("Train model")
        self.train_button.clicked.connect(self.train_model)
        buttons_layout.addWidget(self.train_button)

        self.find_best_k_button = QPushButton("Find best k")
        self.find_best_k_button.clicked.connect(self.find_best_k)
        buttons_layout.addWidget(self.find_best_k_button)

        self.visualize_button = QPushButton("Visualize data")
        self.visualize_button.clicked.connect(self.visualize_data)
        buttons_layout.addWidget(self.visualize_button)

        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        right_layout.addWidget(self.results_text)

        back_button = QPushButton("Return to main")
        back_button.clicked.connect(self.go_back)
        main_layout.addWidget(back_button)

        self.home_window = None

    def load_data(self):
        # Завантаження датасету Iris
        self.iris = load_iris()
        self.X = self.iris.data
        self.y = self.iris.target

        # Попередня обробка даних
        self.scaler = StandardScaler()
        self.X_scaled = self.scaler.fit_transform(self.X)

        # Розбиття на навчальний та тестовий набори (80% навчальний, 20% тестовий)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X_scaled, self.y, test_size=0.2, random_state=42
        )

        # Створення DataFrame для зручності
        self.df = pd.DataFrame(
            data=np.c_[self.X, self.y],
            columns=self.iris.feature_names + ["target"]
        )
        self.df["species"] = self.df["target"].map({
            0: "setosa",
            1: "versicolor",
            2: "virginica"
        })

    def train_model(self):
        k = self.k_spinbox.value()
        metric = self.metric_combo.currentText()

        # Створення та навчання моделі
        self.knn = KNeighborsClassifier(n_neighbors=k, metric=metric)
        self.knn.fit(self.X_train, self.y_train)

        # Прогнозування
        y_pred = self.knn.predict(self.X_test)

        # Оцінка продуктивності
        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred, average='weighted')
        recall = recall_score(self.y_test, y_pred, average='weighted')
        f1 = f1_score(self.y_test, y_pred, average='weighted')

        # Виведення результатів
        results = f"Результати для k={k}, метрика={metric}:\n\n"
        results += f"Accuracy: {accuracy:.4f}\n"
        results += f"Precision: {precision:.4f}\n"
        results += f"Recall: {recall:.4f}\n"
        results += f"F1-score: {f1:.4f}\n\n"

        # Додавання звіту про класифікацію
        report = classification_report(self.y_test, y_pred,
                                       target_names=self.iris.target_names)
        results += "Детальний звіт про класифікацію:\n" + report

        self.results_text.setText(results)

        # Візуалізація матриці помилок
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        cm = confusion_matrix(self.y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=self.iris.target_names,
                    yticklabels=self.iris.target_names, ax=ax)
        ax.set_xlabel('Прогнозовані класи')
        ax.set_ylabel('Фактичні класи')
        ax.set_title('Матриця помилок')
        self.canvas.draw()

    def find_best_k(self):
        metric = self.metric_combo.currentText()
        k_range = range(1, 21)
        k_scores = []

        # Перехресна перевірка для різних значень k
        for k in k_range:
            knn = KNeighborsClassifier(n_neighbors=k, metric=metric)
            scores = cross_val_score(
                knn, self.X_scaled, self.y, cv=10, scoring='accuracy')
            k_scores.append(scores.mean())

        # Знаходження оптимального k
        best_k = k_range[np.argmax(k_scores)]
        best_score = max(k_scores)

        # Виведення результатів
        results = f"Пошук оптимального значення k (метрика={metric}):\n\n"
        results += f"Найкраще значення k: {best_k}\n"
        results += f"Точність при перехресній перевірці: {best_score:.4f}\n"

        self.results_text.setText(results)

        # Встановлення найкращого k у спінбокс
        self.k_spinbox.setValue(best_k)

        # Візуалізація результатів
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.plot(k_range, k_scores)
        ax.set_xlabel('Значення k')
        ax.set_ylabel('Точність перехресної перевірки')
        ax.set_title('Точність k-NN для різних значень k')
        ax.grid(True)
        self.canvas.draw()

    def visualize_data(self):
        # Отримання вибраних ознак
        feature_x = self.feature_x_combo.currentIndex()
        feature_y = self.feature_y_combo.currentIndex()

        # Очищення попередніх графіків
        self.figure.clear()

        # Створення графіка розсіювання
        ax = self.figure.add_subplot(111)

        # Кольори для різних класів
        colors = ['blue', 'red', 'green']

        # Побудова графіка для кожного класу
        for i, species in enumerate(self.iris.target_names):
            ax.scatter(
                self.X[self.y == i, feature_x],
                self.X[self.y == i, feature_y],
                c=colors[i],
                label=species,
                alpha=0.7,
                edgecolors='k'
            )

        ax.set_xlabel(self.iris.feature_names[feature_x])
        ax.set_ylabel(self.iris.feature_names[feature_y])
        ax.set_title('Distribution of iris species by selected features')
        ax.legend()
        ax.grid(True)

        self.canvas.draw()

    def go_back(self):
        self.hide()

        if self.home_window:
            self.home_window.show()

    def closeEvent(self, event):
        if self.home_window:
            self.home_window.show()
        event.accept()
