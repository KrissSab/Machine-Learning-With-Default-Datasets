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
    QCheckBox,
    QTabWidget,
    QFileDialog,
    QDoubleSpinBox,
    QGroupBox,
)
from PyQt6.QtCore import Qt
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
)
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
import numpy as np
import pandas as pd
import seaborn as sns
import os
import io
from contextlib import redirect_stdout


class LoanDefaultPrediction(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(
            "Loan Default Prediction")
        self.setMinimumSize(1280, 720)

        # Центральний віджет
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Головний layout
        main_layout = QVBoxLayout(central_widget)

        # Заголовок
        title_label = QLabel(
            "Loan Default Prediction")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_label.setStyleSheet("font-size: 24px; margin-bottom: 20px;")
        main_layout.addWidget(title_label)

        # Кнопка для завантаження даних
        load_data_layout = QHBoxLayout()
        main_layout.addLayout(load_data_layout)

        self.load_data_button = QPushButton("Завантажити дані")
        self.load_data_button.clicked.connect(self.load_data)
        load_data_layout.addWidget(self.load_data_button)

        self.data_status_label = QLabel("Статус: Дані не завантажено")
        load_data_layout.addWidget(self.data_status_label)

        # Створення горизонтального layout для графіків та контролів
        content_layout = QHBoxLayout()
        main_layout.addLayout(content_layout)

        # Створення вкладок для різних візуалізацій
        self.tabs = QTabWidget()
        content_layout.addWidget(self.tabs, 2)  # Співвідношення 2:1

        # Вкладка для візуалізації даних
        self.data_tab = QWidget()
        self.tabs.addTab(self.data_tab, "Дані")
        data_layout = QVBoxLayout(self.data_tab)

        self.data_figure = plt.figure(figsize=(10, 8))
        self.data_canvas = FigureCanvas(self.data_figure)
        data_layout.addWidget(self.data_canvas)

        # Вкладка для ROC-кривої
        self.roc_tab = QWidget()
        self.tabs.addTab(self.roc_tab, "ROC-крива")
        roc_layout = QVBoxLayout(self.roc_tab)

        self.roc_figure = plt.figure(figsize=(10, 8))
        self.roc_canvas = FigureCanvas(self.roc_figure)
        roc_layout.addWidget(self.roc_canvas)

        # Вкладка для Precision-Recall кривої
        self.pr_tab = QWidget()
        self.tabs.addTab(self.pr_tab, "Precision-Recall")
        pr_layout = QVBoxLayout(self.pr_tab)

        self.pr_figure = plt.figure(figsize=(10, 8))
        self.pr_canvas = FigureCanvas(self.pr_figure)
        pr_layout.addWidget(self.pr_canvas)

        # Вкладка для коефіцієнтів моделі
        self.coef_tab = QWidget()
        self.tabs.addTab(self.coef_tab, "Коефіцієнти")
        coef_layout = QVBoxLayout(self.coef_tab)

        self.coef_figure = plt.figure(figsize=(10, 8))
        self.coef_canvas = FigureCanvas(self.coef_figure)
        coef_layout.addWidget(self.coef_canvas)

        # Права панель для контролів та результатів
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        content_layout.addWidget(right_panel, 1)  # Співвідношення 2:1

        # Контроли для логістичної регресії
        controls_group = QGroupBox("Параметри моделі")
        controls_layout = QGridLayout(controls_group)
        right_layout.addWidget(controls_group)

        # Регуляризація
        regularization_label = QLabel("Тип регуляризації:")
        controls_layout.addWidget(regularization_label, 0, 0)

        self.regularization_combo = QComboBox()
        self.regularization_combo.addItems(["l1", "l2", "elasticnet", "none"])
        self.regularization_combo.setCurrentText("l2")
        controls_layout.addWidget(self.regularization_combo, 0, 1)

        # Параметр C (обернений до сили регуляризації)
        c_param_label = QLabel("Параметр C:")
        controls_layout.addWidget(c_param_label, 1, 0)

        self.c_param_spin = QDoubleSpinBox()
        self.c_param_spin.setMinimum(0.01)
        self.c_param_spin.setMaximum(10.0)
        self.c_param_spin.setSingleStep(0.1)
        self.c_param_spin.setValue(1.0)
        controls_layout.addWidget(self.c_param_spin, 1, 1)

        # Параметр l1_ratio для elasticnet
        l1_ratio_label = QLabel("l1_ratio (для elasticnet):")
        controls_layout.addWidget(l1_ratio_label, 2, 0)

        self.l1_ratio_spin = QDoubleSpinBox()
        self.l1_ratio_spin.setMinimum(0.0)
        self.l1_ratio_spin.setMaximum(1.0)
        self.l1_ratio_spin.setSingleStep(0.1)
        self.l1_ratio_spin.setValue(0.5)
        controls_layout.addWidget(self.l1_ratio_spin, 2, 1)

        # Максимальна кількість ітерацій
        max_iter_label = QLabel("Макс. ітерацій:")
        controls_layout.addWidget(max_iter_label, 3, 0)

        self.max_iter_combo = QComboBox()
        self.max_iter_combo.addItems(["100", "200", "500", "1000"])
        self.max_iter_combo.setCurrentText("100")
        controls_layout.addWidget(self.max_iter_combo, 3, 1)

        # Розмір тестового набору
        test_size_label = QLabel("Розмір тестового набору:")
        controls_layout.addWidget(test_size_label, 4, 0)

        self.test_size_combo = QComboBox()
        self.test_size_combo.addItems(["10%", "20%", "30%", "40%"])
        self.test_size_combo.setCurrentText("20%")
        controls_layout.addWidget(self.test_size_combo, 4, 1)

        # Опції попередньої обробки
        preprocessing_group = QGroupBox("Попередня обробка")
        preprocessing_layout = QGridLayout(preprocessing_group)
        right_layout.addWidget(preprocessing_group)

        # Масштабування даних
        scaling_label = QLabel("Масштабування:")
        preprocessing_layout.addWidget(scaling_label, 0, 0)

        self.scaling_combo = QComboBox()
        self.scaling_combo.addItems(
            ["Без масштабування", "StandardScaler", "MinMaxScaler"])
        self.scaling_combo.setCurrentText("StandardScaler")
        preprocessing_layout.addWidget(self.scaling_combo, 0, 1)

        # Обробка викидів
        outliers_label = QLabel("Обробка викидів:")
        preprocessing_layout.addWidget(outliers_label, 1, 0)

        self.outliers_check = QCheckBox("Видаляти викиди")
        preprocessing_layout.addWidget(self.outliers_check, 1, 1)

        # Кнопки
        buttons_layout = QHBoxLayout()
        right_layout.addLayout(buttons_layout)

        self.train_button = QPushButton("Навчити модель")
        self.train_button.clicked.connect(self.train_model)
        buttons_layout.addWidget(self.train_button)

        self.grid_search_button = QPushButton("Grid Search")
        self.grid_search_button.clicked.connect(self.perform_grid_search)
        buttons_layout.addWidget(self.grid_search_button)

        self.visualize_button = QPushButton("Візуалізувати дані")
        self.visualize_button.clicked.connect(self.visualize_data)
        buttons_layout.addWidget(self.visualize_button)

        # Текстове поле для результатів
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        right_layout.addWidget(self.results_text)

        # Кнопка повернення на головну
        back_button = QPushButton("Повернутися на головну")
        back_button.clicked.connect(self.go_back)
        main_layout.addWidget(back_button)

        # Посилання на головне вікно (буде встановлено при відкритті)
        self.home_window = None

        # Ініціалізація даних
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.model = None
        self.feature_names = None

        # Деактивація кнопок до завантаження даних
        self.train_button.setEnabled(False)
        self.grid_search_button.setEnabled(False)
        self.visualize_button.setEnabled(False)

    def load_data(self):
        # Відкриття діалогу вибору файлу
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Виберіть файл даних", "", "CSV Files (*.csv)")

        if file_path:
            try:
                # Завантаження файлу
                self.data = pd.read_csv(file_path)

                # Перевірка наявності цільової змінної
                if 'Defaulted?' not in self.data.columns:
                    self.data_status_label.setText(
                        "Статус: Помилка! Не знайдено цільову змінну 'Defaulted?'")
                    return

                # Активація кнопок
                self.train_button.setEnabled(True)
                self.grid_search_button.setEnabled(True)
                self.visualize_button.setEnabled(True)

                # Оновлення статусу
                self.data_status_label.setText(f"Статус: Дані завантажено успішно. "
                                               f"Кількість зразків: {self.data.shape[0]}, "
                                               f"Кількість ознак: {self.data.shape[1]-2}")  # -2 для Index та Defaulted?

                # Початкова візуалізація
                self.visualize_data()

            except Exception as e:
                self.data_status_label.setText(
                    f"Статус: Помилка при завантаженні даних: {str(e)}")

    def preprocess_data(self):
        if self.data is None:
            return None, None, None, None

        try:
            # Копіювання даних
            df = self.data.copy()

            # Видалення індексного стовпця, якщо він є
            if 'Index' in df.columns:
                df = df.drop('Index', axis=1)

            # Обробка викидів, якщо увімкнено
            if self.outliers_check.isChecked():
                # Визначення числових стовпців
                numeric_cols = df.select_dtypes(
                    include=['int64', 'float64']).columns.tolist()
                numeric_cols = [
                    col for col in numeric_cols if col != 'Defaulted?']

                # Видалення викидів (значень, що виходять за межі 1.5 * IQR)
                for col in numeric_cols:
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    df = df[(df[col] >= lower_bound) &
                            (df[col] <= upper_bound)]

            # Розділення на ознаки та цільову змінну
            X = df.drop('Defaulted?', axis=1)
            y = df['Defaulted?']

            # Збереження назв ознак
            self.feature_names = X.columns.tolist()

            # Розбиття на навчальний та тестовий набори
            test_size = float(
                self.test_size_combo.currentText().replace("%", "")) / 100
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y
            )

            return X_train, X_test, y_train, y_test

        except Exception as e:
            self.results_text.setText(
                f"Помилка при попередній обробці даних: {str(e)}")
            return None, None, None, None

    def build_pipeline(self):
        # Вибір методу масштабування
        scaling_method = self.scaling_combo.currentText()

        # Створення пайплайну
        steps = []

        # Додавання масштабування, якщо вибрано
        if scaling_method == "StandardScaler":
            steps.append(('scaler', StandardScaler()))
        elif scaling_method == "MinMaxScaler":
            steps.append(('scaler', MinMaxScaler()))

        # Вибір параметрів логістичної регресії
        penalty = self.regularization_combo.currentText()
        C = self.c_param_spin.value()
        l1_ratio = self.l1_ratio_spin.value()
        max_iter = int(self.max_iter_combo.currentText())

        # Налаштування параметрів логістичної регресії
        if penalty == "none":
            logreg_params = {
                'penalty': None,
                'max_iter': max_iter,
                'random_state': 42
            }
        elif penalty == "elasticnet":
            logreg_params = {
                'penalty': penalty,
                'C': C,
                'l1_ratio': l1_ratio,
                'solver': 'saga',  # Єдиний solver, що підтримує elasticnet
                'max_iter': max_iter,
                'random_state': 42
            }
        else:  # l1 або l2
            logreg_params = {
                'penalty': penalty,
                'C': C,
                'solver': 'liblinear',  # Хороший вибір для малих датасетів
                'max_iter': max_iter,
                'random_state': 42
            }

        # Додавання логістичної регресії до пайплайну
        steps.append(('classifier', LogisticRegression(**logreg_params)))

        # Створення пайплайну
        pipeline = Pipeline(steps)

        return pipeline

    def train_model(self):
        if self.data is None:
            self.results_text.setText("Помилка: Дані не завантажено.")
            return

        try:
            # Попередня обробка даних
            X_train, X_test, y_train, y_test = self.preprocess_data()

            if X_train is None:
                return

            # Збереження для подальшого використання
            self.X_train = X_train
            self.X_test = X_test
            self.y_train = y_train
            self.y_test = y_test

            # Створення та навчання моделі
            self.model = self.build_pipeline()
            self.model.fit(X_train, y_train)

            # Прогнозування
            y_pred = self.model.predict(X_test)
            y_prob = self.model.predict_proba(X_test)[:, 1]

            # Оцінка продуктивності
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)

            # Виведення результатів
            penalty = self.regularization_combo.currentText()
            C = self.c_param_spin.value()
            l1_ratio = self.l1_ratio_spin.value()
            max_iter = self.max_iter_combo.currentText()
            scaling_method = self.scaling_combo.currentText()

            results = f"Результати для логістичної регресії:\n\n"
            results += f"Параметри моделі:\n"
            results += f"- Регуляризація: {penalty}\n"
            results += f"- Параметр C: {C}\n"
            if penalty == "elasticnet":
                results += f"- l1_ratio: {l1_ratio}\n"
            results += f"- Макс. ітерацій: {max_iter}\n"
            results += f"- Масштабування: {scaling_method}\n"
            results += f"- Видалення викидів: {'Так' if self.outliers_check.isChecked() else 'Ні'}\n\n"
            results += f"Метрики продуктивності:\n"
            results += f"- Accuracy: {accuracy:.4f}\n"
            results += f"- Precision: {precision:.4f}\n"
            results += f"- Recall: {recall:.4f}\n"
            results += f"- F1-score: {f1:.4f}\n\n"

            # Додавання звіту про класифікацію
            report = classification_report(y_test, y_pred, target_names=[
                                           "Немає дефолту", "Дефолт"])
            results += "Детальний звіт про класифікацію:\n" + report

            self.results_text.setText(results)

            # Візуалізація матриці помилок
            self.data_figure.clear()
            ax = self.data_figure.add_subplot(111)
            cm = confusion_matrix(y_test, y_pred)
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                        xticklabels=["Немає дефолту", "Дефолт"],
                        yticklabels=["Немає дефолту", "Дефолт"], ax=ax)
            ax.set_xlabel('Прогнозовані класи')
            ax.set_ylabel('Фактичні класи')
            ax.set_title('Матриця помилок')
            self.data_canvas.draw()

            # Візуалізація ROC-кривої
            self.visualize_roc_curve(y_test, y_prob)

            # Візуалізація Precision-Recall кривої
            self.visualize_pr_curve(y_test, y_prob)

            # Візуалізація коефіцієнтів моделі
            self.visualize_coefficients()

            # Перехід на вкладку з ROC-кривою
            self.tabs.setCurrentIndex(1)

        except Exception as e:
            self.results_text.setText(f"Помилка при навчанні моделі: {str(e)}")

    def perform_grid_search(self):
        if self.data is None:
            self.results_text.setText("Помилка: Дані не завантажено.")
            return

        try:
            # Попередня обробка даних
            X_train, X_test, y_train, y_test = self.preprocess_data()

            if X_train is None:
                return

            # Вибір методу масштабування
            scaling_method = self.scaling_combo.currentText()

            # Створення базового пайплайну
            steps = []

            # Додавання масштабування, якщо вибрано
            if scaling_method == "StandardScaler":
                steps.append(('scaler', StandardScaler()))
            elif scaling_method == "MinMaxScaler":
                steps.append(('scaler', MinMaxScaler()))

            # Додавання логістичної регресії до пайплайну
            steps.append(('classifier', LogisticRegression(random_state=42)))

            # Створення пайплайну
            pipeline = Pipeline(steps)

            # Параметри для пошуку
            param_grid = {
                'classifier__penalty': ['l1', 'l2', 'elasticnet', None],
                'classifier__C': [0.1, 0.5, 1.0, 5.0],
                'classifier__solver': ['liblinear', 'saga'],
                'classifier__max_iter': [100, 500]
            }

            # Виконання пошуку по сітці
            grid_search = GridSearchCV(
                pipeline, param_grid, cv=5, scoring='accuracy')
            grid_search.fit(X_train, y_train)

            # Отримання найкращих параметрів
            best_params = grid_search.best_params_

            # Оновлення контролів з найкращими параметрами
            if best_params['classifier__penalty'] is None:
                self.regularization_combo.setCurrentText("none")
            else:
                self.regularization_combo.setCurrentText(
                    best_params['classifier__penalty'])

            self.c_param_spin.setValue(best_params['classifier__C'])
            self.max_iter_combo.setCurrentText(
                str(best_params['classifier__max_iter']))

            # Виведення результатів
            results = f"Результати Grid Search:\n\n"
            results += f"Найкращі параметри:\n"
            results += f"- Регуляризація: {best_params['classifier__penalty']}\n"
            results += f"- Параметр C: {best_params['classifier__C']}\n"
            results += f"- Solver: {best_params['classifier__solver']}\n"
            results += f"- Макс. ітерацій: {best_params['classifier__max_iter']}\n\n"
            results += f"Найкраща точність при перехресній перевірці: {grid_search.best_score_:.4f}\n\n"

            # Додавання таблиці результатів
            results += "Таблиця результатів для різних параметрів:\n"
            cv_results = pd.DataFrame(grid_search.cv_results_)
            cv_results = cv_results.sort_values('rank_test_score')

            # Вибір тільки важливих стовпців
            important_cols = [
                col for col in cv_results.columns if 'param_' in col or 'mean_test_score' in col or 'rank_test_score' in col]
            cv_results = cv_results[important_cols].head(
                10)  # Показати тільки топ-10 результатів

            # Перенаправлення виводу DataFrame у рядок
            buffer = io.StringIO()
            with redirect_stdout(buffer):
                print(cv_results.to_string())
            results += buffer.getvalue()

            self.results_text.setText(results)

        except Exception as e:
            self.results_text.setText(
                f"Помилка при виконанні Grid Search: {str(e)}")

    def visualize_roc_curve(self, y_test, y_prob):
        # Очищення попередніх графіків
        self.roc_figure.clear()
        ax = self.roc_figure.add_subplot(111)

        # Обчислення ROC-кривої
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)

        # Побудова ROC-кривої
        ax.plot(fpr, tpr, color='darkorange', lw=2,
                label=f'ROC крива (AUC = {roc_auc:.3f})')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC крива')
        ax.legend(loc="lower right")
        ax.grid(True)

        self.roc_canvas.draw()

    def visualize_pr_curve(self, y_test, y_prob):
        # Очищення попередніх графіків
        self.pr_figure.clear()
        ax = self.pr_figure.add_subplot(111)

        # Обчислення Precision-Recall кривої
        precision, recall, _ = precision_recall_curve(y_test, y_prob)
        average_precision = average_precision_score(y_test, y_prob)

        # Побудова Precision-Recall кривої
        ax.plot(recall, precision, color='blue', lw=2,
                label=f'Precision-Recall крива (AP = {average_precision:.3f})')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('Precision-Recall крива')
        ax.legend(loc="lower left")
        ax.grid(True)

        self.pr_canvas.draw()

    def visualize_coefficients(self):
        if self.model is None:
            return

        try:
            # Отримання логістичної регресії з пайплайну
            logreg = self.model.named_steps['classifier']

            # Отримання коефіцієнтів
            coefficients = logreg.coef_[0]

            # Отримання назв ознак
            feature_names = self.feature_names

            # Створення DataFrame для сортування
            coef_df = pd.DataFrame({
                'Feature': feature_names,
                'Coefficient': coefficients
            })

            # Сортування за абсолютним значенням коефіцієнтів
            coef_df = coef_df.reindex(
                coef_df['Coefficient'].abs().sort_values(ascending=False).index)

            # Очищення попередніх графіків
            self.coef_figure.clear()
            ax = self.coef_figure.add_subplot(111)

            # Побудова графіка коефіцієнтів
            colors = ['red' if c <
                      0 else 'green' for c in coef_df['Coefficient']]
            ax.barh(coef_df['Feature'], coef_df['Coefficient'], color=colors)
            ax.set_xlabel('Коефіцієнт')
            ax.set_ylabel('Ознака')
            ax.set_title('Коефіцієнти логістичної регресії')
            ax.grid(True, axis='x')

            # Додавання вертикальної лінії на нулі
            ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)

            # Додавання значень коефіцієнтів на графік
            for i, v in enumerate(coef_df['Coefficient']):
                ax.text(v + 0.01 if v >= 0 else v -
                        0.2, i, f"{v:.3f}", va='center')

            self.coef_canvas.draw()

        except Exception as e:
            print(f"Помилка при візуалізації коефіцієнтів: {str(e)}")

    def visualize_data(self):
        if self.data is None:
            return

        # Очищення попередніх графіків
        self.data_figure.clear()

        # Створення підграфіків
        gs = self.data_figure.add_gridspec(2, 2)
        ax1 = self.data_figure.add_subplot(gs[0, 0])
        ax2 = self.data_figure.add_subplot(gs[0, 1])
        ax3 = self.data_figure.add_subplot(gs[1, 0])
        ax4 = self.data_figure.add_subplot(gs[1, 1])

        # 1. Розподіл цільової змінної
        target_counts = self.data['Defaulted?'].value_counts()
        ax1.bar(['Немає дефолту', 'Дефолт'], [
                target_counts.get(0, 0), target_counts.get(1, 0)])
        ax1.set_title('Розподіл дефолтів')
        ax1.set_ylabel('Кількість')

        # 2. Розподіл банківського балансу за дефолтом
        if 'Bank Balance' in self.data.columns:
            sns.boxplot(x='Defaulted?', y='Bank Balance',
                        data=self.data, ax=ax2)
            ax2.set_title('Банківський баланс за дефолтом')
            ax2.set_xlabel('Дефолт')
            ax2.set_ylabel('Банківський баланс')
            ax2.set_xticklabels(['Немає дефолту', 'Дефолт'])

        # 3. Розподіл річної зарплати за дефолтом
        if 'Annual Salary' in self.data.columns:
            sns.boxplot(x='Defaulted?', y='Annual Salary',
                        data=self.data, ax=ax3)
            ax3.set_title('Річна зарплата за дефолтом')
            ax3.set_xlabel('Дефолт')
            ax3.set_ylabel('Річна зарплата')
            ax3.set_xticklabels(['Немає дефолту', 'Дефолт'])

        # 4. Розподіл за працевлаштуванням та дефолтом
        if 'Employed' in self.data.columns:
            # Створення крос-таблиці
            employed_default = pd.crosstab(
                self.data['Employed'], self.data['Defaulted?'])
            employed_default.plot(kind='bar', stacked=True, ax=ax4)
            ax4.set_title('Працевлаштування та дефолт')
            ax4.set_xlabel('Працевлаштований')
            ax4.set_ylabel('Кількість')
            ax4.set_xticklabels(['Ні', 'Так'])
            ax4.legend(['Немає дефолту', 'Дефолт'])

        # Налаштування макету
        self.data_figure.tight_layout()
        self.data_canvas.draw()

        # Перехід на вкладку з даними
        self.tabs.setCurrentIndex(0)

    def go_back(self):
        # Приховуємо поточне вікно
        self.hide()

        # Показуємо головне вікно
        if self.home_window:
            self.home_window.show()

    def closeEvent(self, event):
        # При закритті вікна також показуємо головне вікно
        if self.home_window:
            self.home_window.show()
        event.accept()
