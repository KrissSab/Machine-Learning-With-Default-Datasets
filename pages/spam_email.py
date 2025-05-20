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
)
from PyQt6.QtCore import Qt
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
)
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.feature_extraction.text import TfidfTransformer
import numpy as np
import pandas as pd
import seaborn as sns
from ucimlrepo import fetch_ucirepo


class SpamEmail(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Spam Email")
        self.setMinimumSize(1280, 720)

        self.load_data()

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QVBoxLayout(central_widget)

        title_label = QLabel("Spam Email")
        title_label.setAlignment(
            Qt.AlignmentFlag.AlignCenter)
        title_label.setStyleSheet("font-size: 24px; margin-bottom: 20px;")
        main_layout.addWidget(title_label)

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

        # Вкладка для результатів моделі
        self.results_tab = QWidget()
        self.tabs.addTab(self.results_tab, "Результати")
        results_layout = QVBoxLayout(self.results_tab)

        self.results_figure = plt.figure(figsize=(10, 8))
        self.results_canvas = FigureCanvas(self.results_figure)
        results_layout.addWidget(self.results_canvas)

        # Вкладка для ROC-кривої
        self.roc_tab = QWidget()
        self.tabs.addTab(self.roc_tab, "ROC-крива")
        roc_layout = QVBoxLayout(self.roc_tab)

        self.roc_figure = plt.figure(figsize=(10, 8))
        self.roc_canvas = FigureCanvas(self.roc_figure)
        roc_layout.addWidget(self.roc_canvas)

        # Права панель для контролів та результатів
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        content_layout.addWidget(right_panel, 1)  # Співвідношення 2:1

        # Контроли для наївного байєсівського класифікатора
        controls_group = QWidget()
        controls_layout = QGridLayout(controls_group)
        right_layout.addWidget(controls_group)

        # Вибір типу наївного байєсівського класифікатора
        nb_type_label = QLabel("Тип класифікатора:")
        controls_layout.addWidget(nb_type_label, 0, 0)

        self.nb_type_combo = QComboBox()
        self.nb_type_combo.addItems(
            ["Gaussian NB", "Multinomial NB", "Bernoulli NB"])
        controls_layout.addWidget(self.nb_type_combo, 0, 1)

        # Вибір методу попередньої обробки
        preprocessing_label = QLabel("Попередня обробка:")
        controls_layout.addWidget(preprocessing_label, 1, 0)

        self.preprocessing_combo = QComboBox()
        self.preprocessing_combo.addItems(
            ["Без обробки", "StandardScaler", "MinMaxScaler", "TF-IDF"])
        controls_layout.addWidget(self.preprocessing_combo, 1, 1)

        # Вибір відбору ознак
        feature_selection_label = QLabel("Відбір ознак:")
        controls_layout.addWidget(feature_selection_label, 2, 0)

        self.feature_selection_check = QCheckBox("Використовувати SelectKBest")
        controls_layout.addWidget(self.feature_selection_check, 2, 1)

        # Вибір кількості ознак для відбору
        k_features_label = QLabel("Кількість ознак (k):")
        controls_layout.addWidget(k_features_label, 3, 0)

        self.k_features_combo = QComboBox()
        self.k_features_combo.addItems(
            [str(i) for i in [5, 10, 15, 20, 25, 30, 40, 50]])
        self.k_features_combo.setCurrentText("20")
        controls_layout.addWidget(self.k_features_combo, 3, 1)

        # Вибір розміру тестового набору
        test_size_label = QLabel("Розмір тестового набору:")
        controls_layout.addWidget(test_size_label, 4, 0)

        self.test_size_combo = QComboBox()
        self.test_size_combo.addItems(["10%", "20%", "30%", "40%"])
        self.test_size_combo.setCurrentText("20%")
        controls_layout.addWidget(self.test_size_combo, 4, 1)

        # Кнопки
        buttons_layout = QHBoxLayout()
        right_layout.addLayout(buttons_layout)

        self.train_button = QPushButton("Навчити модель")
        self.train_button.clicked.connect(self.train_model)
        buttons_layout.addWidget(self.train_button)

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

        # Початкова візуалізація
        self.visualize_data()

    def load_data(self):
        # Завантаження датасету Spambase
        spambase = fetch_ucirepo(id=94)

        # Отримання даних
        self.X = spambase.data.features
        self.y = spambase.data.targets.values.ravel()  # Перетворення у 1D масив

        # Збереження метаданих
        self.metadata = spambase.metadata
        self.variables = spambase.variables

        # Створення DataFrame для зручності
        self.df = pd.concat([self.X, pd.DataFrame(
            self.y, columns=['is_spam'])], axis=1)

    def preprocess_data(self):
        # Отримання параметрів з інтерфейсу
        preprocessing_method = self.preprocessing_combo.currentText()
        use_feature_selection = self.feature_selection_check.isChecked()
        k_features = int(self.k_features_combo.currentText())
        test_size = float(
            self.test_size_combo.currentText().replace("%", "")) / 100

        # Копіювання даних
        X = self.X.copy()

        # Попередня обробка даних
        if preprocessing_method == "StandardScaler":
            scaler = StandardScaler()
            X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
        elif preprocessing_method == "MinMaxScaler":
            scaler = MinMaxScaler()
            X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
        elif preprocessing_method == "TF-IDF":
            # TF-IDF трансформація для спам-фільтрації
            # Спамбаза вже містить частотні характеристики, тому застосовуємо TF-IDF трансформацію
            tfidf = TfidfTransformer()
            X = pd.DataFrame(tfidf.fit_transform(
                X).toarray(), columns=X.columns)

        # Відбір ознак
        if use_feature_selection:
            selector = SelectKBest(chi2, k=k_features)

            # Для chi2 всі значення мають бути невід'ємними
            if X.min().min() < 0:
                X_min = X.min().min()
                X = X - X_min if X_min < 0 else X

            X_new = selector.fit_transform(X, self.y)
            selected_features = X.columns[selector.get_support()]
            X = pd.DataFrame(X_new, columns=selected_features)

            # Збереження важливості ознак
            self.feature_importances = pd.DataFrame({
                'feature': X.columns,
                'importance': selector.scores_[selector.get_support()]
            }).sort_values('importance', ascending=False)

        # Розбиття на навчальний та тестовий набори
        X_train, X_test, y_train, y_test = train_test_split(
            X, self.y, test_size=test_size, random_state=42, stratify=self.y
        )

        return X, X_train, X_test, y_train, y_test

    def train_model(self):
        # Попередня обробка даних
        X, X_train, X_test, y_train, y_test = self.preprocess_data()

        # Вибір типу класифікатора
        nb_type = self.nb_type_combo.currentText()

        if nb_type == "Gaussian NB":
            clf = GaussianNB()
        elif nb_type == "Multinomial NB":
            # Для MultinomialNB всі значення мають бути невід'ємними
            if X_train.min().min() < 0:
                X_min = X_train.min().min()
                X_train = X_train - X_min if X_min < 0 else X_train
                X_test = X_test - X_min if X_min < 0 else X_test
            clf = MultinomialNB()
        else:  # Bernoulli NB
            clf = BernoulliNB()

        # Навчання моделі
        clf.fit(X_train, y_train)

        # Прогнозування
        y_pred = clf.predict(X_test)
        y_prob = clf.predict_proba(X_test)[:, 1] if hasattr(
            clf, "predict_proba") else None

        # Оцінка продуктивності
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        # Виведення результатів
        preprocessing_method = self.preprocessing_combo.currentText()
        use_feature_selection = self.feature_selection_check.isChecked()
        k_features = self.k_features_combo.currentText() if use_feature_selection else "Всі"

        results = f"Результати для {nb_type}:\n\n"
        results += f"Попередня обробка: {preprocessing_method}\n"
        results += f"Відбір ознак: {'Так' if use_feature_selection else 'Ні'}\n"
        results += f"Кількість ознак: {k_features}\n\n"
        results += f"Accuracy: {accuracy:.4f}\n"
        results += f"Precision: {precision:.4f}\n"
        results += f"Recall: {recall:.4f}\n"
        results += f"F1-score: {f1:.4f}\n\n"

        # Додавання звіту про класифікацію
        report = classification_report(
            y_test, y_pred, target_names=["Не спам", "Спам"])
        results += "Детальний звіт про класифікацію:\n" + report

        self.results_text.setText(results)

        # Візуалізація матриці помилок
        self.results_figure.clear()
        ax = self.results_figure.add_subplot(111)
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["Не спам", "Спам"],
                    yticklabels=["Не спам", "Спам"], ax=ax)
        ax.set_xlabel('Прогнозовані класи')
        ax.set_ylabel('Фактичні класи')
        ax.set_title('Матриця помилок')
        self.results_canvas.draw()

        # Візуалізація ROC-кривої, якщо доступні ймовірності
        if y_prob is not None:
            self.roc_figure.clear()
            ax = self.roc_figure.add_subplot(111)

            fpr, tpr, _ = roc_curve(y_test, y_prob)
            roc_auc = auc(fpr, tpr)

            ax.plot(fpr, tpr, color='darkorange', lw=2,
                    label=f'ROC крива (AUC = {roc_auc:.2f})')
            ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title('ROC крива')
            ax.legend(loc="lower right")
            ax.grid(True)

            self.roc_canvas.draw()

            # Перехід на вкладку з результатами
            self.tabs.setCurrentIndex(1)

    def visualize_data(self):
        # Очищення попередніх графіків
        self.data_figure.clear()

        # Створення підграфіків
        gs = self.data_figure.add_gridspec(2, 2)
        ax1 = self.data_figure.add_subplot(gs[0, 0])
        ax2 = self.data_figure.add_subplot(gs[0, 1])
        ax3 = self.data_figure.add_subplot(gs[1, :])

        # 1. Розподіл класів
        class_counts = self.df['is_spam'].value_counts()
        ax1.bar(['Не спам', 'Спам'], [
                class_counts.get(0, 0), class_counts.get(1, 0)])
        ax1.set_title('Розподіл класів')
        ax1.set_ylabel('Кількість')

        # 2. Кореляція між ознаками та цільовою змінною
        correlations = self.X.corrwith(self.df['is_spam'])
        top_corr = correlations.abs().sort_values(ascending=False).head(10)

        colors = ['red' if c > 0 else 'blue' for c in top_corr]
        ax2.barh(top_corr.index, top_corr.values, color=colors)
        ax2.set_title('Топ-10 ознак за кореляцією')
        ax2.set_xlabel('Кореляція зі спамом')

        # 3. Розподіл деяких важливих ознак
        top_features = top_corr.index[:5]
        for feature in top_features:
            sns.kdeplot(data=self.df, x=feature, hue='is_spam',
                        ax=ax3, common_norm=False)

        ax3.set_title('Розподіл важливих ознак за класами')
        ax3.set_xlabel('Значення ознаки')
        ax3.set_ylabel('Щільність')
        ax3.legend(['Не спам', 'Спам'])

        # Налаштування макету
        self.data_figure.tight_layout()
        self.data_canvas.draw()

        # Перехід на вкладку з даними
        self.tabs.setCurrentIndex(0)

    def go_back(self):
        self.hide()

        if self.home_window:
            self.home_window.show()

    def closeEvent(self, event):
        if self.home_window:
            self.home_window.show()
        event.accept()
