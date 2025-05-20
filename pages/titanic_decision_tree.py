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
    QSpinBox,
)
from PyQt6.QtCore import Qt
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import numpy as np
import pandas as pd
import seaborn as sns
import os
import io
from contextlib import redirect_stdout


class TitanicDecisionTree(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(
            "Titanic Decision Tree")
        self.setMinimumSize(1280, 720)

        # Центральний віджет
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Головний layout
        main_layout = QVBoxLayout(central_widget)

        # Заголовок
        title_label = QLabel(
            "Titanic Decision Tree")
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

        # Вкладка для візуалізації дерева рішень
        self.tree_tab = QWidget()
        self.tabs.addTab(self.tree_tab, "Дерево рішень")
        tree_layout = QVBoxLayout(self.tree_tab)

        self.tree_figure = plt.figure(figsize=(12, 10))
        self.tree_canvas = FigureCanvas(self.tree_figure)
        tree_layout.addWidget(self.tree_canvas)

        # Вкладка для результатів моделі
        self.results_tab = QWidget()
        self.tabs.addTab(self.results_tab, "Результати")
        results_layout = QVBoxLayout(self.results_tab)

        self.results_figure = plt.figure(figsize=(10, 8))
        self.results_canvas = FigureCanvas(self.results_figure)
        results_layout.addWidget(self.results_canvas)

        # Права панель для контролів та результатів
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        content_layout.addWidget(right_panel, 1)  # Співвідношення 2:1

        # Контроли для дерева рішень
        controls_group = QWidget()
        controls_layout = QGridLayout(controls_group)
        right_layout.addWidget(controls_group)

        # Максимальна глибина дерева
        max_depth_label = QLabel("Максимальна глибина:")
        controls_layout.addWidget(max_depth_label, 0, 0)

        self.max_depth_spin = QSpinBox()
        self.max_depth_spin.setMinimum(1)
        self.max_depth_spin.setMaximum(20)
        self.max_depth_spin.setValue(5)
        controls_layout.addWidget(self.max_depth_spin, 0, 1)

        # Мінімальна кількість зразків для розбиття
        min_samples_split_label = QLabel("Мін. зразків для розбиття:")
        controls_layout.addWidget(min_samples_split_label, 1, 0)

        self.min_samples_split_spin = QSpinBox()
        self.min_samples_split_spin.setMinimum(2)
        self.min_samples_split_spin.setMaximum(20)
        self.min_samples_split_spin.setValue(2)
        controls_layout.addWidget(self.min_samples_split_spin, 1, 1)

        # Мінімальна кількість зразків у листі
        min_samples_leaf_label = QLabel("Мін. зразків у листі:")
        controls_layout.addWidget(min_samples_leaf_label, 2, 0)

        self.min_samples_leaf_spin = QSpinBox()
        self.min_samples_leaf_spin.setMinimum(1)
        self.min_samples_leaf_spin.setMaximum(20)
        self.min_samples_leaf_spin.setValue(1)
        controls_layout.addWidget(self.min_samples_leaf_spin, 2, 1)

        # Критерій розбиття
        criterion_label = QLabel("Критерій розбиття:")
        controls_layout.addWidget(criterion_label, 3, 0)

        self.criterion_combo = QComboBox()
        self.criterion_combo.addItems(["gini", "entropy"])
        controls_layout.addWidget(self.criterion_combo, 3, 1)

        # Розмір тестового набору
        test_size_label = QLabel("Розмір тестового набору:")
        controls_layout.addWidget(test_size_label, 4, 0)

        self.test_size_combo = QComboBox()
        self.test_size_combo.addItems(["10%", "20%", "30%", "40%"])
        self.test_size_combo.setCurrentText("20%")
        controls_layout.addWidget(self.test_size_combo, 4, 1)

        # Опції обробки даних
        data_options_label = QLabel("Опції обробки даних:")
        controls_layout.addWidget(data_options_label, 5, 0)

        self.feature_engineering_check = QCheckBox("Створення нових ознак")
        self.feature_engineering_check.setChecked(True)
        controls_layout.addWidget(self.feature_engineering_check, 5, 1)

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
        self.train_data = None
        self.test_data = None
        self.gender_submission = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.model = None

        # Деактивація кнопок до завантаження даних
        self.train_button.setEnabled(False)
        self.grid_search_button.setEnabled(False)
        self.visualize_button.setEnabled(False)

    def load_data(self):
        # Відкриття діалогу вибору директорії
        directory = QFileDialog.getExistingDirectory(
            self, "Виберіть директорію з даними")

        if directory:
            try:
                # Завантаження файлів
                train_path = os.path.join(directory, "train.csv")
                test_path = os.path.join(directory, "test.csv")
                gender_submission_path = os.path.join(
                    directory, "gender_submission.csv")

                if not os.path.exists(train_path) or not os.path.exists(test_path) or not os.path.exists(gender_submission_path):
                    self.data_status_label.setText(
                        "Статус: Помилка! Не всі файли знайдено.")
                    return

                self.train_data = pd.read_csv(train_path)
                self.test_data = pd.read_csv(test_path)
                self.gender_submission = pd.read_csv(gender_submission_path)

                # Об'єднання тестових даних з цільовою змінною
                self.test_data = pd.merge(
                    self.test_data, self.gender_submission, on='PassengerId')

                # Активація кнопок
                self.train_button.setEnabled(True)
                self.grid_search_button.setEnabled(True)
                self.visualize_button.setEnabled(True)

                # Оновлення статусу
                self.data_status_label.setText(f"Статус: Дані завантажено успішно. "
                                               f"Тренувальний набір: {self.train_data.shape[0]} зразків, "
                                               f"Тестовий набір: {self.test_data.shape[0]} зразків.")

                # Початкова візуалізація
                self.visualize_data()

            except Exception as e:
                self.data_status_label.setText(
                    f"Статус: Помилка при завантаженні даних: {str(e)}")

    def preprocess_data(self, data, is_training=True):
        # Копіювання даних
        df = data.copy()

        # Інженерія ознак (якщо увімкнено)
        if self.feature_engineering_check.isChecked():
            # Створення нової ознаки: розмір сім'ї
            df['FamilySize'] = df['SibSp'] + df['Parch'] + 1

            # Створення ознаки: чи людина подорожує сама
            df['IsAlone'] = (df['FamilySize'] == 1).astype(int)

            # Витягнення титулу з імені
            df['Title'] = df['Name'].str.extract(
                ' ([A-Za-z]+)\.', expand=False)

            # Групування рідкісних титулів
            title_mapping = {
                "Mr": "Mr",
                "Miss": "Miss",
                "Mrs": "Mrs",
                "Master": "Master",
                "Dr": "Rare",
                "Rev": "Rare",
                "Col": "Rare",
                "Major": "Rare",
                "Mlle": "Miss",
                "Countess": "Rare",
                "Ms": "Miss",
                "Lady": "Rare",
                "Jonkheer": "Rare",
                "Don": "Rare",
                "Dona": "Rare",
                "Mme": "Mrs",
                "Capt": "Rare",
                "Sir": "Rare"
            }
            df['Title'] = df['Title'].map(title_mapping)
            df['Title'] = df['Title'].fillna("Rare")

            # Створення категорій віку
            df['AgeBand'] = pd.cut(df['Age'], 5)

            # Створення категорій плати за проїзд
            df['FareBand'] = pd.cut(df['Fare'], 4)

        # Вибір ознак для моделі
        features = ['Pclass', 'Sex', 'Age',
                    'SibSp', 'Parch', 'Fare', 'Embarked']

        if self.feature_engineering_check.isChecked():
            features.extend(['FamilySize', 'IsAlone', 'Title'])

        # Підготовка даних для моделі
        X = df[features].copy()

        if is_training:
            y = df['Survived']
            return X, y
        else:
            return X

    def build_pipeline(self):
        # Визначення числових та категоріальних ознак
        numeric_features = ['Age', 'SibSp', 'Parch', 'Fare']
        categorical_features = ['Pclass', 'Sex', 'Embarked']

        if self.feature_engineering_check.isChecked():
            numeric_features.extend(['FamilySize', 'IsAlone'])
            categorical_features.extend(['Title'])

        # Створення препроцесорів для різних типів ознак
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        # Об'єднання препроцесорів
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ])

        # Створення повного пайплайну з препроцесором та моделлю
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', DecisionTreeClassifier(
                max_depth=self.max_depth_spin.value(),
                min_samples_split=self.min_samples_split_spin.value(),
                min_samples_leaf=self.min_samples_leaf_spin.value(),
                criterion=self.criterion_combo.currentText(),
                random_state=42
            ))
        ])

        return pipeline

    def train_model(self):
        if self.train_data is None or self.test_data is None:
            self.results_text.setText("Помилка: Дані не завантажено.")
            return

        try:
            # Попередня обробка даних
            X_train, y_train = self.preprocess_data(
                self.train_data, is_training=True)
            X_test, y_test = self.preprocess_data(
                self.test_data, is_training=True)

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

            # Оцінка продуктивності
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)

            # Виведення результатів
            max_depth = self.max_depth_spin.value()
            min_samples_split = self.min_samples_split_spin.value()
            min_samples_leaf = self.min_samples_leaf_spin.value()
            criterion = self.criterion_combo.currentText()

            results = f"Результати для дерева рішень:\n\n"
            results += f"Параметри моделі:\n"
            results += f"- Максимальна глибина: {max_depth}\n"
            results += f"- Мін. зразків для розбиття: {min_samples_split}\n"
            results += f"- Мін. зразків у листі: {min_samples_leaf}\n"
            results += f"- Критерій розбиття: {criterion}\n\n"
            results += f"Метрики продуктивності:\n"
            results += f"- Accuracy: {accuracy:.4f}\n"
            results += f"- Precision: {precision:.4f}\n"
            results += f"- Recall: {recall:.4f}\n"
            results += f"- F1-score: {f1:.4f}\n\n"

            # Додавання звіту про класифікацію
            report = classification_report(
                y_test, y_pred, target_names=["Не вижив", "Вижив"])
            results += "Детальний звіт про класифікацію:\n" + report

            self.results_text.setText(results)

            # Візуалізація матриці помилок
            self.results_figure.clear()
            ax = self.results_figure.add_subplot(111)
            cm = confusion_matrix(y_test, y_pred)
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                        xticklabels=["Не вижив", "Вижив"],
                        yticklabels=["Не вижив", "Вижив"], ax=ax)
            ax.set_xlabel('Прогнозовані класи')
            ax.set_ylabel('Фактичні класи')
            ax.set_title('Матриця помилок')
            self.results_canvas.draw()

            # Візуалізація дерева рішень
            self.visualize_tree()

            # Перехід на вкладку з результатами
            self.tabs.setCurrentIndex(2)

        except Exception as e:
            self.results_text.setText(f"Помилка при навчанні моделі: {str(e)}")

    def perform_grid_search(self):
        if self.train_data is None or self.test_data is None:
            self.results_text.setText("Помилка: Дані не завантажено.")
            return

        try:
            # Попередня обробка даних
            X_train, y_train = self.preprocess_data(
                self.train_data, is_training=True)

            # Створення базового пайплайну
            pipeline = self.build_pipeline()

            # Параметри для пошуку
            param_grid = {
                'classifier__max_depth': [3, 5, 7, 9],
                'classifier__min_samples_split': [2, 5, 10],
                'classifier__min_samples_leaf': [1, 2, 4],
                'classifier__criterion': ['gini', 'entropy']
            }

            # Виконання пошуку по сітці
            grid_search = GridSearchCV(
                pipeline, param_grid, cv=5, scoring='accuracy')
            grid_search.fit(X_train, y_train)

            # Отримання найкращих параметрів
            best_params = grid_search.best_params_

            # Оновлення контролів з найкращими параметрами
            self.max_depth_spin.setValue(best_params['classifier__max_depth'])
            self.min_samples_split_spin.setValue(
                best_params['classifier__min_samples_split'])
            self.min_samples_leaf_spin.setValue(
                best_params['classifier__min_samples_leaf'])
            self.criterion_combo.setCurrentText(
                best_params['classifier__criterion'])

            # Виведення результатів
            results = f"Результати Grid Search:\n\n"
            results += f"Найкращі параметри:\n"
            results += f"- Максимальна глибина: {best_params['classifier__max_depth']}\n"
            results += f"- Мін. зразків для розбиття: {best_params['classifier__min_samples_split']}\n"
            results += f"- Мін. зразків у листі: {best_params['classifier__min_samples_leaf']}\n"
            results += f"- Критерій розбиття: {best_params['classifier__criterion']}\n\n"
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

    def visualize_tree(self):
        if self.model is None:
            return

        try:
            # Отримання дерева рішень з пайплайну
            tree_classifier = self.model.named_steps['classifier']

            # Отримання назв ознак після препроцесингу
            preprocessor = self.model.named_steps['preprocessor']

            # Спроба отримати назви ознак
            try:
                feature_names = []
                for name, trans, cols in preprocessor.transformers_:
                    if name != 'remainder':
                        if hasattr(trans, 'get_feature_names_out'):
                            feature_names.extend(
                                trans.get_feature_names_out(cols))
                        else:
                            # Для старіших версій scikit-learn
                            if hasattr(trans.named_steps.get('onehot', None), 'get_feature_names_out'):
                                feature_names.extend(
                                    trans.named_steps['onehot'].get_feature_names_out(cols))
                            else:
                                feature_names.extend(
                                    [f"{name}_{col}" for col in cols])
            except:
                # Якщо не вдалося отримати назви ознак, використовуємо загальні назви
                feature_names = [f"feature_{i}" for i in range(
                    tree_classifier.n_features_in_)]

            # Очищення попередніх графіків
            self.tree_figure.clear()
            ax = self.tree_figure.add_subplot(111)

            # Візуалізація дерева
            plot_tree(
                tree_classifier,
                feature_names=feature_names,
                class_names=["Не вижив", "Вижив"],
                filled=True,
                rounded=True,
                ax=ax
            )

            ax.set_title(
                f"Дерево рішень (глибина={tree_classifier.max_depth})")
            self.tree_canvas.draw()

            # Перехід на вкладку з деревом
            self.tabs.setCurrentIndex(1)

        except Exception as e:
            print(f"Помилка при візуалізації дерева: {str(e)}")

    def visualize_data(self):
        if self.train_data is None:
            return

        # Очищення попередніх графіків
        self.data_figure.clear()

        # Створення підграфіків
        gs = self.data_figure.add_gridspec(2, 2)
        ax1 = self.data_figure.add_subplot(gs[0, 0])
        ax2 = self.data_figure.add_subplot(gs[0, 1])
        ax3 = self.data_figure.add_subplot(gs[1, 0])
        ax4 = self.data_figure.add_subplot(gs[1, 1])

        # 1. Розподіл виживання
        survived_counts = self.train_data['Survived'].value_counts()
        ax1.bar(['Не вижив', 'Вижив'], [
                survived_counts.get(0, 0), survived_counts.get(1, 0)])
        ax1.set_title('Розподіл виживання')
        ax1.set_ylabel('Кількість')

        # 2. Виживання за класом
        sns.countplot(x='Pclass', hue='Survived', data=self.train_data, ax=ax2)
        ax2.set_title('Виживання за класом')
        ax2.set_xlabel('Клас')
        ax2.set_ylabel('Кількість')
        ax2.legend(['Не вижив', 'Вижив'])

        # 3. Виживання за статтю
        sns.countplot(x='Sex', hue='Survived', data=self.train_data, ax=ax3)
        ax3.set_title('Виживання за статтю')
        ax3.set_xlabel('Стать')
        ax3.set_ylabel('Кількість')
        ax3.legend(['Не вижив', 'Вижив'])

        # 4. Розподіл віку за виживанням
        sns.boxplot(x='Survived', y='Age', data=self.train_data, ax=ax4)
        ax4.set_title('Розподіл віку за виживанням')
        ax4.set_xlabel('Виживання')
        ax4.set_ylabel('Вік')
        ax4.set_xticklabels(['Не вижив', 'Вижив'])

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
