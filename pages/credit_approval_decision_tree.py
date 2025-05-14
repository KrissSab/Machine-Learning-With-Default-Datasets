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
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import numpy as np
import pandas as pd
import seaborn as sns
import os
import io
from contextlib import redirect_stdout


class CreditApprovalDecisionTree(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Credit Approval Decision Tree")
        self.setMinimumSize(1280, 720)

        # Центральний віджет
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Головний layout
        main_layout = QVBoxLayout(central_widget)

        # Заголовок
        title_label = QLabel(
            "Credit Approval Decision Tree")
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

        # Вкладка для важливості ознак
        self.importance_tab = QWidget()
        self.tabs.addTab(self.importance_tab, "Важливість ознак")
        importance_layout = QVBoxLayout(self.importance_tab)

        self.importance_figure = plt.figure(figsize=(10, 8))
        self.importance_canvas = FigureCanvas(self.importance_figure)
        importance_layout.addWidget(self.importance_canvas)

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

        # Вибір цільової змінної
        target_label = QLabel("Цільова змінна:")
        controls_layout.addWidget(target_label, 6, 0)

        self.target_combo = QComboBox()
        self.target_combo.addItems(["FLAG_OWN_CAR", "FLAG_OWN_REALTY",
                                   "FLAG_MOBIL", "FLAG_WORK_PHONE", "FLAG_PHONE", "FLAG_EMAIL"])
        self.target_combo.setCurrentText("FLAG_OWN_CAR")
        controls_layout.addWidget(self.target_combo, 6, 1)

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

                # Оновлення комбобоксу з цільовими змінними
                binary_columns = []
                for col in self.data.columns:
                    if self.data[col].nunique() == 2:
                        binary_columns.append(col)

                if binary_columns:
                    self.target_combo.clear()
                    self.target_combo.addItems(binary_columns)
                    self.target_combo.setCurrentText(binary_columns[0])

                # Активація кнопок
                self.train_button.setEnabled(True)
                self.grid_search_button.setEnabled(True)
                self.visualize_button.setEnabled(True)

                # Оновлення статусу
                self.data_status_label.setText(f"Статус: Дані завантажено успішно. "
                                               f"Кількість зразків: {self.data.shape[0]}, "
                                               f"Кількість ознак: {self.data.shape[1]-1}")

                # Початкова візуалізація
                self.visualize_data()

            except Exception as e:
                self.data_status_label.setText(
                    f"Статус: Помилка при завантаженні даних: {str(e)}")

    def preprocess_data(self):
        if self.data is None:
            return None, None, None, None

        # Копіювання даних
        df = self.data.copy()

        # Вибір цільової змінної
        target_column = self.target_combo.currentText()

        # Перевірка, чи цільова змінна є бінарною
        if df[target_column].nunique() != 2:
            self.results_text.setText(
                f"Помилка: Цільова змінна '{target_column}' не є бінарною.")
            return None, None, None, None

        # Перетворення цільової змінної у числовий формат, якщо потрібно
        if df[target_column].dtype == 'object':
            label_encoder = LabelEncoder()
            df[target_column] = label_encoder.fit_transform(df[target_column])

        # Інженерія ознак (якщо увімкнено)
        if self.feature_engineering_check.isChecked():
            # Створення нових ознак

            # Вік у роках (DAYS_BIRTH зазвичай від'ємний, тому ділимо на -365)
            if 'DAYS_BIRTH' in df.columns:
                df['AGE_YEARS'] = df['DAYS_BIRTH'] / -365.25

            # Стаж роботи у роках
            if 'DAYS_EMPLOYED' in df.columns:
                df['EMPLOYMENT_YEARS'] = df['DAYS_EMPLOYED'] / 365.25
                # Виправлення аномальних значень (якщо є)
                df['EMPLOYMENT_YEARS'] = df['EMPLOYMENT_YEARS'].apply(
                    lambda x: 0 if x < 0 else x)

            # Співвідношення доходу до кількості членів сім'ї
            if 'AMT_INCOME_TOTAL' in df.columns and 'CNT_FAM_MEMBERS' in df.columns:
                df['INCOME_PER_PERSON'] = df['AMT_INCOME_TOTAL'] / \
                    df['CNT_FAM_MEMBERS'].apply(lambda x: 1 if x == 0 else x)

            # Категорія доходу
            if 'AMT_INCOME_TOTAL' in df.columns:
                df['INCOME_CATEGORY'] = pd.cut(
                    df['AMT_INCOME_TOTAL'],
                    bins=[0, 50000, 100000, 200000, 500000, float('inf')],
                    labels=['Very Low', 'Low', 'Medium', 'High', 'Very High']
                )

        # Видалення ID та інших непотрібних стовпців
        if 'ID' in df.columns:
            df = df.drop('ID', axis=1)

        # Розділення на ознаки та цільову змінну
        X = df.drop(target_column, axis=1)
        y = df[target_column]

        # Визначення числових та категоріальних ознак
        numeric_features = X.select_dtypes(
            include=['int64', 'float64']).columns.tolist()
        categorical_features = X.select_dtypes(
            include=['object', 'category']).columns.tolist()

        # Збереження назв ознак для подальшого використання
        self.feature_names = numeric_features + categorical_features

        # Розбиття на навчальний та тестовий набори
        test_size = float(
            self.test_size_combo.currentText().replace("%", "")) / 100
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )

        return X_train, X_test, y_train, y_test, numeric_features, categorical_features

    def build_pipeline(self, numeric_features, categorical_features):
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
        if self.data is None:
            self.results_text.setText("Помилка: Дані не завантажено.")
            return

        try:
            # Попередня обробка даних
            X_train, X_test, y_train, y_test, numeric_features, categorical_features = self.preprocess_data()

            if X_train is None:
                return

            # Збереження для подальшого використання
            self.X_train = X_train
            self.X_test = X_test
            self.y_train = y_train
            self.y_test = y_test

            # Створення та навчання моделі
            self.model = self.build_pipeline(
                numeric_features, categorical_features)
            self.model.fit(X_train, y_train)

            # Прогнозування
            y_pred = self.model.predict(X_test)

            # Оцінка продуктивності
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)

            # Виведення результатів
            max_depth = self.max_depth_spin.value()
            min_samples_split = self.min_samples_split_spin.value()
            min_samples_leaf = self.min_samples_leaf_spin.value()
            criterion = self.criterion_combo.currentText()
            target_column = self.target_combo.currentText()

            results = f"Результати для дерева рішень (цільова змінна: {target_column}):\n\n"
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
                y_test, y_pred, target_names=["Клас 0", "Клас 1"])
            results += "Детальний звіт про класифікацію:\n" + report

            # Додавання текстового представлення дерева
            tree_classifier = self.model.named_steps['classifier']

            # Спроба отримати назви ознак після препроцесингу
            try:
                preprocessor = self.model.named_steps['preprocessor']
                feature_names_out = []

                for name, trans, cols in preprocessor.transformers_:
                    if name != 'remainder':
                        if hasattr(trans, 'get_feature_names_out'):
                            feature_names_out.extend(
                                trans.get_feature_names_out(cols))
                        else:
                            # Для старіших версій scikit-learn
                            if hasattr(trans.named_steps.get('onehot', None), 'get_feature_names_out'):
                                feature_names_out.extend(
                                    trans.named_steps['onehot'].get_feature_names_out(cols))
                            else:
                                feature_names_out.extend(
                                    [f"{name}_{col}" for col in cols])
            except:
                # Якщо не вдалося отримати назви ознак, використовуємо загальні назви
                feature_names_out = [f"feature_{i}" for i in range(
                    tree_classifier.n_features_in_)]

            # Отримання текстового представлення дерева
            tree_text = export_text(
                tree_classifier, feature_names=feature_names_out)
            results += "\nТекстове представлення дерева рішень:\n" + tree_text

            self.results_text.setText(results)

            # Візуалізація матриці помилок
            self.results_figure.clear()
            ax = self.results_figure.add_subplot(111)
            cm = confusion_matrix(y_test, y_pred)
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                        xticklabels=["Клас 0", "Клас 1"],
                        yticklabels=["Клас 0", "Клас 1"], ax=ax)
            ax.set_xlabel('Прогнозовані класи')
            ax.set_ylabel('Фактичні класи')
            ax.set_title('Матриця помилок')
            self.results_canvas.draw()

            # Візуалізація дерева рішень
            self.visualize_tree()

            # Візуалізація важливості ознак
            self.visualize_feature_importance()

            # Перехід на вкладку з результатами
            self.tabs.setCurrentIndex(2)

        except Exception as e:
            self.results_text.setText(f"Помилка при навчанні моделі: {str(e)}")

    def perform_grid_search(self):
        if self.data is None:
            self.results_text.setText("Помилка: Дані не завантажено.")
            return

        try:
            # Попередня обробка даних
            X_train, X_test, y_train, y_test, numeric_features, categorical_features = self.preprocess_data()

            if X_train is None:
                return

            # Створення базового пайплайну
            pipeline = self.build_pipeline(
                numeric_features, categorical_features)

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
            target_column = self.target_combo.currentText()
            results = f"Результати Grid Search (цільова змінна: {target_column}):\n\n"
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

            # Спроба отримати назви ознак після препроцесингу
            try:
                preprocessor = self.model.named_steps['preprocessor']
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
                class_names=["Клас 0", "Клас 1"],
                filled=True,
                rounded=True,
                ax=ax
            )

            target_column = self.target_combo.currentText()
            ax.set_title(
                f"Дерево рішень для {target_column} (глибина={tree_classifier.max_depth})")
            self.tree_canvas.draw()

            # Перехід на вкладку з деревом
            self.tabs.setCurrentIndex(1)

        except Exception as e:
            print(f"Помилка при візуалізації дерева: {str(e)}")

    def visualize_feature_importance(self):
        if self.model is None:
            return

        try:
            # Отримання дерева рішень з пайплайну
            tree_classifier = self.model.named_steps['classifier']

            # Спроба отримати назви ознак після препроцесингу
            try:
                preprocessor = self.model.named_steps['preprocessor']
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

            # Отримання важливості ознак
            importances = tree_classifier.feature_importances_

            # Створення DataFrame для сортування
            feature_importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False)

            # Обмеження до топ-20 ознак для кращої візуалізації
            feature_importance_df = feature_importance_df.head(20)

            # Очищення попередніх графіків
            self.importance_figure.clear()
            ax = self.importance_figure.add_subplot(111)

            # Візуалізація важливості ознак
            sns.barplot(x='importance', y='feature',
                        data=feature_importance_df, ax=ax)

            target_column = self.target_combo.currentText()
            ax.set_title(f"Важливість ознак для прогнозування {target_column}")
            ax.set_xlabel('Важливість')
            ax.set_ylabel('Ознака')

            self.importance_canvas.draw()

            # Перехід на вкладку з важливістю ознак
            self.tabs.setCurrentIndex(3)

        except Exception as e:
            print(f"Помилка при візуалізації важливості ознак: {str(e)}")

    def visualize_data(self):
        if self.data is None:
            return

        # Вибір цільової змінної
        target_column = self.target_combo.currentText()

        # Очищення попередніх графіків
        self.data_figure.clear()

        # Створення підграфіків
        gs = self.data_figure.add_gridspec(2, 2)
        ax1 = self.data_figure.add_subplot(gs[0, 0])
        ax2 = self.data_figure.add_subplot(gs[0, 1])
        ax3 = self.data_figure.add_subplot(gs[1, 0])
        ax4 = self.data_figure.add_subplot(gs[1, 1])

        # 1. Розподіл цільової змінної
        target_counts = self.data[target_column].value_counts()
        ax1.bar(['Клас 0', 'Клас 1'], [
                target_counts.get(0, 0), target_counts.get(1, 0)])
        ax1.set_title(f'Розподіл {target_column}')
        ax1.set_ylabel('Кількість')

        # 2. Розподіл доходу за цільовою змінною
        if 'AMT_INCOME_TOTAL' in self.data.columns:
            sns.boxplot(x=target_column, y='AMT_INCOME_TOTAL',
                        data=self.data, ax=ax2)
            ax2.set_title(f'Розподіл доходу за {target_column}')
            ax2.set_xlabel(target_column)
            ax2.set_ylabel('Дохід')
            ax2.set_xticklabels(['Клас 0', 'Клас 1'])

        # 3. Розподіл віку за цільовою змінною
        if 'DAYS_BIRTH' in self.data.columns:
            # Перетворення днів у роки
            age_years = self.data['DAYS_BIRTH'] / -365.25
            temp_df = self.data.copy()
            temp_df['AGE_YEARS'] = age_years

            sns.boxplot(x=target_column, y='AGE_YEARS', data=temp_df, ax=ax3)
            ax3.set_title(f'Розподіл віку за {target_column}')
            ax3.set_xlabel(target_column)
            ax3.set_ylabel('Вік (роки)')
            ax3.set_xticklabels(['Клас 0', 'Клас 1'])

        # 4. Розподіл за статтю та цільовою змінною
        if 'CODE_GENDER' in self.data.columns:
            # Створення крос-таблиці
            gender_target = pd.crosstab(
                self.data['CODE_GENDER'], self.data[target_column])
            gender_target.plot(kind='bar', stacked=True, ax=ax4)
            ax4.set_title(f'Розподіл за статтю та {target_column}')
            ax4.set_xlabel('Стать')
            ax4.set_ylabel('Кількість')
            ax4.legend(['Клас 0', 'Клас 1'])

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
