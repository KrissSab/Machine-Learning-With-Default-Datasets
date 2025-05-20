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
    QScrollArea
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import numpy as np
import pandas as pd
import seaborn as sns
import io

from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from tensorflow.keras.datasets import fashion_mnist  # type: ignore


class FashionDecisionTree(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Fashion DT")
        self.setMinimumSize(1280, 720)

        self.class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                            'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QVBoxLayout(central_widget)

        title_label = QLabel("Fashion Decision Tree Classification")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_label.setStyleSheet("font-size: 24px; margin-bottom: 20px;")
        main_layout.addWidget(title_label)

        # Create the content layout (horizontal split)
        content_layout = QHBoxLayout()
        main_layout.addLayout(content_layout)

        # Left panel for visualization
        self.figure = plt.figure(figsize=(10, 8))
        self.canvas = FigureCanvas(self.figure)
        content_layout.addWidget(self.canvas, 2)

        # Right panel for controls and results
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        content_layout.addWidget(right_panel, 1)

        # Controls group
        controls_group = QWidget()
        controls_layout = QGridLayout(controls_group)
        right_layout.addWidget(controls_group)

        # Max depth control
        depth_label = QLabel("Max Depth:")
        controls_layout.addWidget(depth_label, 0, 0)

        self.depth_spinbox = QSpinBox()
        self.depth_spinbox.setMinimum(1)
        self.depth_spinbox.setMaximum(20)
        # Changed to a smaller value to make tree more visible
        self.depth_spinbox.setValue(5)
        controls_layout.addWidget(self.depth_spinbox, 0, 1)

        # Min samples split control
        min_samples_label = QLabel("Min Samples Split:")
        controls_layout.addWidget(min_samples_label, 1, 0)

        self.min_samples_spinbox = QSpinBox()
        self.min_samples_spinbox.setMinimum(2)
        self.min_samples_spinbox.setMaximum(10)
        self.min_samples_spinbox.setValue(2)
        controls_layout.addWidget(self.min_samples_spinbox, 1, 1)

        # Sample size control
        sample_size_label = QLabel("Sample Size:")
        controls_layout.addWidget(sample_size_label, 2, 0)

        self.sample_size_spinbox = QSpinBox()
        self.sample_size_spinbox.setMinimum(1000)
        self.sample_size_spinbox.setMaximum(20000)
        self.sample_size_spinbox.setSingleStep(1000)
        self.sample_size_spinbox.setValue(10000)
        controls_layout.addWidget(self.sample_size_spinbox, 2, 1)

        # Tree display max features
        tree_features_label = QLabel("Max Features to Display:")
        controls_layout.addWidget(tree_features_label, 3, 0)

        self.tree_features_spinbox = QSpinBox()
        self.tree_features_spinbox.setMinimum(1)
        self.tree_features_spinbox.setMaximum(50)
        self.tree_features_spinbox.setValue(10)
        controls_layout.addWidget(self.tree_features_spinbox, 3, 1)

        # Buttons layout
        buttons_layout = QHBoxLayout()
        right_layout.addLayout(buttons_layout)

        self.load_data_button = QPushButton("Load Data")
        self.load_data_button.clicked.connect(self.load_data)
        buttons_layout.addWidget(self.load_data_button)

        self.train_button = QPushButton("Train Model")
        self.train_button.clicked.connect(self.train_model)
        self.train_button.setEnabled(False)
        buttons_layout.addWidget(self.train_button)

        self.optimize_button = QPushButton("Optimize Params")
        self.optimize_button.clicked.connect(self.optimize_params)
        self.optimize_button.setEnabled(False)
        buttons_layout.addWidget(self.optimize_button)

        # Visualization button row
        viz_buttons_layout = QHBoxLayout()
        right_layout.addLayout(viz_buttons_layout)

        self.show_confusion_button = QPushButton("Show Confusion Matrix")
        self.show_confusion_button.clicked.connect(lambda: self.visualize_confusion_matrix(
            self.y_test, self.last_predictions) if hasattr(self, 'last_predictions') else None)
        self.show_confusion_button.setEnabled(False)
        viz_buttons_layout.addWidget(self.show_confusion_button)

        self.show_tree_button = QPushButton("Show Decision Tree")
        self.show_tree_button.clicked.connect(self.visualize_tree)
        self.show_tree_button.setEnabled(False)
        viz_buttons_layout.addWidget(self.show_tree_button)

        # Results text area
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        right_layout.addWidget(self.results_text)

        # Back button
        back_button = QPushButton("Return to main")
        back_button.clicked.connect(self.go_back)
        main_layout.addWidget(back_button)

        self.home_window = None
        self.data_loaded = False
        self.best_model = None
        self.last_predictions = None

    def load_data(self):
        self.results_text.setText(
            "Loading Fashion MNIST dataset, please wait...")
        self.results_text.repaint()

        try:
            # Завантаження даних
            (self.X_train, self.y_train), (self.X_test,
                                           self.y_test) = fashion_mnist.load_data()

            # Перетворення 28x28 зображень на вектори
            self.X_train_flattened = self.X_train.reshape(
                self.X_train.shape[0], -1) / 255.0
            self.X_test_flattened = self.X_test.reshape(
                self.X_test.shape[0], -1) / 255.0

            # Розбиття та підвибірка
            self.X_train_sample, self.X_val, self.y_train_sample, self.y_val = train_test_split(
                self.X_train_flattened, self.y_train, test_size=0.2, random_state=42
            )

            self.data_loaded = True
            self.train_button.setEnabled(True)
            self.optimize_button.setEnabled(True)

            # Display some random images from the dataset
            self.visualize_samples()

            self.results_text.setText(
                f"Fashion MNIST dataset loaded successfully!\n"
                f"Training samples: {self.X_train.shape[0]}\n"
                f"Test samples: {self.X_test.shape[0]}\n"
                f"Validation samples: {self.X_val.shape[0]}\n"
                f"Image shape: {self.X_train.shape[1]}x{self.X_train.shape[2]}\n"
                f"Number of classes: {len(self.class_names)}\n\n"
                f"Classes: {', '.join(self.class_names)}"
            )

        except Exception as e:
            self.results_text.setText(f"Error loading data: {str(e)}")

    def visualize_samples(self):
        """Display random samples from each class"""
        self.figure.clear()
        fig, axes = plt.subplots(2, 5, figsize=(10, 4))
        fig.subplots_adjust(hspace=0.5)
        self.figure = fig

        for i, class_idx in enumerate(range(10)):
            row, col = i // 5, i % 5
            class_samples = self.X_train[self.y_train == class_idx]
            random_idx = np.random.randint(0, class_samples.shape[0])
            axes[row, col].imshow(class_samples[random_idx], cmap='gray')
            axes[row, col].set_title(self.class_names[class_idx])
            axes[row, col].axis('off')

        plt.tight_layout()
        self.canvas.figure = self.figure
        self.canvas.draw()

    def train_model(self):
        if not self.data_loaded:
            self.results_text.setText("Please load the data first!")
            return

        self.results_text.setText(
            "Training decision tree model, please wait...")
        self.results_text.repaint()

        try:
            # Get parameters from UI
            max_depth = self.depth_spinbox.value()
            min_samples_split = self.min_samples_spinbox.value()
            sample_size = self.sample_size_spinbox.value()

            # Use a subset of training data
            X_train_subset = self.X_train_sample[:sample_size]
            y_train_subset = self.y_train_sample[:sample_size]

            # Create and train the model
            self.model = DecisionTreeClassifier(
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                random_state=42
            )
            self.model.fit(X_train_subset, y_train_subset)

            # Make predictions
            y_val_pred = self.model.predict(self.X_val)
            y_test_pred = self.model.predict(self.X_test_flattened)
            self.last_predictions = y_test_pred

            # Calculate metrics
            val_accuracy = accuracy_score(self.y_val, y_val_pred)
            val_precision = precision_score(
                self.y_val, y_val_pred, average='weighted')
            val_recall = recall_score(
                self.y_val, y_val_pred, average='weighted')
            val_f1 = f1_score(self.y_val, y_val_pred, average='weighted')

            test_accuracy = accuracy_score(self.y_test, y_test_pred)
            test_precision = precision_score(
                self.y_test, y_test_pred, average='weighted')
            test_recall = recall_score(
                self.y_test, y_test_pred, average='weighted')
            test_f1 = f1_score(self.y_test, y_test_pred, average='weighted')

            # Display results
            results = (
                f"Decision Tree Results (max_depth={max_depth}, min_samples_split={min_samples_split}):\n\n"
                f"Validation Metrics:\n"
                f"Accuracy: {val_accuracy:.4f}\n"
                f"Precision: {val_precision:.4f}\n"
                f"Recall: {val_recall:.4f}\n"
                f"F1-score: {val_f1:.4f}\n\n"
                f"Test Metrics:\n"
                f"Accuracy: {test_accuracy:.4f}\n"
                f"Precision: {test_precision:.4f}\n"
                f"Recall: {test_recall:.4f}\n"
                f"F1-score: {test_f1:.4f}\n\n"
            )

            # Add classification report
            report = classification_report(
                self.y_test, y_test_pred, target_names=self.class_names)
            results += f"Classification Report:\n{report}"

            self.results_text.setText(results)

            # Visualize confusion matrix
            self.visualize_confusion_matrix(self.y_test, y_test_pred)

            # Enable tree visualization
            self.show_tree_button.setEnabled(True)
            self.show_confusion_button.setEnabled(True)

        except Exception as e:
            self.results_text.setText(f"Error training model: {str(e)}")

    def optimize_params(self):
        if not self.data_loaded:
            self.results_text.setText("Please load the data first!")
            return

        self.results_text.setText(
            "Running grid search for optimal parameters, please wait...")
        self.results_text.repaint()

        try:
            # Get sample size from UI
            sample_size = self.sample_size_spinbox.value()

            # Use a subset of training data
            X_train_subset = self.X_train_sample[:sample_size]
            y_train_subset = self.y_train_sample[:sample_size]

            # Set up grid search
            param_grid = {
                'max_depth': [5, 10, 15, 20],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }

            clf = GridSearchCV(
                DecisionTreeClassifier(random_state=42),
                param_grid,
                cv=3,
                n_jobs=-1,
                verbose=0
            )

            # Fit the grid search
            clf.fit(X_train_subset, y_train_subset)

            # Get best model and parameters
            self.best_model = clf.best_estimator_
            best_params = clf.best_params_

            # Update UI with best parameters
            if 'max_depth' in best_params:
                self.depth_spinbox.setValue(best_params['max_depth'])
            if 'min_samples_split' in best_params:
                self.min_samples_spinbox.setValue(
                    best_params['min_samples_split'])

            # Make predictions with best model
            y_val_pred = self.best_model.predict(self.X_val)
            y_test_pred = self.best_model.predict(self.X_test_flattened)
            self.last_predictions = y_test_pred

            # Calculate metrics
            val_accuracy = accuracy_score(self.y_val, y_val_pred)
            test_accuracy = accuracy_score(self.y_test, y_test_pred)

            # Display results
            results = (
                f"Grid Search Results:\n\n"
                f"Best parameters: {best_params}\n\n"
                f"Validation accuracy: {val_accuracy:.4f}\n"
                f"Test accuracy: {test_accuracy:.4f}\n\n"
                f"The UI has been updated with optimal parameters.\n"
                f"Click 'Train Model' to run the model with these parameters."
            )

            self.results_text.setText(results)

            # Visualize confusion matrix
            self.visualize_confusion_matrix(self.y_test, y_test_pred)

            # Enable tree visualization
            self.show_tree_button.setEnabled(True)
            self.show_confusion_button.setEnabled(True)

        except Exception as e:
            self.results_text.setText(f"Error optimizing parameters: {str(e)}")

    def visualize_confusion_matrix(self, y_true, y_pred):
        """Visualize confusion matrix"""
        self.figure.clear()
        fig = plt.figure(figsize=(10, 8))
        self.figure = fig

        conf_matrix = confusion_matrix(y_true, y_pred)
        ax = fig.add_subplot(111)
        sns.heatmap(
            conf_matrix,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            ax=ax
        )
        ax.set_xlabel("Прогноз")
        ax.set_ylabel("Істинне значення")
        ax.set_title("Матриця помилок Fashion MNIST")

        plt.tight_layout()
        self.canvas.figure = self.figure
        self.canvas.draw()

    def visualize_tree(self):
        """Visualize the decision tree structure"""
        if not hasattr(self, 'model') and not hasattr(self, 'best_model'):
            self.results_text.setText("Please train the model first!")
            return

        self.figure.clear()
        fig = plt.figure(figsize=(15, 10))
        self.figure = fig

        # Get the model to visualize
        tree_model = self.model if hasattr(self, 'model') else self.best_model

        # Get max features to display from UI
        max_features = self.tree_features_spinbox.value()

        # Get the total number of features from the model
        n_features = tree_model.n_features_in_

        # Create feature names for all features used by the model
        feature_names = [f"pixel_{i}" for i in range(n_features)]

        # Plot the tree with all leaves shown
        ax = fig.add_subplot(111)
        plot_tree(
            tree_model,
            max_depth=3,  # Limit depth for visualization
            feature_names=feature_names,
            class_names=self.class_names,
            filled=True,
            rounded=True,
            ax=ax,
            fontsize=8,
            proportion=True,
            precision=2,
            impurity=True,  # Show impurity values
            node_ids=True,   # Show node IDs
        )

        # Enable zoom functionality
        def zoom_factory(ax, base_scale=1.1):
            def zoom_fun(event):
                # Get the current x and y limits
                cur_xlim = ax.get_xlim()
                cur_ylim = ax.get_ylim()

                # Get event location
                xdata = event.xdata
                ydata = event.ydata
                if event.button == 'up':
                    # Deal with zoom in
                    scale_factor = 1 / base_scale
                elif event.button == 'down':
                    # Deal with zoom out
                    scale_factor = base_scale
                else:
                    # Deal with something that should never happen
                    scale_factor = 1

                # Set new limits
                new_width = (cur_xlim[1] - cur_xlim[0]) * scale_factor
                new_height = (cur_ylim[1] - cur_ylim[0]) * scale_factor

                ax.set_xlim([xdata - new_width * (xdata - cur_xlim[0]) / (cur_xlim[1] - cur_xlim[0]),
                             xdata + new_width * (cur_xlim[1] - xdata) / (cur_xlim[1] - cur_xlim[0])])
                ax.set_ylim([ydata - new_height * (ydata - cur_ylim[0]) / (cur_ylim[1] - cur_ylim[0]),
                             ydata + new_height * (cur_ylim[1] - ydata) / (cur_ylim[1] - cur_ylim[0])])

                # Force redraw
                self.canvas.draw()

            # Attach the function to the figure
            fig.canvas.mpl_connect('scroll_event', zoom_fun)

            return zoom_fun

        # Connect the zoom function
        zoom = zoom_factory(ax)

        ax.set_title("Decision Tree Structure (Use mouse wheel to zoom)")
        plt.tight_layout()
        self.canvas.figure = self.figure
        self.canvas.draw()

        # Show explanation in results text
        self.results_text.setText(
            "Decision Tree Visualization:\n\n"
            "This is a graphical representation of the decision tree model. Each node represents a decision based on a pixel value.\n\n"
            "- The color intensity represents the proportion of samples per class\n"
            "- Each decision node shows: the decision rule, Gini impurity, number of samples, and class distribution\n"
            "- Use mouse wheel to zoom in and out of the tree\n"
            "- Only the first 3 levels are shown for clarity\n\n"
            "To view the confusion matrix again, click 'Show Confusion Matrix'."
        )

    def go_back(self):
        self.hide()

        if self.home_window:
            self.home_window.show()

    def closeEvent(self, event):
        if self.home_window:
            self.home_window.show()
        event.accept()
