import os
import numpy as np

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from src.utils import get_paths, read_image


MODELS = {
    "QDA": QuadraticDiscriminantAnalysis(),
    "SVM": SVC(),
    "NB": GaussianNB(),
    "RF": RandomForestClassifier(),
}


class ClassicalML:
    def __init__(self, base_path: str, model_name: str):
        if model_name not in MODELS:
            raise ValueError(f"Model {model_name} is not supported.")
        self.base_path = base_path
        self.model_name = model_name
        self.model = MODELS[model_name]
        self._load_data()

    def _load_data(self):
        paths = get_paths(self.base_path)
        self.paths = paths

        X_train = []
        y_train = []
        X_test = []
        y_test = []
        X_val = []
        y_val = []

        def process_folder(
                folder_path: str, label: str, X_list: list, y_list: list
            ):
            """Helper function to process images from a folder"""
            for img_file in os.listdir(folder_path):
                img_path = os.path.join(folder_path, img_file)
                try:
                    img = read_image(img_path)
                    X_list.append(img)
                    # 0 for NORMAL, 1 for PNEUMONIA
                    y_list.append(0 if label == 'NORMAL' else 1)
                except Exception as e:
                    print(f"Error processing {img_path}: {str(e)}")
                    continue

        # Load training data
        for label, folder_path in paths['train'].items():
            process_folder(folder_path, label, X_train, y_train)

        # Load test data
        for label, folder_path in paths['test'].items():
            process_folder(folder_path, label, X_test, y_test)

        # Load validation data
        for label, folder_path in paths['val'].items():
            process_folder(folder_path, label, X_val, y_val)

        # Convert lists to numpy arrays for easier processing
        self.X_train = np.array(X_train)
        self.y_train = np.array(y_train)
        self.X_test = np.array(X_test)
        self.y_test = np.array(y_test)
        self.X_val = np.array(X_val)
        self.y_val = np.array(y_val)

    def train(self):
        """
        Train a classical ML model with cross-validation and parameter tuning.

        Parameters:
        -----------
        model_type : str ('svm' or 'random_forest')
            Type of model to train

        Returns:
        --------
        best_model : sklearn estimator
            The trained model with best parameters
        """
        # Flatten images for classical ML
        X_train_flat = np.array([img.flatten() for img in self.X_train])
        X_val_flat = np.array([img.flatten() for img in self.X_val])

        # Standardize features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_flat)
        X_val_scaled = scaler.transform(X_val_flat)

        if self.model_name == 'QDA':
            # Define pipeline and parameters for QDA
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('clf', QuadraticDiscriminantAnalysis())
            ])

            param_grid = {
                'clf__reg_param': [0.0, 0.1, 0.5, 1.0]
            }

        elif self.model_name == 'SVM':
            # Define pipeline and parameters for SVM
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('clf', SVC(random_state=42))
            ])

            param_grid = {
                'clf__C': [0.1, 1, 10],
                'clf__gamma': ['scale', 'auto', 0.01, 0.1],
                'clf__kernel': ['rbf', 'linear']
            }

        elif self.model_name == 'NB':
            # Define pipeline and parameters for Naive Bayes
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('clf', GaussianNB())
            ])

            param_grid = {
                'clf__C': [0.1, 1, 10],
                'clf__gamma': ['scale', 'auto', 0.01, 0.1],
                'clf__kernel': ['rbf', 'linear']
            }

        elif self.model_name == 'RF':
            # Define pipeline and parameters for Random Forest
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('clf', RandomForestClassifier(random_state=42))
            ])

            param_grid = {
                'clf__n_estimators': [50, 100, 200],
                'clf__max_depth': [None, 10, 20],
                'clf__min_samples_split': [2, 5, 10]
            }

        else:
            raise ValueError(
                "model_type must be either 'svm' or 'random_forest'"
            )

        # Setup cross-validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        # Grid search with cross-validation
        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            cv=cv,
            scoring='f1',
            n_jobs=-1,
            verbose=1
        )

        print(f"Training {self.model_name} model with cross-validation...")
        grid_search.fit(X_train_scaled, self.y_train)

        # Evaluate on validation set
        val_pred = grid_search.predict(X_val_scaled)
        val_accuracy = accuracy_score(self.y_val, val_pred)
        val_f1 = f1_score(self.y_val, val_pred)

        print("\nBest parameters found:")
        print(grid_search.best_params_)
        print(f"\nValidation Accuracy: {val_accuracy:.4f}")
        print(f"Validation F1 Score: {val_f1:.4f}")

        # Store the best model
        self.best_model = grid_search.best_estimator_
        self.scaler = scaler  # Store scaler for later use

        return self.best_model

    def predict(self):
        """
        Predict labels for input data using the trained model.

        Parameters:
        -----------
        X : np.ndarray
            Input data to predict labels for

        Returns:
        --------
        np.ndarray
            Predicted labels
        """
        if not hasattr(self, 'best_model'):
            raise RuntimeError("Model has not been trained yet.")
        
        # Flatten and scale the input data
        X_flat = np.array([img.flatten() for img in self.X_test])
        X_scaled = self.scaler.transform(X_flat)

        predictions = self.best_model.predict(X_scaled)

        confusion_matrix = confusion_matrix(self.y_test, predictions)
        disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix,
                                      display_labels=['NORMAL', 'PNEUMONIA'])
        disp.plot(cmap='Blues')
        return predictions, confusion_matrix