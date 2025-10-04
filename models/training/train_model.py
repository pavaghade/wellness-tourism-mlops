import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import mlflow
import mlflow.sklearn
from datasets import load_dataset
import joblib
import os

class ModelTrainer:
    def __init__(self):
        self.models = {
            'RandomForest': RandomForestClassifier(random_state=42),
            'GradientBoosting': GradientBoostingClassifier(random_state=42),
            'AdaBoost': AdaBoostClassifier(random_state=42),
            'DecisionTree': DecisionTreeClassifier(random_state=42)
        }

        self.param_grids = {
            'RandomForest': {
                'n_estimators': [100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5]
            },
            'GradientBoosting': {
                'n_estimators': [100, 200],
                'learning_rate': [0.1, 0.01],
                'max_depth': [3, 5]
            },
            'AdaBoost': {
                'n_estimators': [50, 100],
                'learning_rate': [0.1, 1.0]
            },
            'DecisionTree': {
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10]
            }
        }

    def load_data(self):
        """Load data from Hugging Face"""
        train_dataset = load_dataset("wellness-tourism-train", split='train')
        test_dataset = load_dataset("wellness-tourism-test", split='train')

        train_df = train_dataset.to_pandas()
        test_df = test_dataset.to_pandas()

        X_train = train_df.drop('ProdTaken', axis=1)
        y_train = train_df['ProdTaken']
        X_test = test_df.drop('ProdTaken', axis=1)
        y_test = test_df['ProdTaken']

        return X_train, X_test, y_train, y_test

    def train_and_evaluate(self):
        """Train models with hyperparameter tuning and track experiments"""
        X_train, X_test, y_train, y_test = self.load_data()

        best_model = None
        best_score = 0

        mlflow.set_experiment("wellness-tourism-prediction")

        for model_name, model in self.models.items():
            with mlflow.start_run(run_name=model_name):
                # Hyperparameter tuning
                grid_search = GridSearchCV(
                    model,
                    self.param_grids[model_name],
                    cv=5,
                    scoring='f1',
                    n_jobs=-1
                )

                grid_search.fit(X_train, y_train)

                # Best model predictions
                y_pred = grid_search.predict(X_test)

                # Calculate metrics
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred)
                recall = recall_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred)

                # Log parameters and metrics
                mlflow.log_params(grid_search.best_params_)
                mlflow.log_metrics({
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1
                })

                # Log model
                mlflow.sklearn.log_model(
                    grid_search.best_estimator_,
                    f"{model_name}_model"
                )

                print(f"{model_name} - F1 Score: {f1:.4f}")

                # Track best model
                if f1 > best_score:
                    best_score = f1
                    best_model = grid_search.best_estimator_
                    best_model_name = model_name

        # Save best model
        joblib.dump(best_model, 'models/best_model.joblib')

        # Upload to Hugging Face Model Hub
        from huggingface_hub import HfApi
        api = HfApi()
        api.upload_file(
            path_or_fileobj="models/best_model.joblib",
            path_in_repo="best_model.joblib",
            repo_id="wellness-tourism-model",
            token=os.getenv("HF_TOKEN")
        )

        return best_model, best_model_name, best_score
