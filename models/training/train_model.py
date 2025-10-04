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
import json
from huggingface_hub import HfApi, create_repo

class ModelTrainer:
    def __init__(self):
        self.hf_token = os.getenv("HF_TOKEN")
        self.model_repo_name = os.getenv("HF_MODEL_REPO", "wellness-tourism-model")

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
        print(f"\n{'='*60}")
        print(f"LOADING TRAINING DATA")
        print(f"{'='*60}\n")

        try:
            train_dataset = load_dataset("wellness-tourism-train", split='train', token=self.hf_token)
            test_dataset = load_dataset("wellness-tourism-test", split='train', token=self.hf_token)

            train_df = train_dataset.to_pandas()
            test_df = test_dataset.to_pandas()

            print(f"✓ Train dataset loaded: {train_df.shape}")
            print(f"✓ Test dataset loaded: {test_df.shape}")

        except Exception as e:
            print(f"Error loading from Hugging Face: {e}")
            print("Trying local files...")

            train_df = pd.read_csv('data/processed/train.csv')
            test_df = pd.read_csv('data/processed/test.csv')
            print(f"✓ Loaded from local files")

        X_train = train_df.drop('ProdTaken', axis=1)
        y_train = train_df['ProdTaken']
        X_test = test_df.drop('ProdTaken', axis=1)
        y_test = test_df['ProdTaken']

        return X_train, X_test, y_train, y_test

    def upload_model_to_hf(self, model, model_name, metrics):
        print(f"\n{'='*60}")
        print(f"UPLOADING MODEL TO HUGGING FACE")
        print(f"{'='*60}\n")

        try:
            api = HfApi()

            os.makedirs('models', exist_ok=True)

            model_path = 'models/best_model.joblib'
            joblib.dump(model, model_path)
            print(f"✓ Model saved locally: {model_path}")

            if not self.hf_token:
                print("✗ HF_TOKEN not found in environment variables!")
                print("  Model saved locally but not uploaded to Hugging Face")
                return False

            print(f"Using model repository: {self.model_repo_name}")

            try:
                repo_url = create_repo(
                    repo_id=self.model_repo_name,
                    repo_type="model",
                    token=self.hf_token,
                    private=False,
                    exist_ok=True
                )
                print(f"✓ Repository created/verified: {self.model_repo_name}")
            except Exception as e:
                print(f"Repository status: {str(e)[:200]}")

            print(f"\nUploading model file to Hugging Face...")
            upload_result = api.upload_file(
                path_or_fileobj=model_path,
                path_in_repo="best_model.joblib",
                repo_id=self.model_repo_name,
                repo_type="model",
                token=self.hf_token,
                commit_message=f"Upload {model_name} model with F1={metrics.get('f1_score', 0):.4f}"
            )
            print(f"✓ Model uploaded successfully!")

            readme_content = f"""---
tags:
- sklearn
- classification
- wellness-tourism
library_name: scikit-learn
---

# Wellness Tourism Package Prediction Model

## Model Details
- **Algorithm**: {model_name}
- **Task**: Binary Classification
- **Framework**: scikit-learn

## Performance Metrics
- **Accuracy**: {metrics.get('accuracy', 0):.4f}
- **Precision**: {metrics.get('precision', 0):.4f}
- **Recall**: {metrics.get('recall', 0):.4f}
- **F1 Score**: {metrics.get('f1_score', 0):.4f}

## Usage

```python
import joblib
from huggingface_hub import hf_hub_download

model_path = hf_hub_download(
    repo_id="{self.model_repo_name}",
    filename="best_model.joblib"
)

model = joblib.load(model_path)
predictions = model.predict(X_test)
```
"""

            print(f"\nUploading README...")
            api.upload_file(
                path_or_fileobj=readme_content.encode(),
                path_in_repo="README.md",
                repo_id=self.model_repo_name,
                repo_type="model",
                token=self.hf_token,
                commit_message="Add model documentation"
            )
            print(f"✓ README uploaded")

            model_info = {
                'model_name': model_name,
                'metrics': metrics,
                'repository': self.model_repo_name,
                'hf_url': f"https://huggingface.co/{self.model_repo_name}"
            }

            with open('models/model_info.json', 'w') as f:
                json.dump(model_info, f, indent=4)

            print(f"\n{'='*60}")
            print(f"✅ MODEL SUCCESSFULLY UPLOADED TO HUGGING FACE!")
            print(f"{'='*60}")
            print(f"Repository: https://huggingface.co/{self.model_repo_name}")
            print(f"{'='*60}\n")

            return True

        except Exception as e:
            print(f"\n✗ ERROR: {str(e)}")
            print(f"Model saved locally at: {model_path}")
            return False

    def train_and_evaluate(self):
        print(f"\n{'='*60}")
        print(f"MODEL TRAINING AND EVALUATION")
        print(f"{'='*60}\n")

        X_train, X_test, y_train, y_test = self.load_data()

        best_model = None
        best_score = 0
        best_model_name = ""
        best_metrics = {}

        try:
            mlflow.set_experiment("wellness-tourism-prediction")
            print("✓ MLflow experiment set up")
        except Exception as e:
            print(f"MLflow setup: {e}")

        for model_name, model in self.models.items():
            print(f"\n--- Training {model_name} ---")

            try:
                with mlflow.start_run(run_name=model_name):
                    grid_search = GridSearchCV(
                        model,
                        self.param_grids[model_name],
                        cv=5,
                        scoring='f1',
                        n_jobs=-1,
                        verbose=1
                    )

                    grid_search.fit(X_train, y_train)

                    y_pred = grid_search.predict(X_test)

                    accuracy = accuracy_score(y_test, y_pred)
                    precision = precision_score(y_test, y_pred, zero_division=0)
                    recall = recall_score(y_test, y_pred, zero_division=0)
                    f1 = f1_score(y_test, y_pred, zero_division=0)

                    current_metrics = {
                        'accuracy': accuracy,
                        'precision': precision,
                        'recall': recall,
                        'f1_score': f1
                    }

                    try:
                        mlflow.log_params(grid_search.best_params_)
                        mlflow.log_metrics(current_metrics)
                        mlflow.sklearn.log_model(grid_search.best_estimator_, f"{model_name}_model")
                    except Exception as e:
                        print(f"MLflow logging error: {e}")

                    print(f"✓ {model_name} trained successfully")
                    print(f"  F1 Score: {f1:.4f}")

                    if f1 > best_score:
                        best_score = f1
                        best_model = grid_search.best_estimator_
                        best_model_name = model_name
                        best_metrics = current_metrics

            except Exception as e:
                print(f"✗ Error training {model_name}: {e}")
                continue

        if best_model is None:
            raise Exception("No models were successfully trained!")

        print(f"\n{'='*60}")
        print(f"BEST MODEL: {best_model_name}")
        print(f"{'='*60}")
        print(f"F1 Score: {best_score:.4f}")
        print(f"Accuracy: {best_metrics['accuracy']:.4f}")
        print(f"{'='*60}\n")

        os.makedirs('models', exist_ok=True)
        joblib.dump(best_model, 'models/best_model.joblib')
        print(f"✓ Best model saved locally")

        self.upload_model_to_hf(best_model, best_model_name, best_metrics)

        return best_model, best_model_name, best_score

def main():
    trainer = ModelTrainer()
    model, model_name, score = trainer.train_and_evaluate()
    print("\n✓ Training pipeline completed successfully!\n")

if __name__ == "__main__":
    main()