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
        """Load data from Hugging Face"""
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
        """Upload model to Hugging Face Model Hub with proper error handling"""
        print(f"\n{'='*60}")
        print(f"UPLOADING MODEL TO HUGGING FACE")
        print(f"{'='*60}\n")

        try:
            api = HfApi()

            # Create model directory
            os.makedirs('models', exist_ok=True)

            # Save model locally
            model_path = 'models/best_model.joblib'
            joblib.dump(model, model_path)
            print(f"✓ Model saved locally: {model_path}")

            # Verify token exists
            if not self.hf_token:
                print("✗ HF_TOKEN not found in environment variables!")
                print("  Model saved locally but not uploaded to Hugging Face")
                print(f"  You can upload manually or set HF_TOKEN and rerun")
                return False

            print(f"Using model repository: {self.model_repo_name}")

            # Create repository if it doesn't exist
            try:
                repo_url = create_repo(
                    repo_id=self.model_repo_name,
                    repo_type="model",
                    token=self.hf_token,
                    private=False,
                    exist_ok=True
                )
                print(f"✓ Repository created/verified: {self.model_repo_name}")
                print(f"  URL: https://huggingface.co/{self.model_repo_name}")
            except Exception as e:
                print(f"Repository creation: {str(e)[:200]}")

            # Upload model file
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
            print(f"  File URL: {upload_result}")
        except Exception as e:
          print(f"Repository creation: {str(e)[:200]}")