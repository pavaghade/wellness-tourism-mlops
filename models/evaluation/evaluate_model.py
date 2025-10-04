import pandas as pd
import numpy as np
import joblib
import json
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
from datasets import load_dataset
import os

# Import visualization libraries with error handling
try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend for CI/CD
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Plotting libraries not available: {e}")
    print("Continuing without visualization capabilities")
    PLOTTING_AVAILABLE = False

from huggingface_hub import hf_hub_download

class ModelEvaluator:
    def __init__(self):
        self.metrics = {}
        self.plotting_available = PLOTTING_AVAILABLE
        self.hf_token = os.getenv("HF_TOKEN")
        self.model_repo_name = os.getenv("HF_MODEL_REPO", "wellness-tourism-model")

    def load_model_and_data(self):
        """Load the best model from Hugging Face and test data"""
        print(f"\n{'='*60}")
        print(f"LOADING MODEL AND TEST DATA")
        print(f"{'='*60}\n")

        # Try to download model from Hugging Face
        model = None
        try:
            print(f"Attempting to download from: {self.model_repo_name}")
            model_path = hf_hub_download(
                repo_id=self.model_repo_name,
                filename="best_model.joblib",
                token=self.hf_token
            )
            model = joblib.load(model_path)
            print(f"✓ Model loaded from Hugging Face: {self.model_repo_name}")
        except Exception as e:
            print(f"Could not load from Hugging Face: {e}")
            print("Trying local model file...")

            # Try local file
            try:
                model = joblib.load('models/best_model.joblib')
                print(f"✓ Model loaded from local file: models/best_model.joblib")
            except Exception as e2:
                print(f"✗ Error loading model: {e2}")
                raise Exception(
                    "Model not found! Please ensure:\n"
                    "1. Model is uploaded to Hugging Face, OR\n"
                    "2. Model exists locally at models/best_model.joblib\n"
                    f"Expected HF repo: {self.model_repo_name}"
                )

        # Load test data
        try:
            test_dataset = load_dataset("wellness-tourism-test", split='train', token=self.hf_token)
            test_df = test_dataset.to_pandas()
            print(f"✓ Test data loaded from Hugging Face")
        except Exception as e:
            print(f"Could not load from Hugging Face: {e}")
            print("Trying local test data...")
            test_df = pd.read_csv('data/processed/test.csv')
            print(f"✓ Test data loaded from local file")

        X_test = test_df.drop('ProdTaken', axis=1)
        y_test = test_df['ProdTaken']

        print(f"  Test set shape: {X_test.shape}")
        print(f"{'='*60}\n")

        return model, X_test, y_test

    def calculate_metrics(self, y_true, y_pred, y_pred_proba=None):
        """Calculate comprehensive evaluation metrics"""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, zero_division=0)
        }

        if y_pred_proba is not None:
            metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)

        return metrics

    def generate_confusion_matrix(self, y_true, y_pred, save_path='reports/confusion_matrix.png'):
        """Generate and save confusion matrix visualization"""
        if not self.plotting_available:
            print("Skipping confusion matrix plot - matplotlib/seaborn not available")
            return

        cm = confusion_matrix(y_true, y_pred)

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Not Purchase', 'Purchase'],
                    yticklabels=['Not Purchase', 'Purchase'])
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')

        os.makedirs('reports', exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Confusion matrix saved to {save_path}")

    def generate_roc_curve(self, y_true, y_pred_proba, save_path='reports/roc_curve.png'):
        """Generate and save ROC curve"""
        if not self.plotting_available:
            print("Skipping ROC curve plot - matplotlib not available")
            return

        fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
        roc_auc = roc_auc_score(y_true, y_pred_proba)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2,
                 label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")

        os.makedirs('reports', exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ ROC curve saved to {save_path}")

    def generate_classification_report(self, y_true, y_pred, save_path='reports/classification_report.txt'):
        """Generate and save detailed classification report"""
        report = classification_report(y_true, y_pred,
                                       target_names=['Not Purchase', 'Purchase'])

        os.makedirs('reports', exist_ok=True)
        with open(save_path, 'w') as f:
            f.write("Classification Report\n")
            f.write("=" * 50 + "\n\n")
            f.write(report)

        print(f"✓ Classification report saved to {save_path}")
        return report

    def evaluate_model(self):
        """Complete model evaluation pipeline"""
        print("\nSTARTING MODEL EVALUATION...")

        # Load model and data
        model, X_test, y_test = self.load_model_and_data()

        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        # Calculate metrics
        self.metrics = self.calculate_metrics(y_test, y_pred, y_pred_proba)

        # Print metrics
        print("\n" + "=" * 50)
        print("MODEL EVALUATION RESULTS")
        print("=" * 50)
        for metric_name, metric_value in self.metrics.items():
            print(f"{metric_name.upper()}: {metric_value:.4f}")
        print("=" * 50 + "\n")

        # Generate visualizations (if available)
        if self.plotting_available:
            self.generate_confusion_matrix(y_test, y_pred)
            self.generate_roc_curve(y_test, y_pred_proba)

        # Generate classification report
        self.generate_classification_report(y_test, y_pred)

        # Save metrics to JSON
        os.makedirs('reports', exist_ok=True)
        with open('reports/metrics.json', 'w') as f:
            json.dump(self.metrics, f, indent=4)
        print("✓ Metrics saved to reports/metrics.json")

        # Feature importance (if available)
        if hasattr(model, 'feature_importances_'):
            self.plot_feature_importance(model, X_test.columns)

        return self.metrics

    def plot_feature_importance(self, model, feature_names, save_path='reports/feature_importance.png'):
        """Plot and save feature importance"""
        if not self.plotting_available:
            print("Skipping feature importance plot - matplotlib not available")
            return

        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:15]  # Top 15 features

        plt.figure(figsize=(10, 6))
        plt.title('Top 15 Feature Importances')
        plt.bar(range(len(indices)), importances[indices])
        plt.xticks(range(len(indices)), [feature_names[i] for i in indices], rotation=45, ha='right')
        plt.tight_layout()

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Feature importance plot saved to {save_path}")

def main():
    """Main execution function"""
    evaluator = ModelEvaluator()
    metrics = evaluator.evaluate_model()

    print("\n✓ Evaluation completed successfully!\n")

if __name__ == "__main__":
    main()
```=['Not Purchase', 'Purchase'])
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')

        os.makedirs('reports', exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Confusion matrix saved to {save_path}")

    def generate_roc_curve(self, y_true, y_pred_proba, save_path='reports/roc_curve.png'):
        """Generate and save ROC curve"""
        if not self.plotting_available:
            print("Skipping ROC curve plot - matplotlib not available")
            return

        fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
        roc_auc = roc_auc_score(y_true, y_pred_proba)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2,
                 label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")

        os.makedirs('reports', exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"ROC curve saved to {save_path}")

    def generate_classification_report(self, y_true, y_pred, save_path='reports/classification_report.txt'):
        """Generate and save detailed classification report"""
        report = classification_report(y_true, y_pred,
                                       target_names=['Not Purchase', 'Purchase'])

        os.makedirs('reports', exist_ok=True)
        with open(save_path, 'w') as f:
            f.write("Classification Report\n")
            f.write("=" * 50 + "\n\n")
            f.write(report)

        print(f"Classification report saved to {save_path}")
        return report

    def evaluate_model(self):
        """Complete model evaluation pipeline"""
        print("Starting model evaluation...")

        # Load model and data
        model, X_test, y_test = self.load_model_and_data()

        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        # Calculate metrics
        self.metrics = self.calculate_metrics(y_test, y_pred, y_pred_proba)

        # Print metrics
        print("\n" + "=" * 50)
        print("MODEL EVALUATION RESULTS")
        print("=" * 50)
        for metric_name, metric_value in self.metrics.items():
            print(f"{metric_name.upper()}: {metric_value:.4f}")
        print("=" * 50 + "\n")

        # Generate visualizations (if available)
        if self.plotting_available:
            self.generate_confusion_matrix(y_test, y_pred)
            self.generate_roc_curve(y_test, y_pred_proba)

        # Generate classification report
        self.generate_classification_report(y_test, y_pred)

        # Save metrics to JSON
        os.makedirs('reports', exist_ok=True)
        with open('reports/metrics.json', 'w') as f:
            json.dump(self.metrics, f, indent=4)
        print("Metrics saved to reports/metrics.json")

        # Feature importance (if available)
        if hasattr(model, 'feature_importances_'):
            self.plot_feature_importance(model, X_test.columns)

        return self.metrics

    def plot_feature_importance(self, model, feature_names, save_path='reports/feature_importance.png'):
        """Plot and save feature importance"""
        if not self.plotting_available:
            print("Skipping feature importance plot - matplotlib not available")
            return

        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:15]  # Top 15 features

        plt.figure(figsize=(10, 6))
        plt.title('Top 15 Feature Importances')
        plt.bar(range(len(indices)), importances[indices])
        plt.xticks(range(len(indices)), [feature_names[i] for i in indices], rotation=45, ha='right')
        plt.tight_layout()

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Feature importance plot saved to {save_path}")

if __name__ == "__main__":
    evaluator = ModelEvaluator()
    metrics = evaluator.evaluate_model()

    print("\nEvaluation completed successfully!")