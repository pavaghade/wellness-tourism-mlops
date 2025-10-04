import pandas as pd
import numpy as np
import joblib
import json
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
from datasets import load_dataset
import matplotlib.pyplot as plt
import seaborn as sns
from huggingface_hub import hf_hub_download
import os

class ModelEvaluator:
    def __init__(self):
        self.metrics = {}

    def load_model_and_data(self):
        """Load the best model from Hugging Face and test data"""
        # Download model from Hugging Face
        try:
            model_path = hf_hub_download(
                repo_id="wellness-tourism-model",
                filename="best_model.joblib",
                token=os.getenv("HF_TOKEN")
            )
            model = joblib.load(model_path)
            print("Model loaded successfully from Hugging Face")
        except Exception as e:
            print(f"Error loading from HF, trying local: {e}")
            model = joblib.load('models/best_model.joblib')

        # Load test data
        test_dataset = load_dataset("wellness-tourism-test", split='train')
        test_df = test_dataset.to_pandas()

        X_test = test_df.drop('ProdTaken', axis=1)
        y_test = test_df['ProdTaken']

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
        print(f"Confusion matrix saved to {save_path}")

    def generate_roc_curve(self, y_true, y_pred_proba, save_path='reports/roc_curve.png'):
        """Generate and save ROC curve"""
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

        # Generate visualizations
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
