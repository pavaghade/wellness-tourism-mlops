# Wellness Tourism Package Prediction MLOps Pipeline

## Overview
This project implements an end-to-end MLOps pipeline for predicting customer purchases of wellness tourism packages using machine learning and automated CI/CD workflows.

## Features
- Automated data preprocessing and feature engineering
- Multiple ML algorithm comparison and hyperparameter tuning
- MLflow experiment tracking
- Model deployment to Hugging Face Spaces
- GitHub Actions CI/CD pipeline
- Streamlit web application for predictions

## Setup Instructions

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Set up environment variables:
   - `HF_TOKEN`: Hugging Face API token
   - `MLFLOW_TRACKING_URI`: MLflow tracking server URI

4. Run the pipeline:
```bash
   python data/scripts/data_preparation.py
   python models/training/train_model.py
   streamlit run deployment/app.py
