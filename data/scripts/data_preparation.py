import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from datasets import Dataset
import warnings
warnings.filterwarnings('ignore')

class DataPreparation:
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()

    def clean_data(self, df):
        """Clean and preprocess the dataset"""
        # Remove unnecessary columns
        columns_to_drop = ['CustomerID']
        if 'CustomerID' in df.columns:
            df = df.drop(columns_to_drop, axis=1)

        # Handle missing values
        df = df.dropna()

        # Remove duplicates
        df = df.drop_duplicates()

        return df

    def encode_categorical_features(self, df):
        """Encode categorical features"""
        categorical_columns = df.select_dtypes(include=['object']).columns

        for col in categorical_columns:
            if col != 'ProdTaken':  # Skip target variable if present
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col])
                self.label_encoders[col] = le

        return df

    def prepare_features(self, df):
        """Prepare features for model training"""
        # Separate features and target
        if 'ProdTaken' in df.columns:
            X = df.drop('ProdTaken', axis=1)
            y = df['ProdTaken']
        else:
            X = df
            y = None

        return X, y

    def split_and_save_data(self, df, test_size=0.2, random_state=42):
        """Split data and save to Hugging Face"""
        # Clean and encode data
        df_cleaned = self.clean_data(df)
        df_encoded = self.encode_categorical_features(df_cleaned)

        # Split the data
        train_df, test_df = train_test_split(
            df_encoded,
            test_size=test_size,
            random_state=random_state,
            stratify=df_encoded['ProdTaken'] if 'ProdTaken' in df_encoded.columns else None
        )

        # Upload to Hugging Face
        train_dataset = Dataset.from_pandas(train_df)
        test_dataset = Dataset.from_pandas(test_df)

        train_dataset.push_to_hub("wellness-tourism-train", token=os.getenv("HF_TOKEN"))
        test_dataset.push_to_hub("wellness-tourism-test", token=os.getenv("HF_TOKEN"))

        return train_df, test_df
