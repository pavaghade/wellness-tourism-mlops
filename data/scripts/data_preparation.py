import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from datasets import Dataset, load_dataset
import warnings
import os
from dotenv import load_dotenv

warnings.filterwarnings('ignore')
load_dotenv()

class DataPreparation:
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.hf_token = os.getenv("HF_TOKEN")
        self.raw_dataset_name = os.getenv("HF_DATASET_NAME", "wellness-tourism-raw-data")

    def load_data_from_hf(self):
        """Load raw data from Hugging Face"""
        print(f"\n{'='*60}")
        print(f"LOADING DATA FROM HUGGING FACE")
        print(f"{'='*60}\n")

        try:
            # Load from Hugging Face
            dataset = load_dataset(
                self.raw_dataset_name,
                split='train',
                token=self.hf_token
            )
            df = dataset.to_pandas()
            print(f"✓ Data loaded from Hugging Face: {self.raw_dataset_name}")
            print(f"  Shape: {df.shape}")
            print(f"  Columns: {list(df.columns)}")
            return df

        except Exception as e:
            print(f"✗ Error loading from Hugging Face: {e}")
            print("\nTrying to load from local file...")

            # Fallback to local file
            try:
                df = pd.read_csv('data/raw/tourism_data.csv')
                print(f"✓ Data loaded from local file")
                print(f"  Shape: {df.shape}")
                return df
            except Exception as e2:
                print(f"✗ Error loading from local file: {e2}")
                raise Exception("Could not load data from Hugging Face or local file")

    def clean_data(self, df):
        """Clean and preprocess the dataset"""
        print(f"\n{'='*60}")
        print(f"DATA CLEANING")
        print(f"{'='*60}\n")

        initial_rows = len(df)
        print(f"Initial number of rows: {initial_rows}")

        # Remove unnecessary columns
        columns_to_drop = []
        if 'CustomerID' in df.columns:
            columns_to_drop.append('CustomerID')

        if 'ID' in df.columns:
            columns_to_drop.append('ID')

        if columns_to_drop:
            df = df.drop(columns_to_drop, axis=1)
            print(f"✓ Dropped columns: {columns_to_drop}")

        # Handle missing values
        missing_before = df.isnull().sum().sum()
        if missing_before > 0:
            print(f"Missing values found: {missing_before}")
            df = df.dropna()
            print(f"✓ Missing values removed")
        else:
            print(f"✓ No missing values found")

        # Remove duplicates
        duplicates = df.duplicated().sum()
        if duplicates > 0:
            df = df.drop_duplicates()
            print(f"✓ Removed {duplicates} duplicate rows")
        else:
            print(f"✓ No duplicate rows found")

        final_rows = len(df)
        print(f"\nFinal number of rows: {final_rows}")
        print(f"Rows removed: {initial_rows - final_rows}")
        print(f"  Final Columns: {list(df.columns)}")
        return df

    def encode_categorical_features(self, df, fit=True):
        """Encode categorical features"""
        print(f"\n{'='*60}")
        print(f"ENCODING CATEGORICAL FEATURES")
        print(f"{'='*60}\n")

        categorical_columns = df.select_dtypes(include=['object']).columns

        for col in categorical_columns:
            if col != 'ProdTaken':  # Skip target variable if present
                if fit:
                    le = LabelEncoder()
                    df[col] = le.fit_transform(df[col])
                    self.label_encoders[col] = le
                else:
                    if col in self.label_encoders:
                        df[col] = self.label_encoders[col].transform(df[col])
                    else:
                        raise ValueError(f"No encoder found for column: {col}")

                print(f"✓ Encoded column: {col}")

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
        """Split data and upload to Hugging Face"""
        print(f"\n{'='*60}")
        print(f"SPLITTING AND SAVING DATA")
        print(f"{'='*60}\n")

        # Clean data
        df_cleaned = self.clean_data(df)

        # Encode categorical features
        df_encoded = self.encode_categorical_features(df_cleaned, fit=True)

        # Split the data
        stratify_col = df_encoded['ProdTaken'] if 'ProdTaken' in df_encoded.columns else None

        train_df, test_df = train_test_split(
            df_encoded,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify_col
        )

        print(f"✓ Data split completed")
        print(f"  Training set: {len(train_df)} rows ({(1-test_size)*100:.0f}%)")
        print(f"  Test set: {len(test_df)} rows ({test_size*100:.0f}%)")

        # Save locally first
        os.makedirs('data/processed', exist_ok=True)
        train_df.to_csv('data/processed/train.csv', index=False)
        test_df.to_csv('data/processed/test.csv', index=False)
        print(f"✓ Saved locally to data/processed/")

        # Upload to Hugging Face
        try:
            train_dataset = Dataset.from_pandas(train_df)
            test_dataset = Dataset.from_pandas(test_df)

            train_dataset.push_to_hub(
                "wellness-tourism-train",
                token=self.hf_token,
                private=False
            )
            test_dataset.push_to_hub(
                "wellness-tourism-test",
                token=self.hf_token,
                private=False
            )

            print(f"✓ Uploaded to Hugging Face")
            print(f"  Train dataset: wellness-tourism-train")
            print(f"  Test dataset: wellness-tourism-test")

        except Exception as e:
            print(f"⚠ Warning: Could not upload to Hugging Face: {e}")
            print(f"  Data is available locally in data/processed/")

        return train_df, test_df

    def get_data_summary(self, train_df, test_df):
        """Generate data summary statistics"""
        print(f"\n{'='*60}")
        print(f"DATA SUMMARY")
        print(f"{'='*60}\n")

        print(f"Training Set:")
        print(f"  Shape: {train_df.shape}")
        if 'ProdTaken' in train_df.columns:
            print(f"  Class 0: {(train_df['ProdTaken']==0).sum()} ({(train_df['ProdTaken']==0).sum()/len(train_df)*100:.2f}%)")
            print(f"  Class 1: {(train_df['ProdTaken']==1).sum()} ({(train_df['ProdTaken']==1).sum()/len(train_df)*100:.2f}%)")

        print(f"\nTest Set:")
        print(f"  Shape: {test_df.shape}")
        if 'ProdTaken' in test_df.columns:
            print(f"  Class 0: {(test_df['ProdTaken']==0).sum()} ({(test_df['ProdTaken']==0).sum()/len(test_df)*100:.2f}%)")
            print(f"  Class 1: {(test_df['ProdTaken']==1).sum()} ({(test_df['ProdTaken']==1).sum()/len(test_df)*100:.2f}%)")

        print(f"\n{'='*60}\n")

def main():
    """Main execution function"""
    print("\n" + "="*60)
    print("DATA PREPARATION PIPELINE")
    print("="*60)

    # Initialize
    data_prep = DataPreparation()

    # Load data from Hugging Face
    df = data_prep.load_data_from_hf()

    # Split and save data
    train_df, test_df = data_prep.split_and_save_data(df)

    # Generate summary
    data_prep.get_data_summary(train_df, test_df)

    print("✓ Data preparation completed successfully!\n")

if __name__ == "__main__":
    main()