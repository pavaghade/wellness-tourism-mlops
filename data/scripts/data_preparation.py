import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from datasets import Dataset, load_dataset
from huggingface_hub import create_repo
import warnings
import os
from dotenv import load_dotenv

warnings.filterwarnings('ignore')
load_dotenv()

class DataPreparation:
    def __init__(self):
        self.label_encoders = {}
        self.hf_token = os.getenv("HF_TOKEN")
        self.raw_dataset_name = os.getenv("HF_DATASET_NAME", "wellness-tourism-raw-data")
        
    def load_data_from_hf(self):
        print(f"\n{'='*60}")
        print(f"LOADING DATA FROM HUGGING FACE")
        print(f"{'='*60}\n")
        
        try:
            dataset = load_dataset(
                self.raw_dataset_name, 
                split='train',
                token=self.hf_token
            )
            df = dataset.to_pandas()
            print(f"✓ Data loaded from Hugging Face: {self.raw_dataset_name}")
            print(f"  Shape: {df.shape}")
            return df
            
        except Exception as e:
            print(f"✗ Error loading from Hugging Face: {e}")
            print("\nTrying to load from local file...")
            
            try:
                df = pd.read_csv('data/raw/tourism_data.csv')
                print(f"✓ Data loaded from local file")
                print(f"  Shape: {df.shape}")
                return df
            except Exception as e2:
                raise Exception(f"Could not load data: {e2}")
    
    def clean_data(self, df):
        print(f"\n{'='*60}")
        print(f"DATA CLEANING")
        print(f"{'='*60}\n")
        
        initial_rows = len(df)
        print(f"Initial rows: {initial_rows}")
        
        columns_to_drop = []
        if 'CustomerID' in df.columns:
            columns_to_drop.append('CustomerID')
            
        if columns_to_drop:
            df = df.drop(columns_to_drop, axis=1)
            print(f"✓ Dropped columns: {columns_to_drop}")
        
        missing_before = df.isnull().sum().sum()
        if missing_before > 0:
            print(f"Missing values found: {missing_before}")
            df = df.dropna()
            print(f"✓ Missing values removed")
        else:
            print(f"✓ No missing values")
        
        duplicates = df.duplicated().sum()
        if duplicates > 0:
            df = df.drop_duplicates()
            print(f"✓ Removed {duplicates} duplicates")
        else:
            print(f"✓ No duplicates")
        
        final_rows = len(df)
        print(f"\nFinal rows: {final_rows}")
        print(f"Rows removed: {initial_rows - final_rows}")
        
        return df
    
    def encode_categorical_features(self, df):
        print(f"\n{'='*60}")
        print(f"ENCODING CATEGORICAL FEATURES")
        print(f"{'='*60}\n")
        
        categorical_columns = df.select_dtypes(include=['object']).columns
        
        for col in categorical_columns:
            if col != 'ProdTaken':
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col])
                self.label_encoders[col] = le
                print(f"✓ Encoded: {col}")
                
        return df
    
    def split_and_save_data(self, df, test_size=0.2, random_state=42):
        print(f"\n{'='*60}")
        print(f"SPLITTING AND SAVING DATA")
        print(f"{'='*60}\n")
        
        df_cleaned = self.clean_data(df)
        df_encoded = self.encode_categorical_features(df_cleaned)
        
        stratify_col = df_encoded['ProdTaken'] if 'ProdTaken' in df_encoded.columns else None
        
        train_df, test_df = train_test_split(
            df_encoded, 
            test_size=test_size, 
            random_state=random_state,
            stratify=stratify_col
        )
        
        print(f"✓ Data split completed")
        print(f"  Training: {len(train_df)} rows")
        print(f"  Test: {len(test_df)} rows")
        
        os.makedirs('data/processed', exist_ok=True)
        train_df.to_csv('data/processed/train.csv', index=False)
        test_df.to_csv('data/processed/test.csv', index=False)
        print(f"✓ Saved locally to data/processed/")
        
        if self.hf_token:
            try:
                print(f"\nUploading to Hugging Face...")
                
                train_dataset = Dataset.from_pandas(train_df)
                test_dataset = Dataset.from_pandas(test_df)
                
                try:
                    create_repo("wellness-tourism-train", repo_type="dataset", token=self.hf_token, exist_ok=True)
                    create_repo("wellness-tourism-test", repo_type="dataset", token=self.hf_token, exist_ok=True)
                except:
                    pass
                
                train_dataset.push_to_hub(
                    "wellness-tourism-train", 
                    token=self.hf_token,
                    private=False
                )
                print(f"✓ Train dataset uploaded: wellness-tourism-train")
                
                test_dataset.push_to_hub(
                    "wellness-tourism-test", 
                    token=self.hf_token,
                    private=False
                )
                print(f"✓ Test dataset uploaded: wellness-tourism-test")
                
            except Exception as e:
                print(f"⚠ Warning: Could not upload to HF: {e}")
                print(f"  Data available locally in data/processed/")
        else:
            print(f"⚠ HF_TOKEN not set, skipping upload")
        
        return train_df, test_df
    
    def get_data_summary(self, train_df, test_df):
        print(f"\n{'='*60}")
        print(f"DATA SUMMARY")
        print(f"{'='*60}\n")
        
        print(f"Training Set: {train_df.shape}")
        if 'ProdTaken' in train_df.columns:
            print(f"  Class 0: {(train_df['ProdTaken']==0).sum()} ({(train_df['ProdTaken']==0).sum()/len(train_df)*100:.1f}%)")
            print(f"  Class 1: {(train_df['ProdTaken']==1).sum()} ({(train_df['ProdTaken']==1).sum()/len(train_df)*100:.1f}%)")
        
        print(f"\nTest Set: {test_df.shape}")
        if 'ProdTaken' in test_df.columns:
            print(f"  Class 0: {(test_df['ProdTaken']==0).sum()} ({(test_df['ProdTaken']==0).sum()/len(test_df)*100:.1f}%)")
            print(f"  Class 1: {(test_df['ProdTaken']==1).sum()} ({(test_df['ProdTaken']==1).sum()/len(test_df)*100:.1f}%)")
        
        print(f"\n{'='*60}\n")

def main():
    print("\n" + "="*60)
    print("DATA PREPARATION PIPELINE")
    print("="*60)
    
    data_prep = DataPreparation()
    df = data_prep.load_data_from_hf()
    train_df, test_df = data_prep.split_and_save_data(df)
    data_prep.get_data_summary(train_df, test_df)
    
    print("✓ Data preparation completed!\n")

if __name__ == "__main__":
    main()