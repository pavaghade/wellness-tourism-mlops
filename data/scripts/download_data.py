import os
from datasets import load_dataset
import pandas as pd
from huggingface_hub import hf_hub_download
from dotenv import load_dotenv

load_dotenv()

class DataDownloader:
    def __init__(self):
        self.hf_token = os.getenv("HF_TOKEN")
        self.dataset_name = os.getenv("HF_DATASET_NAME", "wellness-tourism-raw-data")
        self.raw_data_dir = "data/raw"
        
    def create_directories(self):
        """Create necessary directories if they don't exist"""
        os.makedirs(self.raw_data_dir, exist_ok=True)
        print(f"✓ Created directory: {self.raw_data_dir}")
    
    def download_from_huggingface(self, save_filename="tourism_data.csv"):
        """
        Download dataset from Hugging Face and save as CSV
        
        Args:
            save_filename: Name of the file to save locally
        """
        try:
            print(f"Downloading dataset from Hugging Face: {self.dataset_name}")
            
            # Method 1: Load dataset using datasets library
            dataset = load_dataset(
                self.dataset_name, 
                split='train',
                token=self.hf_token
            )
            
            # Convert to pandas DataFrame
            df = dataset.to_pandas()
            
            # Save to local CSV
            save_path = os.path.join(self.raw_data_dir, save_filename)
            df.to_csv(save_path, index=False)
            
            print(f"✓ Dataset downloaded successfully!")
            print(f"✓ Saved to: {save_path}")
            print(f"✓ Shape: {df.shape}")
            print(f"✓ Columns: {list(df.columns)}")
            
            return save_path
            
        except Exception as e:
            print(f"Error downloading dataset: {e}")
            print("\nTrying alternative method...")
            
            # Method 2: Download specific file from HF Hub
            try:
                file_path = hf_hub_download(
                    repo_id=self.dataset_name,
                    filename="tourism_data.csv",
                    repo_type="dataset",
                    token=self.hf_token
                )
                
                # Copy to data/raw directory
                import shutil
                save_path = os.path.join(self.raw_data_dir, save_filename)
                shutil.copy(file_path, save_path)
                
                print(f"✓ Dataset downloaded successfully using alternative method!")
                print(f"✓ Saved to: {save_path}")
                
                return save_path
                
            except Exception as e2:
                print(f"Error with alternative method: {e2}")
                raise Exception("Failed to download dataset from Hugging Face")
    
    def verify_download(self, filepath):
        """Verify the downloaded dataset"""
        try:
            df = pd.read_csv(filepath)
            print(f"\n{'='*60}")
            print("DATA VERIFICATION")
            print(f"{'='*60}")
            print(f"File exists: ✓")
            print(f"Number of rows: {len(df)}")
            print(f"Number of columns: {len(df.columns)}")
            print(f"\nColumn names:")
            for col in df.columns:
                print(f"  - {col}")
            print(f"{'='*60}\n")
            return True
        except Exception as e:
            print(f"Verification failed: {e}")
            return False

def main():
    """Main execution function"""
    print("\n" + "="*60)
    print("DOWNLOADING RAW DATA FROM HUGGING FACE")
    print("="*60 + "\n")
    
    downloader = DataDownloader()
    
    # Create directories
    downloader.create_directories()
    
    # Download dataset
    filepath = downloader.download_from_huggingface()
    
    # Verify download
    if downloader.verify_download(filepath):
        print("✓ Data download completed successfully!\n")
    else:
        print("✗ Data verification failed!\n")
        exit(1)

if __name__ == "__main__":
    main()