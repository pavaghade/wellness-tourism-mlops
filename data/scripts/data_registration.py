import pandas as pd
from datasets import Dataset
from huggingface_hub import HfApi
import os
from dotenv import load_dotenv

load_dotenv()

class DataRegistration:
    def __init__(self):
        self.hf_token = os.getenv("HF_TOKEN")
        self.api = HfApi()

    def register_raw_data(self, data_path, dataset_name):
        """Register raw data to Hugging Face dataset space"""
        df = pd.read_csv(data_path)
        dataset = Dataset.from_pandas(df)

        dataset.push_to_hub(
            dataset_name,
            token=self.hf_token,
            private=False
        )
        print(f"Raw data registered to {dataset_name}")

    def load_from_hf(self, dataset_name):
        """Load data from Hugging Face dataset space"""
        dataset = Dataset.from_hub(dataset_name)
        return dataset.to_pandas()
