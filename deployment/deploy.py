import os
from huggingface_hub import HfApi, create_repo

def deploy_to_hf_spaces():
    """Deploy Streamlit app to Hugging Face Spaces"""
    
    print("\n" + "="*60)
    print("DEPLOYING TO HUGGING FACE SPACES")
    print("="*60 + "\n")
    
    api = HfApi()
    token = os.getenv("HF_TOKEN")
    space_name = os.getenv("HF_SPACE_NAME", "wellness-tourism-app")
    
    if not token:
        print("✗ Error: HF_TOKEN not found")
        return False
    
    print(f"Space name: {space_name}")
    
    try:
        print(f"\nCreating/verifying Space repository...")
        repo_id = create_repo(
            repo_id=space_name,
            repo_type="space",
            space_sdk="streamlit",
            token=token,
            exist_ok=True,
            private=False
        )
        print(f"✓ Space repository ready: {repo_id}")
    except Exception as e:
        print(f"Repository status: {str(e)[:200]}")
    
    files_to_upload = [
        ("app.py", "app.py"),
        ("requirements.txt", "requirements.txt"),
    ]
    
    print(f"\nUploading files to Space...")
    for local_file, remote_file in files_to_upload:
        try:
            if os.path.exists(local_file):
                api.upload_file(
                    path_or_fileobj=local_file,
                    path_in_repo=remote_file,
                    repo_id=space_name,
                    repo_type="space",
                    token=token
                )
                print(f"✓ Uploaded: {local_file}")
            else:
                print(f"⚠ File not found: {local_file}")
        except Exception as e:
            print(f"✗ Error uploading {local_file}: {e}")
    
    readme_content = """---
title: Wellness Tourism Package Predictor
emoji: ✈️
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: 1.28.1
app_file: app.py
pinned: false
---

# Wellness Tourism Package Predictor

Predict customer likelihood to purchase wellness tourism packages using machine learning.

## Features
- Real-time predictions
- Interactive user interface
- Comprehensive customer profiling
- Purchase probability analysis

## Usage
1. Enter customer information
2. Provide travel preferences
3. Add interaction data
4. Get instant predictions

## Model
Uses trained ML model from Hugging Face Model Hub for accurate predictions.
"""
    
    try:
        print(f"\nUploading README...")
        api.upload_file(
            path_or_fileobj=readme_content.encode(),
            path_in_repo="README.md",
            repo_id=space_name,
            repo_type="space",
            token=token
        )
        print(f"✓ README uploaded")
    except Exception as e:
        print(f"README upload: {e}")
    
    print(f"\n{'='*60}")
    print(f"✅ DEPLOYMENT COMPLETED!")
    print(f"{'='*60}")
    print(f"\nSpace URL: https://huggingface.co/spaces/{space_name}")
    print(f"\nNote: It may take a few minutes for the Space to build and start.")
    print(f"{'='*60}\n")
    
    return True

if __name__ == "__main__":
    success = deploy_to_hf_spaces()
    if not success:
        exit(1)