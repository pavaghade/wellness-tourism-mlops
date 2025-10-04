import os
from huggingface_hub import HfApi

def deploy_to_hf_spaces():
    """Deploy Streamlit app to Hugging Face Spaces"""
    api = HfApi()

    # Create space
    api.create_repo(
        repo_id="wellness-tourism-app",
        repo_type="space",
        space_sdk="streamlit",
        token=os.getenv("HF_TOKEN")
    )

    # Upload files
    files_to_upload = [
        "app.py",
        "requirements.txt",
        "Dockerfile"
    ]

    for file in files_to_upload:
        api.upload_file(
            path_or_fileobj=file,
            path_in_repo=file,
            repo_id="wellness-tourism-app",
            repo_type="space",
            token=os.getenv("HF_TOKEN")
        )

    print("Deployment to Hugging Face Spaces completed!")

if __name__ == "__main__":
    deploy_to_hf_spaces()
