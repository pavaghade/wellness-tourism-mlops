import os
from huggingface_hub import HfApi, SpaceSdk

def deploy_to_hf_spaces():
    """Deploy Streamlit app to Hugging Face Spaces"""
    token = os.getenv("HF_TOKEN")
    if not token:
        raise EnvironmentError("❌ HF_TOKEN not set. Make sure it's in your GitHub secrets.")

    api = HfApi()

    # Create or reuse the Space
    api.create_repo(
        repo_id="wellness-tourism-app",
        repo_type="space",
        token=token,
        space_sdk="streamlit",
        exist_ok=True  # ✅ don’t fail if it already exists
    )

    # Upload files
    files_to_upload = [
        "app.py",
        "requirements.txt",
        "Dockerfile"
    ]

    for file in files_to_upload:
        if not os.path.exists(file):
            raise FileNotFoundError(f"File not found: {file}")
        api.upload_file(
            path_or_fileobj=file,
            path_in_repo=file,
            repo_id="wellness-tourism-app",
            repo_type="space",
            token=token
        )

    print("✅ Deployment to Hugging Face Spaces completed!")

if __name__ == "__main__":
    deploy_to_hf_spaces()
