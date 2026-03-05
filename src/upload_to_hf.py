import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import HfHubHTTPError

HF_USERNAME = "ml8r"   # your HuggingFace username
DATASET_NAME = "cinematch-data"    # name for the HF dataset repo

load_dotenv()
token = os.environ.get("HF_TOKEN")
if not token:
    print("ERROR: HF_TOKEN not set. Add it to your .env file or export it:")
    sys.exit(1)

repo_id = f"{HF_USERNAME}/{DATASET_NAME}"
data_dir = Path(__file__).parent / "Data"/"outputs"

if not data_dir.exists():
    print(f"ERROR: Data directory not found at {data_dir}")
    sys.exit(1)

api = HfApi(token=token)

# Create the repo if it doesn't exist
try:
    create_repo(
        repo_id=repo_id,
        repo_type="dataset",
        private=True,   # set True if you want a private dataset
        token=token,
        exist_ok=True,
    )
    print(f"Repo ready: https://huggingface.co/datasets/{repo_id}")
except HfHubHTTPError as e:
    print(f"Failed to create repo: {e}")
    sys.exit(1)

# Files and patterns to skip
IGNORE = [
    ".DS_Store",
    "*.DS_Store",
    "__pycache__",
]

print(f"\nUploading {data_dir} -> {repo_id}")
print("This may take a while for large files. Progress is shown per file.\n")

api.upload_large_folder(
    repo_id=repo_id,
    repo_type="dataset",
    folder_path=str(data_dir),
    ignore_patterns=IGNORE,
    num_workers=4,              # parallel uploads; reduce to 2 on slow connections
)

print(f"\nDone. Dataset available at: https://huggingface.co/datasets/{repo_id}")
