# Run this as a Python script — save as upload_models.py
from huggingface_hub import HfApi
import os

api = HfApi()
token = "your_hf_token_here"
repo_id = "your_username/deceptionxai-models"

# Upload model files
files_to_upload = [
    ("checkpoints/best_emotion_aware_detector.pth", "best_emotion_aware_detector.pth"),
    ("anomaly_detection_results/anomaly_models.pt", "anomaly_models.pt"),
    ("platt_calibrator.pkl", "platt_calibrator.pkl"),
]

for local_path, repo_path in files_to_upload:
    print(f"Uploading {local_path}...")
    api.upload_file(
        path_or_fileobj=local_path,
        path_in_repo=repo_path,
        repo_id=repo_id,
        repo_type="model",
        token=token
    )
    print(f" {repo_path} uploaded")

print("All models uploaded successfully")