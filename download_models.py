from huggingface_hub import hf_hub_download
import os

token = os.environ.get("HF_TOKEN")
repo_id = os.environ.get("HF_REPO_ID", "sharonnnnn245/deceptionxai-models")

os.makedirs("checkpoints", exist_ok=True)
os.makedirs("anomaly_detection_results", exist_ok=True)

print("Downloading models from Hugging Face...")

hf_hub_download(
    repo_id=repo_id,
    filename="best_emotion_aware_detector.pth",
    local_dir="checkpoints",
    token=token,
    repo_type="model"
)
print("✅ best_emotion_aware_detector.pth")

hf_hub_download(
    repo_id=repo_id,
    filename="anomaly_models.pt",
    local_dir="anomaly_detection_results",
    token=token,
    repo_type="model"
)
print("✅ anomaly_models.pt")

hf_hub_download(
    repo_id=repo_id,
    filename="platt_calibrator.pkl",
    local_dir=".",
    token=token,
    repo_type="model"
)
print("✅ platt_calibrator.pkl")

print("All models ready")