import os
import sys
from pathlib import Path

def download_from_huggingface():
    """
    Download models from Hugging Face Hub.
    """
    
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print(" Installing huggingface_hub...")
        os.system(f"{sys.executable} -m pip install huggingface_hub")
        from huggingface_hub import hf_hub_download
    
    print("="*70)
    print(" DOWNLOADING MODELS FROM HUGGING FACE")
    print("="*70)
    

    REPO_ID = "sharonnnnn245/Multimodal-Deception-Framework-for-Fake-News-Detection"
    

    models = {
        'checkpoints/best_emotion_aware_detector.pth': 'best_emotion_aware_detector.pth',
        'best_model_safe.pt': 'best_model_safe.pt',
    }
    
    success = True
    for local_path, hf_filename in models.items():

        directory = os.path.dirname(local_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
            print(f" Ensured directory exists: {directory}")

        if os.path.exists(local_path):
            file_size = os.path.getsize(local_path) / (1024 * 1024)
            print(f" {local_path} already exists ({file_size:.1f} MB)")
            continue
        
        try:
            print(f"\n Downloading {hf_filename}...")
            print(f"   From: {REPO_ID}")
            
            # Download from Hugging Face
            downloaded_path = hf_hub_download(
                repo_id=REPO_ID,
                filename=hf_filename,
                cache_dir="./.cache/huggingface"
            )
            
            # Copy to correct location
            import shutil
            shutil.copy(downloaded_path, local_path)
            
            file_size = os.path.getsize(local_path) / (1024 * 1024)
            print(f" Downloaded {local_path} ({file_size:.1f} MB)")
            
        except Exception as e:
            print(f"✗ Failed to download {hf_filename}: {e}")
            print("\n  Make sure you:")
            print("   1. Uploaded models to Hugging Face")
            print("   2. Repo exists and is public: https://huggingface.co/" + REPO_ID)
            print("   3. Files are named: 'best_emotion_aware_detector.pth' and 'best_model_safe.pt'")
            success = False
    
    print("\n" + "="*70)
    if success:
        print(" All models downloaded successfully!")
    else:
        print("  Some models failed. Check errors above.")
    print("="*70 + "\n")
    
    return success


def verify_models():
    """Verify that models exist and seem valid."""
    
    print("\n Verifying models...")
    
    required = [
        'checkpoints/best_emotion_aware_detector.pth',
        'best_model_safe.pt'
    ]
    
    all_valid = True
    for path in required:
        if not os.path.exists(path):
            print(f"   Missing: {path}")
            all_valid = False
        else:
            size_mb = os.path.getsize(path) / (1024 * 1024)
            if size_mb < 1:
                print(f"    {path} too small ({size_mb:.2f} MB) - may be corrupt")
                all_valid = False
            else:
                print(f"   {path} ({size_mb:.1f} MB)")
    
    return all_valid


def main():
    """Main function."""
    
    print("\n" + "="*70)
    print(" VERITYGUARD MODEL DOWNLOADER")
    print("="*70 + "\n")
    
    # Try downloading
    if download_from_huggingface():
        if verify_models():
            print("\n Setup complete! All models ready.\n")
            return 0
    
    # If download failed, check if models already exist
    print("\nChecking if models are already present...")
    if verify_models():
        print("\n Models already available! Continuing.\n")
        return 0
    
    # If both fail, continue anyway (don't block startup)
    print("\n" + "="*70)
    print("  MODEL DOWNLOAD INCOMPLETE")
    print("="*70)
    print("\nApplication will start, but models may not work.")
    print("\nTo fix, upload models to:")
    print("https://huggingface.co/sharonnnnn245/Multimodal-Deception-Framework-for-Fake-News-Detection")
    print("\nFiles needed:")
    print("  - best_emotion_aware_detector.pth")
    print("  - best_model_safe.pt")
    print("="*70 + "\n")
    
    # Return 0 so app continues startup
    return 0


if __name__ == "__main__":
    sys.exit(main())