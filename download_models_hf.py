"""
OPTIMIZED MODEL DOWNLOADER FOR RAILWAY
========================================
Downloads models from Hugging Face on startup.
This keeps your Docker image small!
"""

import os
import sys
from pathlib import Path

def download_from_huggingface():
    """
    Download models from Hugging Face Hub.
    
    SETUP:
    1. Create account at huggingface.co
    2. Install CLI: pip install huggingface_hub
    3. Login: huggingface-cli login
    4. Create repo: huggingface-cli repo create verityguard-models
    5. Upload models:
       huggingface-cli upload verityguard-models checkpoints/best_emotion_aware_detector.pth
       huggingface-cli upload verityguard-models best_model_safe.pt
    6. Replace REPO_ID below with your repo
    """
    
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("📦 Installing huggingface_hub...")
        os.system(f"{sys.executable} -m pip install huggingface_hub")
        from huggingface_hub import hf_hub_download
    
    print("="*70)
    print("🤗 DOWNLOADING MODELS FROM HUGGING FACE")
    print("="*70)
    
    # ⚠️ IMPORTANT: Replace with your Hugging Face username and repo name!
    REPO_ID = "sharonnnnn245/Multimodal-Deception-Framework-for-Fake-News-Detection"
    
    # Models to download
    models = {
        'checkpoints/best_emotion_aware_detector.pth': 'best_emotion_aware_detector.pth',
        'best_model_safe.pt': 'best_model_safe.pt',
        # Add more models if needed
    }
    
    success = True
    for local_path, hf_filename in models.items():
        # ✅ FIX: Create directory FIRST, before anything else
        directory = os.path.dirname(local_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
            print(f"✓ Ensured directory exists: {directory}")
        
        # ✅ FIX: NOW check if file already exists
        if os.path.exists(local_path):
            file_size = os.path.getsize(local_path) / (1024 * 1024)
            print(f"✓ {local_path} already exists ({file_size:.1f} MB)")
            continue
        
        try:
            print(f"\n📥 Downloading {hf_filename}...")
            print(f"   From: {REPO_ID}")
            
            # Download from Hugging Face
            downloaded_path = hf_hub_download(
                repo_id=REPO_ID,
                filename=hf_filename,
                cache_dir="./.cache/huggingface"
            )
            
            # Move to correct location
            import shutil
            shutil.move(downloaded_path, local_path)
            
            file_size = os.path.getsize(local_path) / (1024 * 1024)
            print(f"✓ Downloaded {local_path} ({file_size:.1f} MB)")
            
        except Exception as e:
            print(f"✗ Failed to download {hf_filename}: {e}")
            print("\n⚠️  Make sure you:")
            print("   1. Uploaded models to Hugging Face")
            print("   2. Updated REPO_ID in this script")
            print("   3. Made the repo public or set HF_TOKEN env variable")
            success = False
    
    print("\n" + "="*70)
    if success:
        print("✅ All models downloaded successfully!")
    else:
        print("⚠️  Some models failed. Check errors above.")
    print("="*70 + "\n")
    
    return success


def download_from_google_drive():
    """
    Alternative: Download from Google Drive direct links.
    
    SETUP:
    1. Upload models to Google Drive
    2. Share with "Anyone with the link"
    3. Get file ID from URL
    4. Use: https://drive.google.com/uc?export=download&id=FILE_ID
    """
    
    import urllib.request
    
    print("="*70)
    print("📥 DOWNLOADING MODELS FROM GOOGLE DRIVE")
    print("="*70)
    
    # ⚠️ IMPORTANT: Replace FILE_ID with your actual Google Drive file IDs!
    models = {
        'checkpoints/best_emotion_aware_detector.pth': 
            'https://drive.google.com/uc?export=download&id=YOUR_FILE_ID_1',
        'best_model_safe.pt': 
            'https://drive.google.com/uc?export=download&id=YOUR_FILE_ID_2'
    }
    
    success = True
    for local_path, url in models.items():
        if os.path.exists(local_path):
            file_size = os.path.getsize(local_path) / (1024 * 1024)
            print(f"✓ {local_path} already exists ({file_size:.1f} MB)")
            continue
        
        if 'YOUR_FILE_ID' in url:
            print(f"⚠️  Please configure Google Drive URL for {local_path}")
            success = False
            continue
        
        try:
            print(f"\n📥 Downloading {local_path}...")
            Path(local_path).parent.mkdir(parents=True, exist_ok=True)
            
            urllib.request.urlretrieve(url, local_path)
            
            file_size = os.path.getsize(local_path) / (1024 * 1024)
            print(f"✓ Downloaded {local_path} ({file_size:.1f} MB)")
            
        except Exception as e:
            print(f"✗ Failed: {e}")
            success = False
    
    return success


def verify_models():
    """Verify that models exist and seem valid."""
    
    print("\n🔍 Verifying models...")
    
    required = [
        'checkpoints/best_emotion_aware_detector.pth',
        'best_model_safe.pt'
    ]
    
    all_valid = True
    for path in required:
        if not os.path.exists(path):
            print(f"  ✗ Missing: {path}")
            all_valid = False
        else:
            size_mb = os.path.getsize(path) / (1024 * 1024)
            if size_mb < 1:
                print(f"  ⚠️  {path} too small ({size_mb:.2f} MB) - may be corrupt")
                all_valid = False
            else:
                print(f"  ✓ {path} ({size_mb:.1f} MB)")
    
    return all_valid


def main():
    """Main function - tries Hugging Face first, falls back to Google Drive."""
    
    print("\n" + "="*70)
    print("🚀 VERITYGUARD MODEL DOWNLOADER")
    print("="*70 + "\n")
    
    # Try Hugging Face first (recommended)
    print("Attempting download from Hugging Face...\n")
    if download_from_huggingface():
        if verify_models():
            print("\n✅ Setup complete! All models ready.\n")
            return 0
    
    # Fallback to Google Drive
    print("\n\nFalling back to Google Drive...\n")
    if download_from_google_drive():
        if verify_models():
            print("\n✅ Setup complete! All models ready.\n")
            return 0
    
    # If both fail
    print("\n" + "="*70)
    print("❌ MODEL DOWNLOAD FAILED")
    print("="*70)
    print("\nPlease:")
    print("1. Upload your models to Hugging Face (recommended) or Google Drive")
    print("2. Update the REPO_ID or FILE_IDs in this script")
    print("3. For Hugging Face: Make repo public or set HF_TOKEN env variable")
    print("4. For Google Drive: Make files accessible with 'Anyone with link'")
    print("\nSee RAILWAY_FIX.md for detailed instructions.")
    print("="*70 + "\n")
    
    return 1


if __name__ == "__main__":
    sys.exit(main())