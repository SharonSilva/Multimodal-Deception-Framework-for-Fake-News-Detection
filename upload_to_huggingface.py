"""
UPLOAD MODELS TO HUGGING FACE
==============================
Run this script to upload your model files to Hugging Face Hub.

BEFORE RUNNING:
1. Install: pip install huggingface_hub
2. Create account at huggingface.co
3. Run this script
"""

import os
import sys

def main():
    print("="*70)
    print("🤗 HUGGING FACE MODEL UPLOADER")
    print("="*70)
    print()
    
    # Check if huggingface_hub is installed
    try:
        from huggingface_hub import HfApi, login
    except ImportError:
        print("❌ huggingface_hub not installed!")
        print()
        print("Installing now...")
        os.system(f"{sys.executable} -m pip install huggingface_hub")
        from huggingface_hub import HfApi, login
        print("✅ Installed!")
        print()
    
    # Step 1: Login
    print("STEP 1: Login to Hugging Face")
    print("-" * 70)
    print()
    print("You'll need to:")
    print("1. Go to: https://huggingface.co/settings/tokens")
    print("2. Click 'New token'")
    print("3. Name: 'verityguard-upload'")
    print("4. Type: 'Write' access")
    print("5. Copy the token")
    print()
    
    try:
        login()
        print("✅ Logged in successfully!")
        print()
    except Exception as e:
        print(f"❌ Login failed: {e}")
        print()
        print("Please make sure you:")
        print("1. Created a Hugging Face account at huggingface.co")
        print("2. Generated an access token with 'Write' permissions")
        print("3. Pasted the token when prompted")
        return 1
    
    # Step 2: Get repository name
    print("STEP 2: Repository Setup")
    print("-" * 70)
    print()
    
    username = input("Enter your Hugging Face username: ").strip()
    if not username:
        print("❌ Username required!")
        return 1
    
    repo_name = input("Enter repository name [verityguard-models]: ").strip() or "verityguard-models"
    repo_id = f"{username}/{repo_name}"
    
    print()
    print(f"📦 Repository: {repo_id}")
    print(f"🔗 URL: https://huggingface.co/{repo_id}")
    print()
    
    # Step 3: Create repository
    print("STEP 3: Creating Repository")
    print("-" * 70)
    print()
    
    api = HfApi()
    
    try:
        api.create_repo(
            repo_id=repo_id,
            repo_type="model",
            exist_ok=True,
            private=False
        )
        print(f"✅ Repository '{repo_id}' created/verified")
        print()
    except Exception as e:
        print(f"❌ Failed to create repo: {e}")
        return 1
    
    # Step 4: Check for model files
    print("STEP 4: Checking Model Files")
    print("-" * 70)
    print()
    
    models = [
        'checkpoints/best_emotion_aware_detector.pth',
        'best_model_safe.pt'
    ]
    
    files_to_upload = []
    for model_path in models:
        if os.path.exists(model_path):
            size_mb = os.path.getsize(model_path) / (1024 * 1024)
            print(f"✅ Found: {model_path} ({size_mb:.1f} MB)")
            files_to_upload.append(model_path)
        else:
            print(f"⚠️  Not found: {model_path}")
    
    print()
    
    if not files_to_upload:
        print("❌ No model files found!")
        print()
        print("Please make sure you have:")
        print("  - checkpoints/best_emotion_aware_detector.pth")
        print("  - best_model_safe.pt")
        print()
        return 1
    
    print(f"📤 Ready to upload {len(files_to_upload)} file(s)")
    print()
    
    # Confirm upload
    confirm = input("Proceed with upload? (yes/no): ").strip().lower()
    if confirm not in ['yes', 'y']:
        print("❌ Upload cancelled")
        return 0
    
    print()
    
    # Step 5: Upload files
    print("STEP 5: Uploading Files")
    print("-" * 70)
    print()
    
    uploaded = 0
    failed = 0
    
    for local_path in files_to_upload:
        filename = os.path.basename(local_path)
        
        print(f"📤 Uploading: {filename}...")
        
        try:
            api.upload_file(
                path_or_fileobj=local_path,
                path_in_repo=filename,
                repo_id=repo_id,
                repo_type="model"
            )
            print(f"   ✅ Uploaded successfully!")
            uploaded += 1
        except Exception as e:
            print(f"   ❌ Failed: {e}")
            failed += 1
        
        print()
    
    # Step 6: Summary
    print("="*70)
    print("📊 UPLOAD SUMMARY")
    print("="*70)
    print()
    print(f"✅ Uploaded: {uploaded} file(s)")
    if failed > 0:
        print(f"❌ Failed: {failed} file(s)")
    print()
    print(f"🔗 View at: https://huggingface.co/{repo_id}")
    print()
    
    # Step 7: Next steps
    print("="*70)
    print("📝 NEXT STEPS")
    print("="*70)
    print()
    print("1. Update download_models.py:")
    print(f"   Change: REPO_ID = \"your-username/verityguard-models\"")
    print(f"   To:     REPO_ID = \"{repo_id}\"")
    print()
    print("2. Test download (optional):")
    print("   python download_models.py")
    print()
    print("3. Commit and push to Railway:")
    print("   git add .")
    print("   git commit -m 'Add Hugging Face model download'")
    print("   git push")
    print()
    print("="*70)
    print()
    
    if uploaded == len(files_to_upload):
        print("🎉 All files uploaded successfully!")
        return 0
    else:
        print("⚠️  Some files failed to upload. Please check errors above.")
        return 1


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n❌ Upload cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)