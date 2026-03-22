import pandas as pd
import numpy as np
import torch
import os


print("="*80)
print("STEP 1: INSPECTING YOUR DATASET")
print("="*80)

try:
    df = pd.read_pickle("Dataset/twitter/df_with_contradiction_scores.pkl")
    print(f" Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")
except FileNotFoundError:
    print(" ERROR: Could not find Dataset/twitter/df_with_contradiction_scores.pkl")
    print("\nAvailable pickle files:")
    for f in os.listdir("Dataset/twitter/"):
        if f.endswith('.pkl'):
            print(f"  - {f}")
    exit(1)

print(f"\nAll columns: {df.columns.tolist()}")

if 'label' in df.columns:
    print("\n Found 'label' column!")
    print(f"Label dtype: {df['label'].dtype}")
    print(f"Unique values: {df['label'].unique()}")
    print(f"\nValue counts:")
    print(df['label'].value_counts())
    print(f"\nFirst 20 labels: {df['label'].head(20).tolist()}")
else:
    print("\n NO 'label' column found!")
    print("\nSearching for columns that might contain labels...")
    
    potential_cols = [col for col in df.columns if any(keyword in col.lower() 
                      for keyword in ['label', 'class', 'fake', 'real', 'target', 'y', 'truth'])]
    
    if potential_cols:
        print(f"Found potential label columns: {potential_cols}")
        for col in potential_cols:
            print(f"\n{col}:")
            print(f"  Type: {df[col].dtype}")
            print(f"  Sample values: {df[col].head(10).tolist()}")
            print(f"  Unique values: {df[col].nunique()}")
    else:
        print(" No obvious label column found")


# STEP 2: ATTEMPT TO FIX LABELS


print("\n" + "="*80)
print("STEP 2: ATTEMPTING TO FIX LABELS")
print("="*80)

label_fixed = False

# Attempt 1: Try to convert existing labels
if 'label' in df.columns:
    print("\nAttempt 1: Converting existing 'label' column...")
    
    try:
        # Check if it's already numeric
        if pd.api.types.is_numeric_dtype(df['label']):
            df['label'] = df['label'].astype(int)
            print(f" Labels are already numeric!")
            label_fixed = True
        else:
            # Try mapping text labels
            print(f"  Labels are type: {df['label'].dtype}")
            print(f"  Attempting text-to-numeric mapping...")
            
            label_mapping = {
                'real': 0, 'Real': 0, 'REAL': 0, 'true': 0, 'True': 0, 'TRUE': 0,
                'fake': 1, 'Fake': 1, 'FAKE': 1, 'false': 1, 'False': 1, 'FALSE': 1,
                0: 0, 1: 1, '0': 0, '1': 1,
                'negative': 0, 'positive': 1, 'ham': 0, 'spam': 1,
                'legitimate': 0, 'fraudulent': 1, 'honest': 0, 'deceptive': 1
            }
            
            df['label'] = df['label'].map(label_mapping)
            
            if df['label'].isna().sum() > 0:
                print(f"  ⚠️  {df['label'].isna().sum()} labels could not be mapped")
                unmapped = df[df['label'].isna()]['label'].unique()
                print(f"  Unmapped values: {unmapped}")
            else:
                df['label'] = df['label'].astype(int)
                print(f"   Successfully mapped text labels to 0/1!")
                label_fixed = True
                
    except Exception as e:
        print(f"   Conversion failed: {e}")

if not label_fixed:
    print(" ERROR: No real ground-truth labels found.")
    print("You CANNOT train a scientific model without real labels.")
    print("Please use a labeled dataset like:")
    print("- FakeNewsNet")
    print("- LIAR")
    print("- PolitiFact / GossipCop")
    exit(1)

# STEP 3: VALIDATE AND SAVE


print("\n" + "="*80)
print("STEP 3: VALIDATING AND SAVING")
print("="*80)

if label_fixed and 'label' in df.columns:
    # Remove NaN labels
    original_len = len(df)
    df = df.dropna(subset=['label'])
    if len(df) < original_len:
        print(f"  Dropped {original_len - len(df)} rows with NaN labels")
    
    # Ensure integer type
    df['label'] = df['label'].astype(int)
    
    # Validate binary
    unique_labels = df['label'].unique()
    print(f"\nUnique label values: {sorted(unique_labels)}")
    
    if set(unique_labels) == {0, 1}:
        print(" Labels are in correct binary format (0 = Real, 1 = Fake)")
        
        # Show distribution
        print(f"\nFinal label distribution:")
        counts = df['label'].value_counts().sort_index()
        print(f"  Real (0):  {counts[0]} samples ({counts[0]/len(df)*100:.1f}%)")
        print(f"  Fake (1):  {counts[1]} samples ({counts[1]/len(df)*100:.1f}%)")
        
        balance = counts.min() / counts.max()
        print(f"  Balance ratio: {balance:.3f}")
        
        if balance < 0.5:
            print("  ⚠️  Dataset is imbalanced - consider using class weights in training")
        
        # Save corrected dataset
        output_path = "Dataset/twitter/df_with_corrected_labels.pkl"
        df.to_pickle(output_path)
        print(f"\n Saved corrected dataset to: {output_path}")
        
        # Save labels as tensor
        labels_tensor = torch.tensor(df['label'].values, dtype=torch.long)
        labels_path = "Dataset/twitter/labels.pt"
        torch.save(labels_tensor, labels_path)
        print(f" Saved label tensor to: {labels_path}")
        
        print("\n" + "="*80)
        print(" SUCCESS! LABELS ARE READY ")
        print("="*80)
        
        print("\n NEXT STEPS:")
        print("1. Run diagnostics:")
        print("   python root_cause_analyzer.py")
        print("\n2. If diagnostics pass, start training:")
        print("   python fixed_training.py")
        
        print("\n  IMPORTANT: Update your training script to load labels:")
        print("   labels = torch.load('Dataset/twitter/labels.pt').float()")
        
    else:
        print(f" ERROR: Expected labels to be 0 and 1, found: {unique_labels}")
        print("Manual intervention required!")
else:
    print("\n FAILED TO CREATE VALID LABELS")
   