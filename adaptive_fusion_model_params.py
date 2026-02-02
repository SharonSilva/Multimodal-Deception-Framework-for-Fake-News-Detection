"""
inspect_fusion_model.py
=======================
Inspect the AdaptiveMultimodalFakeNewsDetector to see what layers and attributes exist.
"""

import torch
from multimodal_fakenews_model import AdaptiveMultimodalFakeNewsDetector

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1️⃣ Instantiate the model
fusion_model = AdaptiveMultimodalFakeNewsDetector(
    d_text=64, d_image=64, d_meta=64, d_common=256
).to(device)

# 2️⃣ Load state_dict if available
try:
    fusion_model.load_state_dict(torch.load("best_model_safe.pt", map_location=device))
    print("✅ Loaded state_dict successfully!")
except FileNotFoundError:
    print("⚠️ best_model_safe.pt not found. Model loaded with random weights.")

fusion_model.eval()  # set to eval mode

# 3️⃣ Print full model structure
print("\n=== Full Model Structure ===")
print(fusion_model)

# 4️⃣ Print all instance variables
print("\n=== Instance Variables ===")
for name, param in fusion_model.named_parameters():
    print(name, param.shape)

# 5️⃣ List all attributes of the object
print("\n=== All Attributes (dir) ===")
print([attr for attr in dir(fusion_model) if not attr.startswith("__")])

# 6️⃣ List all linear layers (most likely projections)
print("\n=== Linear Layers ===")
for name, layer in fusion_model.named_modules():
    if isinstance(layer, torch.nn.Linear):
        print(name, layer)
