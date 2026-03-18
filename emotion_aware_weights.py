import torch
from rough_work import EmotionAwareFakeNewsDetector  

emotion_ckpt_path = "checkpoints/best_emotion_aware_detector.pth"


ckpt = torch.load(emotion_ckpt_path, map_location="cpu")
print(" Checkpoint loaded")
for i, k in enumerate(list(ckpt.keys())[:20]):
    print(f"{i+1}. {k} -> {ckpt[k].shape if isinstance(ckpt[k], torch.Tensor) else type(ckpt[k])}")


emotion_model = EmotionAwareFakeNewsDetector(
    d_text=128,
    d_image=1024,
    d_meta=128,
    d_common=256,
    vad_dim=3,
    meta_affective_dim=128,
    mismatch_dim=128,
    temporal_hidden=64,
    num_classes=1
)
model_keys = list(emotion_model.state_dict().keys())
for i, k in enumerate(model_keys[:20]):
    print(f"{i+1}. {k} -> {emotion_model.state_dict()[k].shape}")
    

matching_keys = [k for k in ckpt if k in emotion_model.state_dict()]
print(f"\n Matching keys in checkpoint and model: {len(matching_keys)}/{len(model_keys)}")
