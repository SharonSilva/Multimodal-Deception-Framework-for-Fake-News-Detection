# Multimodal Deception XAI — Fake News Detection Framework

This repository contains the implementation of the final year research project, 
a Multimodal Deception Framework for Fake News Detection leveraging cross-modal 
Valence-Arousal-Dominance (VAD) emotional incongruence as a manipulation signal, 
with a full Explainable AI (XAI) narrative layer.

## Live Demo
🚀 [HuggingFace Spaces Deployment](https://huggingface.co/spaces/sharonnnnn245/deceptionxai)

## Features
- Cross-modal VAD mismatch detection between text and image modalities
- EmotionAwareFakeNewsDetector — custom trained neural network (925,638 parameters)
- CLIP ViT-L/14 zero-shot image VAD scoring and entity consistency checking
- Anomaly detection ensemble (Entropy, VAD Mismatch Magnitude, Variance, Kurtosis)
- GDELT real-world event verification with dynamic threshold adjustment
- Ollama LLM entity consistency checker (llama3.2)
- Full XAI narrative generation explaining each verdict
- React frontend with VAD radar, mismatch bars, and pipeline visualisation

## Requirements
- Python 3.11 or later
- Node.js 18 or later (for React frontend)
- Ollama with llama3.2 and moondream models
- Dependencies listed in requirements.txt

## Installation
```bash
git clone https://github.com/sharonnnnn245/Multimodal-Deception-Framework-for-Fake-News-Detection
cd Multimodal-Deception-Framework-for-Fake-News-Detection
pip install -r requirements.txt
cd xai-demo && npm install && npm run build
```

## Usage
```bash
# Start Ollama
ollama serve &
ollama pull llama3.2
ollama pull moondream

# Run the Flask backend
python app.py
```

## Dataset
Trained and evaluated on the MediaEval Verifying Multimedia Use (VMU) benchmark 
dataset — 11,844 Twitter posts from Hurricane Sandy, Nepal Earthquake, Paris 
Attacks, and Solar Eclipse events.

## Results
| Metric | Score |
|--------|-------|
| Accuracy | 94.76% |
| F1-Score | 0.9482 |
| AUC-ROC | 0.9823 |
| Precision | 0.9326 |
| Recall | 0.9643 |
