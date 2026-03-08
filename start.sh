#!/bin/bash
set -e

echo "Starting Ollama..."
ollama serve &

echo "Waiting for Ollama to be ready..."
sleep 10

echo "Pulling moondream..."
ollama pull moondream

echo "Pulling llama3.2..."
ollama pull llama3.2

echo "Ollama models ready"

echo "Downloading model files from Hugging Face..."
python /app/download_models.py

echo "Starting Flask app..."
python app.py