#!/bin/bash
set -e

echo "===== Application Startup at $(date) ====="

echo "Starting Ollama..."
ollama serve &

echo "Waiting for Ollama to be ready..."
for i in $(seq 1 60); do
    if curl -s http://127.0.0.1:11434 > /dev/null 2>&1; then
        echo "Ollama is ready after ${i}s"
        break
    fi
    sleep 1
done

echo "Pulling moondream..."
ollama pull moondream

echo "Pulling llama3.2:1b..."
ollama pull llama3.2:1b

echo "Downloading model files from Hugging Face..."
python /app/download_models.py

echo "Starting Flask app..."
exec python app.py