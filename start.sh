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

echo "Starting Flask app..."
exec python app.py