FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    curl \
    wget \
    git \
    zstd \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    llvm \
    procps \
    && rm -rf /var/lib/apt/lists/*

# Install Ollama
RUN curl -fsSL https://ollama.com/install.sh | sh

# Pull models at build time while network is available
RUN ollama serve & sleep 5 && \
    ollama pull moondream && \
    ollama pull llama3.2:1b && \
    kill $(pgrep ollama) 2>/dev/null || true

WORKDIR /app

# Stage 1 — PyTorch CPU (heaviest, ~800MB, cached independently)
RUN pip install --no-cache-dir \
    torch==2.1.0+cpu \
    torchvision==0.16.0+cpu \
    torchaudio==2.1.0+cpu \
    --extra-index-url https://download.pytorch.org/whl/cpu

# Stage 2 — ML/AI packages
RUN pip install --no-cache-dir \
    transformers==4.36.2 \
    open-clip-torch>=2.20.0 \
    sentence-transformers>=2.2.0 \
    scikit-learn==1.8.0 \
    scipy>=1.10 \
    umap-learn

# Stage 3 — App packages
RUN pip install --no-cache-dir \
    Flask==3.1.0 \
    flask-cors==6.0.2 \
    gunicorn>=21.2.0 \
    numpy==1.26.4 \
    pandas>=2.0 \
    opencv-python-headless>=4.10 \
    Pillow>=10.0 \
    emoji==2.2.0 \
    python-dotenv==1.1.1 \
    "huggingface_hub>=0.19.3,<1.0"

# Download Python models at build time
COPY download_models.py .
RUN python download_models.py

COPY . .
COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh

EXPOSE 7860

CMD ["/app/start.sh"]
