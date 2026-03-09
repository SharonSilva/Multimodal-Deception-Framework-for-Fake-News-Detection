FROM python:3.10-slim

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
    && rm -rf /var/lib/apt/lists/*

# Install Ollama first
RUN curl -fsSL https://ollama.com/install.sh | sh

# Then pull models at build time while network is available
RUN ollama serve & sleep 5 && \
    ollama pull moondream && \
    ollama pull llama3.2:1b && \
    pkill ollama || true

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download Python models at build time too
COPY download_models.py .
RUN python download_models.py

COPY . .
COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh

EXPOSE 7860

CMD ["/app/start.sh"]