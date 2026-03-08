FROM python:3.10-slim

# Install system dependencies including curl for Ollama
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Ollama
RUN curl -fsSL https://ollama.com/install.sh | sh

# Set working directory
WORKDIR /app

# Copy requirements and install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create startup script
COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh

# Hugging Face Spaces requires port 7860
EXPOSE 7860

CMD ["/app/start.sh"]