# Start from lightweight Python image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    wget \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for layer caching)
COPY verityguard/requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY verityguard/ ./verityguard/

# Create necessary directories
RUN mkdir -p verityguard/uploads verityguard/results verityguard/checkpoints

# Expose port for Render
EXPOSE 5001

# Set environment variables
ENV FLASK_APP=verityguard.app
ENV FLASK_ENV=production
ENV PYTHONUNBUFFERED=1
ENV PORT=5001

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:5001/api/health')"

# Start the app with Gunicorn
CMD ["gunicorn", "verityguard.app:app", "--bind", "0.0.0.0:5001", "--workers", "1", "--threads", "2", "--timeout", "300", "--max-requests", "100", "--max-requests-jitter", "10", "--log-level", "info", "--access-logfile", "-", "--error-logfile", "-"]
