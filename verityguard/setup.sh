#!/bin/bash

# VerityGuard - Quick Start Script
# This script helps you get started quickly with VerityGuard

set -e

echo "========================================"
echo "  VerityGuard - Quick Start Setup"
echo "========================================"
echo ""

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check Python version
echo "Checking Python installation..."
if command_exists python3; then
    PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
    echo "✓ Python $PYTHON_VERSION found"
else
    echo "✗ Python 3 is required but not installed"
    exit 1
fi

# Create virtual environment
echo ""
echo "Creating virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment already exists"
fi

# Activate virtual environment
echo ""
echo "Activating virtual environment..."
source venv/bin/activate || . venv/Scripts/activate
echo "✓ Virtual environment activated"

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt
echo "✓ Dependencies installed"

# Create directories
echo ""
echo "Creating necessary directories..."
mkdir -p uploads results checkpoints templates
echo "✓ Directories created"

# Check for model files
echo ""
echo "Checking for model checkpoints..."
if [ -f "checkpoints/best_emotion_aware_detector.pth" ]; then
    echo "✓ Emotion model found"
else
    echo "⚠ Emotion model not found at checkpoints/best_emotion_aware_detector.pth"
    echo "  Please place your trained model file there"
fi

if [ -f "best_model_safe.pt" ]; then
    echo "✓ Fusion model found"
else
    echo "⚠ Fusion model not found at best_model_safe.pt"
    echo "  Please place your trained model file there"
fi

# Print usage instructions
echo ""
echo "========================================"
echo "  Setup Complete!"
echo "========================================"
echo ""
echo "To start the server:"
echo "  python app.py"
echo ""
echo "To process a batch CSV:"
echo "  python batch_processor.py sample_posts.csv"
echo ""
echo "To run with Docker:"
echo "  docker-compose up -d"
echo ""
echo "Web interface will be available at:"
echo "  http://localhost:5001"
echo ""
echo "========================================"
echo ""