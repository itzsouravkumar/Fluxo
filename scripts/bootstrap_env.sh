#!/bin/bash
set -e

echo "Setting up FLUXO development environment..."

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install Python dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Create necessary directories
mkdir -p models data outputs clips
touch models/.gitkeep data/.gitkeep outputs/.gitkeep

# Copy env file
if [ ! -f .env ]; then
    cp .env.example .env
    echo ".env created from .env.example"
fi

# Download YOLOv11 weights
python -c "from ultralytics import YOLO; YOLO('yolo11n.pt')" 2>/dev/null || echo "YOLO weights download skipped"

# Install dashboard dependencies
if [ -d dashboard ]; then
    cd dashboard
    npm install
    cd ..
fi

echo ""
echo "FLUXO environment ready!"
echo ""
echo "Quick start:"
echo "  make dev        # Start development stack"
echo "  make demo       # Start full demo"
echo "  make test       # Run tests"
echo "  make help       # See all commands"
