#!/bin/bash
set -e

echo "ResisTrack Setup Script"
echo "======================="

# Python Setup
if ! command -v python3 &> /dev/null; then
    echo "Error: python3 could not be found."
    exit 1
fi

echo "Setting up Python environment..."
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
    echo "Created virtual environment in .venv"
fi

source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .

echo "Python setup complete."

# Frontend Setup
if [ -d "dashboard" ]; then
    echo "Setting up Frontend environment..."
    if ! command -v npm &> /dev/null; then
        echo "Warning: npm could not be found. Skipping frontend setup."
    else
        cd dashboard
        npm install
        cd ..
        echo "Frontend setup complete."
    fi
else
    echo "Warning: dashboard directory not found."
fi

# Infra Setup (optional)
if [ -d "infra" ]; then
    echo "Setting up Infra environment (CDK)..."
    if ! command -v npm &> /dev/null; then
        echo "Warning: npm could not be found. Skipping infra setup."
    else
        cd infra
        npm install
        cd ..
        echo "Infra setup complete."
    fi
fi

echo "======================="
echo "Setup complete!"
echo "To activate the environment, run: source .venv/bin/activate"
