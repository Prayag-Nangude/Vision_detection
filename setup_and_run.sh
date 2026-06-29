#!/bin/bash
# Setup and Run script for Linux/macOS

# 1. Check Python Version
echo "Checking Python version..."
if command -v python3.12 &>/dev/null; then
    echo "Python 3.12 found."
    PYTHON_CMD="python3.12"
elif python3 --version 2>&1 | grep -q "3.12"; then
    echo "Python 3.12 found."
    PYTHON_CMD="python3"
else
    echo "Warning: Python 3.12.x is required but not found. Please install Python 3.12."
    exit 1
fi

# 2. Create required directories
echo "Creating log, screenshot, and recording directories..."
mkdir -p logs screenshots recordings

# 3. Create virtual environment
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment .venv..."
    $PYTHON_CMD -m venv .venv
else
    echo "Existing .venv found, skipping creation."
fi

# 4. Install dependencies
echo "Installing dependencies from requirements.txt..."
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt

# 5. Start application
echo "Starting the application..."
.venv/bin/python main.py
