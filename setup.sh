#!/bin/bash
set -e

# Turnkey setup script for renting fresh vast.ai instances

echo "======================================"
echo " Starting Setup for Robust Reasoning"
echo "======================================"

echo "1. Upgrading PIP..."
pip install --upgrade pip

echo "2. Installing requirements..."
pip install -r requirements.txt

echo "3. Downloading language models for SpaCy and NLTK..."
python -m spacy download en_core_web_sm
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"

echo "======================================"
echo " Setup Complete!"
echo "======================================"
echo "Don't forget to:"
echo "  1) Add your API keys to the .env file"
echo "  2) Authenticate with Hugging Face via: huggingface-cli login"
