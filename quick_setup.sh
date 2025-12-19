#!/bin/bash
# Quick Setup - Just Get Your AI Image Studio Running!

echo "🚀 AI Image Studio - Quick Setup"
echo "================================"

# Install core dependencies (always works)
echo "📦 Installing core dependencies..."
pip install Flask==2.3.3 Pillow numpy requests openai python-dotenv

# Check if app.py exists
if [ -f "app.py" ]; then
    echo "✅ Found app.py - starting your AI Image Studio!"
    echo ""
    echo "🎉 Your AI Image Studio is ready!"
    echo "   Run: python app.py"
    echo "   Then open: http://localhost:5000"
    echo ""
    echo "💰 Start generating income with AI images!"
else
    echo "⚠️  app.py not found"
    echo "   Make sure you're in the AI_Image_Studio directory"
    echo "   Run: git clone https://github.com/shaneabel36/AI_Image_Studio.git"
fi