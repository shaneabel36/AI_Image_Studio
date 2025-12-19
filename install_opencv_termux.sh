#!/bin/bash
# OpenCV Installation Script for Termux
# Run this script on your Android tablet in Termux

echo "🔧 Installing OpenCV for Termux..."

# Update packages
pkg update && pkg upgrade -y

# Install system dependencies
pkg install -y python python-pip clang cmake ninja libjpeg-turbo libpng libtiff libwebp

# Install Python dependencies first
pip install --upgrade pip setuptools wheel

# Try different OpenCV installation methods
echo "📦 Attempting Method 1: Standard pip install..."
pip install opencv-python-headless==4.8.1.78

if python -c "import cv2; print('✅ OpenCV installed successfully')" 2>/dev/null; then
    echo "✅ OpenCV installation successful!"
    python -c "import cv2; print('OpenCV version:', cv2.__version__)"
    exit 0
fi

echo "❌ Method 1 failed. Trying Method 2: Headless version..."
pip uninstall opencv-python opencv-python-headless -y
pip install opencv-contrib-python-headless==4.8.1.78

if python -c "import cv2; print('✅ OpenCV installed successfully')" 2>/dev/null; then
    echo "✅ OpenCV installation successful!"
    python -c "import cv2; print('OpenCV version:', cv2.__version__)"
    exit 0
fi

echo "❌ Method 2 failed. Trying Method 3: Older version..."
pip uninstall opencv-python opencv-python-headless opencv-contrib-python-headless -y
pip install opencv-python-headless==4.5.5.64

if python -c "import cv2; print('✅ OpenCV installed successfully')" 2>/dev/null; then
    echo "✅ OpenCV installation successful!"
    python -c "import cv2; print('OpenCV version:', cv2.__version__)"
    exit 0
fi

echo "❌ All methods failed. OpenCV may not be compatible with your device."
echo "💡 Consider using the OpenCV-free version of the AI Image Studio."