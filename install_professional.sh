#!/bin/bash
# Professional AI Image Studio Installation Script
# Optimized for income generation on Android/Termux

echo "🚀 AI Image Studio - Professional Installation"
echo "=============================================="
echo ""

# Function to test if a package can be imported
test_import() {
    python3 -c "import $1" 2>/dev/null
    return $?
}

# Function to install with fallback
install_with_fallback() {
    local package=$1
    local fallback=$2
    
    echo "📦 Installing $package..."
    if pip install "$package"; then
        echo "✅ $package installed successfully"
        return 0
    else
        echo "❌ $package failed, trying fallback: $fallback"
        if pip install "$fallback"; then
            echo "✅ $fallback installed successfully"
            return 0
        else
            echo "❌ Both $package and $fallback failed"
            return 1
        fi
    fi
}

echo "🔧 Step 1: Installing core requirements..."
pip install -r requirements.txt

echo ""
echo "🎯 Step 2: Installing Professional Real-SR (for income generation)..."

# Try to install Real-SR components
if install_with_fallback "realesrgan==0.3.0" "realesrgan==0.2.5"; then
    REALSR_SUCCESS=true
else
    REALSR_SUCCESS=false
fi

if install_with_fallback "basicsr==1.4.2" "basicsr==1.3.5"; then
    BASICSR_SUCCESS=true
else
    BASICSR_SUCCESS=false
fi

echo ""
echo "🔍 Step 3: Testing OpenCV compatibility..."

# Test OpenCV installation
if test_import "cv2"; then
    echo "✅ OpenCV is available"
    OPENCV_AVAILABLE=true
else
    echo "⚠️  OpenCV not available - trying to install..."
    OPENCV_AVAILABLE=false
    
    # Try different OpenCV versions
    if install_with_fallback "opencv-python-headless==4.8.1.78" "opencv-python-headless==4.5.5.64"; then
        if test_import "cv2"; then
            echo "✅ OpenCV installed successfully"
            OPENCV_AVAILABLE=true
        fi
    fi
fi

echo ""
echo "🧪 Step 4: Running compatibility tests..."
python3 test_opencv.py

echo ""
echo "📊 Installation Summary:"
echo "======================="

if [ "$REALSR_SUCCESS" = true ] && [ "$BASICSR_SUCCESS" = true ]; then
    echo "✅ Real-SR: AVAILABLE (Professional income generation ready!)"
    PROFESSIONAL_MODE=true
else
    echo "❌ Real-SR: NOT AVAILABLE (Will use traditional upscaling)"
    PROFESSIONAL_MODE=false
fi

if [ "$OPENCV_AVAILABLE" = true ]; then
    echo "✅ OpenCV: AVAILABLE (Enhanced image processing)"
else
    echo "⚠️  OpenCV: NOT AVAILABLE (Will use PIL fallback)"
fi

echo ""
if [ "$PROFESSIONAL_MODE" = true ]; then
    echo "🎉 PROFESSIONAL MODE ENABLED!"
    echo "   💰 Your AI Image Studio is ready for income generation"
    echo "   🎯 Real-SR AI upscaling available"
    echo "   📈 Market-competitive image quality"
else
    echo "⚠️  BASIC MODE ONLY"
    echo "   📝 Traditional upscaling available"
    echo "   💡 Consider fixing Real-SR installation for better income potential"
fi

echo ""
echo "🚀 Ready to start your AI Image Studio!"
echo "   Run: python3 app.py"
echo "   Then open: http://localhost:5000"
echo ""
echo "💡 For troubleshooting, see: REALSR_PROFESSIONAL_SETUP.md"