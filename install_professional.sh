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
echo "🎯 Step 2: Checking for professional upscaling options..."

# Check if user wants heavy ML dependencies
echo "Choose your professional setup:"
echo "1. Lightweight Professional (API-based, ~50MB) - RECOMMENDED for Android"
echo "2. Full AI Professional (Real-SR, ~2-3GB) - Best quality but heavy"
echo "3. Skip professional features for now"
echo ""
read -p "Enter choice (1-3) [1]: " CHOICE
CHOICE=${CHOICE:-1}

REALSR_SUCCESS=false
API_UPSCALER_SUCCESS=true  # Always available (no dependencies)

if [ "$CHOICE" = "2" ]; then
    echo "Installing Full AI Professional (Real-SR)..."
    echo "⚠️  This will download ~2-3GB of PyTorch dependencies"
    
    # Try to install Real-SR components
    if install_with_fallback "realesrgan==0.3.0" "realesrgan==0.2.5"; then
        if install_with_fallback "basicsr==1.4.2" "basicsr==1.3.5"; then
            REALSR_SUCCESS=true
            echo "✅ Full AI Professional installed"
        fi
    fi
    
    if [ "$REALSR_SUCCESS" = false ]; then
        echo "❌ Full AI installation failed, falling back to Lightweight Professional"
    fi
elif [ "$CHOICE" = "1" ]; then
    echo "✅ Lightweight Professional selected - no heavy dependencies needed"
else
    echo "⚠️  Skipping professional features"
    API_UPSCALER_SUCCESS=false
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

if [ "$REALSR_SUCCESS" = true ]; then
    echo "✅ Full AI Professional: AVAILABLE (Real-SR with PyTorch)"
    PROFESSIONAL_MODE="full"
elif [ "$API_UPSCALER_SUCCESS" = true ]; then
    echo "✅ Lightweight Professional: AVAILABLE (API-based upscaling)"
    PROFESSIONAL_MODE="lightweight"
else
    echo "❌ Professional Features: NOT AVAILABLE (Basic mode only)"
    PROFESSIONAL_MODE="basic"
fi

if [ "$OPENCV_AVAILABLE" = true ]; then
    echo "✅ OpenCV: AVAILABLE (Enhanced image processing)"
else
    echo "⚠️  OpenCV: NOT AVAILABLE (Will use PIL fallback)"
fi

echo ""
if [ "$PROFESSIONAL_MODE" = "full" ]; then
    echo "🎉 FULL PROFESSIONAL MODE ENABLED!"
    echo "   💰 Maximum income generation potential"
    echo "   🎯 Real-SR AI upscaling (best quality)"
    echo "   📈 Market-leading image quality"
    echo "   ⚠️  Uses ~2-3GB of dependencies"
elif [ "$PROFESSIONAL_MODE" = "lightweight" ]; then
    echo "🚀 LIGHTWEIGHT PROFESSIONAL MODE ENABLED!"
    echo "   💰 Professional income generation ready"
    echo "   🎯 API-based AI upscaling (excellent quality)"
    echo "   📱 Perfect for Android/mobile devices"
    echo "   ✅ Only ~50MB of dependencies"
else
    echo "⚠️  BASIC MODE ONLY"
    echo "   📝 Traditional upscaling available"
    echo "   💡 Run script again to enable professional features"
fi

echo ""
echo "🚀 Ready to start your AI Image Studio!"
echo "   Run: python3 app.py"
echo "   Then open: http://localhost:5000"
echo ""
echo "💡 For troubleshooting, see: REALSR_PROFESSIONAL_SETUP.md"