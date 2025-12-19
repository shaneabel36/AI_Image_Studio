#!/usr/bin/env python3
"""
OpenCV Test Script for AI Image Studio
Run this to check if OpenCV works on your system
"""

print("🔧 Testing OpenCV installation for AI Image Studio...")
print("=" * 50)

# Test 1: OpenCV Import
print("\n📦 Test 1: OpenCV Import")
try:
    import cv2
    print(f"✅ OpenCV imported successfully!")
    print(f"📦 Version: {cv2.__version__}")
    opencv_available = True
except ImportError as e:
    print(f"❌ OpenCV import failed: {e}")
    print("💡 This is OK - AI Image Studio will use PIL instead")
    opencv_available = False

# Test 2: Basic OpenCV Operations
if opencv_available:
    print("\n🔧 Test 2: Basic OpenCV Operations")
    try:
        import numpy as np
        
        # Create test image
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        img[:, :] = [64, 128, 255]  # Blue color
        
        # Test resize
        resized = cv2.resize(img, (200, 200), interpolation=cv2.INTER_CUBIC)
        print(f"✅ Image resize works!")
        print(f"📏 Original: {img.shape}, Resized: {resized.shape}")
        
        # Test file operations
        cv2.imwrite('/tmp/test_opencv.png', img)
        loaded = cv2.imread('/tmp/test_opencv.png')
        if loaded is not None:
            print(f"✅ File I/O works!")
        else:
            print(f"❌ File I/O failed")
            
    except Exception as e:
        print(f"❌ OpenCV operations failed: {e}")
        print("💡 Will use PIL fallback mode")
        opencv_available = False

# Test 3: PIL Fallback
print("\n🎨 Test 3: PIL Fallback (Always Available)")
try:
    from PIL import Image, ImageDraw
    import numpy as np
    
    # Create test image with PIL
    img_array = np.zeros((100, 100, 3), dtype=np.uint8)
    img_array[:, :] = [255, 128, 64]  # Orange color
    img = Image.fromarray(img_array)
    
    # Test resize with PIL
    resized = img.resize((200, 200), Image.Resampling.LANCZOS)
    print(f"✅ PIL resize works!")
    print(f"📏 Original: {img.size}, Resized: {resized.size}")
    
    # Test file operations
    img.save('/tmp/test_pil.png', 'PNG')
    loaded = Image.open('/tmp/test_pil.png')
    print(f"✅ PIL file I/O works!")
    
    pil_available = True
    
except Exception as e:
    print(f"❌ PIL failed: {e}")
    print("🚨 This is a problem - PIL is required!")
    pil_available = False

# Test 4: AI Image Studio Compatibility
print("\n🚀 Test 4: AI Image Studio Compatibility")
print("=" * 50)

if opencv_available:
    print("✅ FULL MODE: OpenCV + PIL available")
    print("   🎨 All image processing features enabled")
    print("   🔧 Advanced upscaling with OpenCV")
    print("   📊 Optimal performance")
elif pil_available:
    print("✅ COMPATIBLE MODE: PIL only")
    print("   🎨 All core features available")
    print("   🔧 PIL-based upscaling")
    print("   📊 Good performance")
else:
    print("❌ INCOMPATIBLE: Neither OpenCV nor PIL working")
    print("   🚨 AI Image Studio will not work properly")

# Test 5: Recommendations
print("\n💡 Recommendations:")
print("=" * 50)

if not opencv_available:
    print("📋 To install OpenCV on Termux:")
    print("   1. pkg install clang cmake ninja libjpeg-turbo")
    print("   2. pip install opencv-python-headless==4.8.1.78")
    print("   3. If that fails: pip install opencv-python-headless==4.5.5.64")
    print("   4. Or use: ./install_opencv_termux.sh")
    print()
    print("🎯 Alternative: Use requirements_no_opencv.txt")
    print("   pip install -r requirements_no_opencv.txt")

if pil_available:
    print("\n🎉 Your AI Image Studio is ready to run!")
    print("   python app.py")
    print("   Then open: http://localhost:5000")

print("\n" + "=" * 50)
print("🎨 AI Image Studio - Professional Creative Workspace")
print("=" * 50)