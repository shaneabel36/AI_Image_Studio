# 🔧 OpenCV-Python Debugging Guide for Termux

## 🚨 Common Issues on Android/Termux

### Issue 1: Installation Fails
```bash
ERROR: Failed building wheel for opencv-python
```

**Solutions:**
1. **Use the installation script:**
   ```bash
   chmod +x install_opencv_termux.sh
   ./install_opencv_termux.sh
   ```

2. **Manual installation:**
   ```bash
   pkg install clang cmake ninja libjpeg-turbo libpng
   pip install opencv-python-headless==4.8.1.78
   ```

### Issue 2: Import Error
```python
ImportError: No module named 'cv2'
```

**Solutions:**
1. **Try headless version:**
   ```bash
   pip uninstall opencv-python
   pip install opencv-python-headless
   ```

2. **Use older version:**
   ```bash
   pip install opencv-python-headless==4.5.5.64
   ```

### Issue 3: Runtime Crashes
```
Segmentation fault (core dumped)
```

**Solutions:**
1. **Use the OpenCV-free version:**
   ```bash
   pip install -r requirements_no_opencv.txt
   ```

2. **The app will automatically fallback to PIL for image processing**

## 🎯 Testing OpenCV Installation

Run this test script:

```python
#!/usr/bin/env python3
# test_opencv.py

print("🔧 Testing OpenCV installation...")

try:
    import cv2
    print(f"✅ OpenCV imported successfully!")
    print(f"📦 Version: {cv2.__version__}")
    
    # Test basic functionality
    import numpy as np
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    resized = cv2.resize(img, (200, 200))
    print(f"✅ Basic operations work!")
    print(f"📏 Resized image shape: {resized.shape}")
    
except ImportError as e:
    print(f"❌ OpenCV import failed: {e}")
    print("💡 Try: pip install opencv-python-headless")
    
except Exception as e:
    print(f"❌ OpenCV runtime error: {e}")
    print("💡 Try using the OpenCV-free version")

print("\n🎨 Testing PIL fallback...")
try:
    from PIL import Image
    import numpy as np
    
    # Create test image
    img_array = np.zeros((100, 100, 3), dtype=np.uint8)
    img = Image.fromarray(img_array)
    resized = img.resize((200, 200), Image.Resampling.LANCZOS)
    print(f"✅ PIL works as fallback!")
    print(f"📏 Resized image size: {resized.size}")
    
except Exception as e:
    print(f"❌ PIL error: {e}")

print("\n🚀 Your AI Image Studio will work with or without OpenCV!")
```

## 🛠️ Troubleshooting Steps

### Step 1: Check System Architecture
```bash
uname -m
# Should show: aarch64 (64-bit ARM) or armv7l (32-bit ARM)
```

### Step 2: Update Termux
```bash
pkg update && pkg upgrade
pkg install python python-pip
```

### Step 3: Install System Dependencies
```bash
pkg install clang cmake ninja
pkg install libjpeg-turbo libpng libtiff libwebp
```

### Step 4: Try Different OpenCV Versions
```bash
# Method 1: Latest headless
pip install opencv-python-headless

# Method 2: Specific version
pip install opencv-python-headless==4.8.1.78

# Method 3: Older stable version
pip install opencv-python-headless==4.5.5.64

# Method 4: Contrib version
pip install opencv-contrib-python-headless
```

### Step 5: Use OpenCV-Free Mode
If all else fails, your AI Image Studio will automatically work without OpenCV:

```bash
# Install without OpenCV
pip install -r requirements_no_opencv.txt

# Run the app - it will use PIL instead
python app.py
```

## 🎨 Features Available Without OpenCV

Your AI Image Studio works great even without OpenCV:

✅ **Available:**
- AI image generation
- Image editing and inpainting  
- Gallery management
- Batch processing
- PIL-based upscaling
- All web interface features

❌ **Limited:**
- Advanced computer vision features
- Some image processing optimizations

## 🚀 Quick Fix Commands

```bash
# Quick install attempt
pip install opencv-python-headless==4.8.1.78

# Test if it works
python -c "import cv2; print('OpenCV version:', cv2.__version__)"

# If it fails, use OpenCV-free mode
pip install -r requirements_no_opencv.txt
python app.py
```

## 📱 Termux-Specific Tips

1. **Use headless versions** - GUI versions don't work in Termux
2. **Install system deps first** - OpenCV needs compiled libraries
3. **Try older versions** - Newer versions may not support your ARM architecture
4. **Use the fallback** - PIL works great for most image operations

## 🎯 Success Indicators

✅ **OpenCV Working:**
```
OpenCV imported successfully
OpenCV version: 4.x.x
Basic operations work!
```

✅ **PIL Fallback Working:**
```
OpenCV not available - some features will be disabled
PIL works as fallback!
Your AI Image Studio will work with or without OpenCV!
```

Both scenarios mean your AI Image Studio will work perfectly! 🎉