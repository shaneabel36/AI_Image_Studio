# 📱 AI Image Studio - Complete Installation Guide
## Professional Setup for Income Generation on Android/Termux

This guide provides multiple installation methods to get your AI Image Studio running perfectly on your Android tablet.

## 🎯 **Choose Your Installation Method**

### **Method 1: Professional Auto-Install (Recommended)** 🚀
**Best for: Income generation, automatic dependency management**

```bash
# Clone your AI Image Studio
git clone https://github.com/shaneabel36/AI_Image_Studio.git
cd AI_Image_Studio

# Run the smart installation script
./install_professional.sh

# Start your professional studio
python app.py
```

**What this does:**
- ✅ Installs core dependencies without OpenCV conflicts
- ✅ Attempts Real-SR installation for professional upscaling
- ✅ Tests compatibility and provides fallbacks
- ✅ Gives you a detailed status report

---

### **Method 2: Manual Professional Setup** 💰
**Best for: Maximum control, professional income generation**

```bash
# Step 1: Core dependencies (OpenCV-free)
pip install -r requirements.txt

# Step 2: Professional Real-SR for income generation
pip install -r requirements_professional.txt

# Step 3: Test your setup
python test_opencv.py

# Step 4: Start earning!
python app.py
```

---

### **Method 3: Full Features (If OpenCV Works)** 🔧
**Best for: Maximum features, if you can get OpenCV working**

```bash
# Try full installation
pip install -r requirements_opencv.txt

# If that fails, try individual OpenCV versions:
pip install opencv-python-headless==4.8.1.78
# OR
pip install opencv-python-headless==4.5.5.64
# OR
pip install opencv-python-headless==4.7.1.72
```

---

### **Method 4: Minimal Setup (Fallback)** 🛡️
**Best for: When everything else fails, basic functionality**

```bash
# Minimal dependencies only
pip install -r requirements_minimal.txt

# This gives you:
# - Basic Flask app
# - PIL-based image processing
# - No Real-SR (traditional upscaling only)
```

---

### **Method 5: Android Optimized** 📱
**Best for: Termux-specific issues, mobile optimization**

```bash
# Use the no-opencv version
pip install -r requirements_no_opencv.txt

# Then try to add Real-SR separately
pip install realesrgan basicsr --no-deps
```

## 🔍 **Troubleshooting Common Issues**

### **Issue 1: OpenCV Installation Fails**
```bash
# Solution: Use OpenCV-free mode
pip install -r requirements.txt  # This now excludes OpenCV
python test_opencv.py  # Will show PIL fallback mode
```

### **Issue 2: Real-SR Installation Fails**
```bash
# Try older versions
pip install realesrgan==0.2.5
pip install basicsr==1.3.5

# Or skip Real-SR for now
pip install -r requirements_minimal.txt
```

### **Issue 3: Memory Issues on Android**
```bash
# Reduce memory usage in app.py
# Edit line ~95: change tile=512 to tile=256
# This reduces memory usage for Real-SR
```

### **Issue 4: Permission Errors**
```bash
# Fix Termux permissions
termux-setup-storage
pkg install clang cmake ninja
```

### **Issue 5: Network/Download Issues**
```bash
# Use offline installation
pip install --no-deps realesrgan
pip install --no-deps basicsr
```

## 📊 **Installation Status Check**

After installation, run this to check your setup:

```bash
python test_opencv.py
```

**Expected Output for Professional Mode:**
```
✅ FULL MODE: OpenCV + PIL available
✅ Real-SR: AVAILABLE (Professional income generation ready!)
🎉 Your AI Image Studio is ready to run!
```

**Expected Output for Compatible Mode:**
```
✅ COMPATIBLE MODE: PIL only
✅ Real-SR: AVAILABLE (Professional income generation ready!)
🎉 Your AI Image Studio is ready to run!
```

## 🎯 **Recommended Setup by Use Case**

### **For Income Generation (Primary Goal)** 💰
```bash
./install_professional.sh
```
- Prioritizes Real-SR for professional quality
- Handles OpenCV gracefully
- Optimized for reliability

### **For Learning/Experimentation** 🧪
```bash
pip install -r requirements_opencv.txt
```
- Maximum features
- All capabilities enabled
- May require troubleshooting

### **For Reliable Basic Use** 🛡️
```bash
pip install -r requirements_minimal.txt
```
- Always works
- Basic functionality
- No advanced AI features

### **For Android/Termux Specifically** 📱
```bash
pip install -r requirements_no_opencv.txt
```
- Mobile-optimized
- Avoids common Termux issues
- Good balance of features

## 🚀 **Quick Start Commands**

**One-liner for professional setup:**
```bash
git clone https://github.com/shaneabel36/AI_Image_Studio.git && cd AI_Image_Studio && ./install_professional.sh && python app.py
```

**One-liner for minimal setup:**
```bash
git clone https://github.com/shaneabel36/AI_Image_Studio.git && cd AI_Image_Studio && pip install -r requirements_minimal.txt && python app.py
```

## 💡 **Pro Tips**

1. **Start with Method 1** - The auto-installer handles most issues
2. **Check test_opencv.py** - Always run this after installation
3. **Use HTTPS cloning** - More reliable than SSH on mobile
4. **Keep it simple** - For income generation, reliability > features
5. **Monitor logs** - Check app.log for Real-SR status

## 🎉 **Success Indicators**

Your installation is successful when:
- ✅ `python app.py` starts without errors
- ✅ `http://localhost:5000` loads the interface
- ✅ Real-SR status shows "Professional mode enabled"
- ✅ Image upscaling produces high-quality results

---

**🎯 Your AI Image Studio is now ready for professional income generation!**

*For specific Real-SR setup and optimization, see: `REALSR_PROFESSIONAL_SETUP.md`*