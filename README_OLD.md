# 📱 Flask Web App for Termux (Android Tablet)

A beautiful, responsive Flask web application designed specifically for Android tablets running Termux. Features a modern UI with notes, tasks, and system information.

## 🚀 Features

- **📝 Notes Management**: Add, view, and delete notes with timestamps
- **✅ Task Management**: Create tasks, mark as complete, and delete
- **📱 Mobile-Optimized**: Responsive design perfect for tablets
- **⚙️ System Info**: View detailed system and environment information
- **🎨 Modern UI**: Beautiful gradient design with smooth animations
- **🔄 Real-time Updates**: AJAX-powered interface for smooth interactions

## 📋 Prerequisites

1. **Android Tablet** (tested on Galaxy Tab A9+)
2. **Termux App** installed from F-Droid or Google Play Store
3. **Internet connection** for initial setup

## 🛠️ Installation Guide

### Step 1: Install Termux
Download Termux from:
- **F-Droid** (recommended): https://f-droid.org/packages/com.termux/
- **Google Play Store**: https://play.google.com/store/apps/details?id=com.termux

### Step 2: Update Termux Packages
Open Termux and run:
```bash
pkg update && pkg upgrade -y
```

### Step 3: Install Python and Git
```bash
pkg install python git -y
```

### Step 4: Install pip (if not already installed)
```bash
pkg install python-pip -y
```

### Step 5: Clone or Download This Project
Option A - Using Git:
```bash
git clone <your-repo-url>
cd flask-termux-app
```

Option B - Manual Setup:
```bash
mkdir flask-app
cd flask-app
# Copy all files from this project to the flask-app directory
```

### Step 6: Install Python Dependencies
```bash
pip install -r requirements.txt
```

### Step 7: Run the Flask App
```bash
python app.py
```

## 🌐 Accessing Your App

### On Your Tablet
Once the app is running, you'll see output like:
```
Starting Flask app on port 5000
Access your app at:
  Local: http://localhost:5000
  Network: http://0.0.0.0:5000
```

Open your tablet's web browser and go to: `http://localhost:5000`

### From Other Devices (Optional)
1. Find your tablet's IP address:
   ```bash
   ifconfig wlan0
   ```
2. On other devices, access: `http://YOUR_TABLET_IP:5000`

## 📱 Usage Instructions

### Notes Tab
- **Add Notes**: Type in the text area and click "Save Note"
- **View Notes**: All notes display with timestamps
- **Delete Notes**: Click the red "Delete" button on any note

### Tasks Tab
- **Add Tasks**: Enter task description and click "Add Task"
- **Complete Tasks**: Check the checkbox to mark as complete
- **Delete Tasks**: Click the red "Delete" button on any task

### System Tab
- **Quick Info**: View basic system statistics
- **Detailed Info**: Click "Detailed System Info" for comprehensive details
- **Termux Tips**: Helpful tips for using the app on Termux

## ⚙️ Configuration

### Custom Port
To run on a different port:
```bash
PORT=8080 python app.py
```

### Environment Variables
- `PORT`: Set custom port (default: 5000)
- `SECRET_KEY`: Set in app.py for production use

## 🔧 Troubleshooting

### Common Issues

**1. "Permission denied" errors**
```bash
termux-setup-storage
```

**2. "Module not found" errors**
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**3. "Address already in use"**
```bash
# Kill existing processes
pkill -f python
# Or use a different port
PORT=8080 python app.py
```

**4. Can't access from browser**
- Make sure you're using `http://localhost:5000` (not https)
- Try `http://127.0.0.1:5000` instead
- Check if Termux has network permissions

### Performance Tips

1. **Keep tablet plugged in** during development
2. **Close other apps** to free up memory
3. **Use WiFi** for better stability
4. **Enable "Stay awake"** in Developer Options

## 📁 Project Structure

```
flask-termux-app/
├── app.py                 # Main Flask application
├── requirements.txt       # Python dependencies
├── README.md             # This file
└── templates/
    ├── base.html         # Base template with styling
    ├── index.html        # Main dashboard
    └── system_info.html  # System information page
```

## 🎨 Customization

### Changing Colors
Edit the CSS in `templates/base.html`:
- Main gradient: `background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);`
- Button colors: `.btn-primary` and `.btn-danger` classes

### Adding Features
The app is designed to be easily extensible:
1. Add new routes in `app.py`
2. Create new templates in `templates/`
3. Add new tabs in `index.html`

## 🔒 Security Notes

- **Development Mode**: The app runs in debug mode by default
- **Production Use**: Change the `SECRET_KEY` in `app.py`
- **Network Access**: Only enable network access if needed
- **Data Storage**: Notes and tasks are stored in memory (lost on restart)

## 📊 System Requirements

- **Android**: 7.0+ (API level 24+)
- **RAM**: 2GB+ recommended
- **Storage**: 100MB+ free space
- **Python**: 3.8+ (installed via Termux)

## 🆘 Support

If you encounter issues:

1. **Check Termux logs** for error messages
2. **Restart Termux** and try again
3. **Update packages**: `pkg update && pkg upgrade`
4. **Reinstall dependencies**: `pip install -r requirements.txt --force-reinstall`

## 📝 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- Built with Flask web framework
- Designed for Termux Android terminal emulator
- Optimized for tablet usage and touch interfaces

---

**Happy coding on your Android tablet! 🚀📱**