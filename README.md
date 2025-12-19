# AI-Powered Flask Web App for Termux (Android Tablets)

A comprehensive mobile-optimized Flask web application with AI image generation and editing capabilities, designed to run on Android tablets using Termux. Features a responsive interface perfect for touch interaction and powerful AI-driven image processing.

## 🌟 Features

### Core Features
- 📝 **Notes Management**: Add, view, and delete notes with timestamps
- ✅ **Task Tracking**: Create and manage tasks with completion status
- 📱 **Mobile-Optimized**: Responsive design perfect for tablets
- 🎨 **Beautiful UI**: Gradient backgrounds and smooth animations
- ⚙️ **System Info**: View detailed system information

### AI Image Features
- 🤖 **AI Image Generation**: Create stunning images using Flux and Stable Diffusion models
- 🎯 **Smart Analysis**: Automatic image rating and categorization using vision AI
- 📁 **Auto-Sorting**: Images automatically sorted into Top, Mids, and Meh folders
- ✏️ **Advanced Image Editing**: Powerful inpainting, transformation, and background removal
- 🖌️ **Canvas Drawing**: Touch-friendly mask drawing for precise inpainting
- 🔍 **Image Upscaling**: Enhance image quality with AI upscaling
- 🖼️ **Gallery Management**: Browse, organize, and manage your AI-generated images

### Image Editing Capabilities
- **Inpainting**: Draw on images to select areas for AI-powered replacement
- **Image-to-Image**: Transform images while maintaining composition
- **Background Removal**: Automatically remove backgrounds using AI
- **Batch Processing**: Generate and process multiple images at once
- **Real-time Progress**: Live progress tracking for long operations

## 🚀 Installation on Termux

### Step 1: Install Termux
Download Termux from [F-Droid](https://f-droid.org/packages/com.termux/) (recommended) or Google Play Store.

### Step 2: Update Termux packages
```bash
pkg update && pkg upgrade
```

### Step 3: Install Python and Git
```bash
pkg install python git
```

### Step 4: Clone or create the project
```bash
# Create project directory
mkdir flask-ai-app
cd flask-ai-app

# Copy the app files (or clone from repository)
```

### Step 5: Install Python dependencies
```bash
pip install -r requirements.txt
```

### Step 6: Get OpenRouter API Key
1. Visit [OpenRouter.ai](https://openrouter.ai)
2. Sign up for an account
3. Generate an API key
4. Keep it secure for configuration

### Step 7: Run the application
```bash
python app.py
```

The app will start on port 5000. Access it at:
- Local: `http://localhost:5000`
- Network: `http://YOUR_TABLET_IP:5000`

## 🎯 Usage Guide

### Initial Setup
1. **Configure AI**: Go to "AI Images" tab → "Configure"
2. **Enter API Key**: Add your OpenRouter API key
3. **Set Preferences**: Choose models, batch size, and prompts
4. **Save Configuration**: Click "Save Configuration"

### AI Image Generation
1. **Navigate**: Go to "AI Images" tab
2. **Generate**: Click "🚀 Generate & Analyze Images"
3. **Monitor**: Watch real-time progress updates
4. **View Results**: Check the gallery when complete

### Image Editing
1. **Access Editor**: Click "✏️ Image Editor" (after configuration)
2. **Select Image**: Choose from generated images or upload new ones
3. **Choose Tool**:
   - **Inpainting**: Draw on areas to replace with AI
   - **Transform**: Apply style changes to entire image
   - **Remove BG**: Automatically remove backgrounds
4. **Process**: Follow prompts and wait for AI processing

### Canvas Drawing (Inpainting)
- **Brush Tool**: Draw areas you want to replace
- **Erase Tool**: Remove drawn areas
- **Brush Size**: Adjust brush size with slider
- **Touch Support**: Optimized for finger/stylus drawing
- **Clear All**: Reset the mask completely

### Gallery Management
- **Browse Categories**: Top Notch, Mids, Meh, Upscaled
- **View Images**: Click to open in full-screen modal
- **Upscale**: Enhance image quality with 2x upscaling
- **Delete**: Remove unwanted images
- **Download**: Save edited results

## ⚙️ Configuration Options

### Generation Models
- **Flux Schnell**: Fast, good quality (recommended for testing)
- **Flux Pro**: Highest quality (premium)
- **Stable Diffusion 3**: Alternative high-quality option

### Vision Models
- **Grok Vision Beta**: Excellent analysis (recommended)
- **GPT-4 Vision**: Premium analysis option
- **Claude 3.5 Sonnet**: Alternative vision model

### Batch Settings
- **Batch Size**: 1-5 images per generation
- **Base Prompt**: Template for all generated images
- **Custom Prompts**: Variations added automatically

## 🛠️ Customization

### Changing the Port
```bash
PORT=8080 python app.py
```

### Custom Prompts
Edit prompts in the configuration page or modify defaults in `app.py`

### Styling Changes
- Edit `templates/base.html` for CSS modifications
- Customize colors, fonts, and layouts
- Mobile-first responsive design

### Adding Features
- Extend `ImageWorkflowAutomator` class
- Add new routes in `app.py`
- Create additional templates

## 📁 File Structure

```
flask-ai-app/
├── app.py                      # Main Flask application with AI features
├── requirements.txt            # Python dependencies
├── templates/
│   ├── base.html              # Base template with styling
│   ├── index.html             # Main dashboard with AI features
│   ├── workflow_config.html   # AI configuration page
│   ├── workflow_gallery.html  # Image gallery
│   ├── image_editor.html      # Advanced image editor
│   └── system_info.html       # System information
├── ImageWorkflow/             # AI-generated images (created automatically)
│   ├── Generated/             # Raw generated images
│   ├── Top_Notch/            # High-rated images (8-10)
│   ├── Mids/                 # Mid-rated images (6-7)
│   ├── Meh/                  # Lower-rated images (1-5)
│   ├── Upscaled/             # Enhanced images
│   └── Edited/               # Edited/inpainted images
└── README.md                  # This file
```

## 🔧 Troubleshooting

### API Issues
- **Invalid API Key**: Check your OpenRouter API key
- **Rate Limits**: Monitor your API usage on OpenRouter dashboard
- **Model Errors**: Try different models if one fails

### Performance Issues
- **Memory**: Close other apps for better performance
- **Battery**: Keep tablet plugged in during AI processing
- **Network**: Ensure stable internet for API calls

### Common Errors
- **Port in Use**: Use `PORT=8080 python app.py`
- **Dependencies**: Run `pip install -r requirements.txt` again
- **Permissions**: Run `termux-setup-storage` for file access

### Canvas Drawing Issues
- **Touch Not Working**: Try refreshing the page
- **Brush Size**: Adjust with the slider for better control
- **Mask Not Saving**: Ensure you draw before submitting

## 💡 Tips for Best Results

### Image Generation
- **Specific Prompts**: Be detailed and descriptive
- **Style Keywords**: Include art styles (e.g., "watercolor", "digital art")
- **Quality Terms**: Add "high quality", "detailed", "masterpiece"

### Inpainting
- **Precise Masks**: Draw carefully around areas to edit
- **Clear Prompts**: Describe exactly what should replace the area
- **Multiple Attempts**: Try different prompts for better results

### Performance
- **Batch Size**: Start with 3 images, increase if performance allows
- **Model Selection**: Use Flux Schnell for speed, Pro for quality
- **Network**: Use WiFi for faster API calls

## 📊 Cost Management

### OpenRouter Pricing
- **Flux Schnell**: ~$0.003 per image
- **Flux Pro**: ~$0.05 per image
- **Vision Analysis**: ~$0.01 per analysis

### Cost Control
- **Monitor Usage**: Check OpenRouter dashboard regularly
- **Batch Wisely**: Generate fewer images initially
- **Model Selection**: Balance cost vs quality needs

## 🔒 Security Notes

- **API Key**: Never share your OpenRouter API key
- **Local Storage**: Images stored locally on your device
- **Network**: App runs locally, only API calls go external
- **Privacy**: No data sent except to OpenRouter for processing

## 🤝 Contributing

Ideas for extensions:
- **More AI Models**: Add support for additional image models
- **Video Processing**: Extend to video generation/editing
- **Batch Operations**: More advanced batch processing
- **Cloud Storage**: Integration with cloud storage services
- **User Accounts**: Multi-user support with authentication

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- **OpenRouter**: For providing access to cutting-edge AI models
- **Flask**: For the excellent web framework
- **Termux**: For enabling Python on Android
- **AI Model Providers**: Stability AI, Black Forest Labs, xAI, and others

---

**Ready to create amazing AI-powered images on your Android tablet? Get started now!** 🚀