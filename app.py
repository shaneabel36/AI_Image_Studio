from flask import Flask, render_template, request, jsonify, flash, redirect, url_for, send_from_directory
import os
import json
import logging
import traceback
import threading
import time
import base64
import requests
from datetime import datetime
from typing import List, Dict, Optional

# Set up logging first
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
    logger.info("Environment variables loaded from .env file")
except ImportError:
    logger.warning("python-dotenv not installed - .env file won't be loaded automatically")
except Exception as e:
    logger.warning(f"Could not load .env file: {e}")

try:
    import cv2
    OPENCV_AVAILABLE = True
    logger.info("OpenCV imported successfully")
except ImportError:
    OPENCV_AVAILABLE = False
    logger.warning("OpenCV not available - some features will be disabled")
    cv2 = None

# Professional AI upscaling imports (Real-SR)
try:
    from realesrgan import RealESRGANer
    from basicsr.archs.rrdbnet_arch import RRDBNet
    REALSR_AVAILABLE = True
    logger.info("Real-SR imported successfully - Professional upscaling enabled")
except ImportError:
    REALSR_AVAILABLE = False
    logger.warning("Real-SR not available - Using fallback upscaling")
    RealESRGANer = None
    RRDBNet = None

# Aliases for compatibility
ESRGAN_AVAILABLE = REALSR_AVAILABLE  # Alias for backward compatibility

# Lightweight API upscaler (no heavy dependencies)
try:
    from api_upscaler import api_upscaler
    API_UPSCALER_AVAILABLE = True
    logger.info("API upscaler loaded - Professional upscaling available without heavy dependencies")
except ImportError:
    API_UPSCALER_AVAILABLE = False
    api_upscaler = None
    logger.warning("API upscaler not available")

import numpy as np
from werkzeug.utils import secure_filename
from PIL import Image, ImageDraw
import io

# Logger already configured above

app = Flask(__name__)

# Configure for mobile/tablet use
app.config['SECRET_KEY'] = 'your-secret-key-change-this-in-production'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = os.path.join(os.getcwd(), 'uploads')

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Sample data for demonstration
notes = []
tasks = []

# Global variables for workflow status
workflow_status = {
    'running': False,
    'progress': 0,
    'message': '',
    'error': None,
    'results': []
}

class ProfessionalUpscaler:
    """Professional AI upscaling using Real-SR for income generation"""
    
    def __init__(self):
        self.realsr_models = {}
        self.model_urls = {
            'RealESRGAN_x4plus': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth',
            'RealESRGAN_x2plus': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth',
            'RealESRGAN_x4plus_anime_6B': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth'
        }
        self.models_dir = os.path.join(os.getcwd(), 'realsr_models')
        os.makedirs(self.models_dir, exist_ok=True)
        
        if REALSR_AVAILABLE:
            self._initialize_models()
    
    def _initialize_models(self):
        """Initialize Real-SR models for professional upscaling"""
        try:
            # Real-SR x4 - Best for general images (income generation quality)
            model_path = self._ensure_model_downloaded('RealESRGAN_x4plus')
            if model_path:
                model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
                self.realsr_models['x4_professional'] = RealESRGANer(
                    scale=4,
                    model_path=model_path,
                    model=model,
                    tile=512,  # Optimize for mobile memory
                    tile_pad=10,
                    pre_pad=0,
                    half=False  # Set to True if you have GPU
                )
                logger.info("Real-SR x4 model loaded - Professional quality enabled for income generation")
            
            # Real-SR x2 - Faster processing for quick turnaround
            model_path = self._ensure_model_downloaded('RealESRGAN_x2plus')
            if model_path:
                model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
                self.realsr_models['x2_fast'] = RealESRGANer(
                    scale=2,
                    model_path=model_path,
                    model=model,
                    tile=512,
                    tile_pad=10,
                    pre_pad=0,
                    half=False
                )
                logger.info("Real-SR x2 model loaded - Fast processing enabled")
                
        except Exception as e:
            logger.error(f"Error initializing Real-SR models: {e}")
    
    def _ensure_model_downloaded(self, model_name: str) -> str:
        """Download Real-SR model if not present"""
        model_path = os.path.join(self.models_dir, f"{model_name}.pth")
        
        if os.path.exists(model_path):
            return model_path
        
        try:
            logger.info(f"Downloading {model_name} model for professional upscaling...")
            url = self.model_urls[model_name]
            
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            
            with open(model_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            progress = (downloaded / total_size) * 100
                            if downloaded % (1024 * 1024) == 0:  # Log every MB
                                logger.info(f"Downloading {model_name}: {progress:.1f}%")
            
            logger.info(f"Successfully downloaded {model_name} model")
            return model_path
            
        except Exception as e:
            logger.error(f"Failed to download {model_name}: {e}")
            return None
    
    def professional_upscale(self, image_path: str, scale_factor: int = 4, quality: str = 'best') -> str:
        """Professional AI upscaling using Real-SR for income generation"""
        try:
            if not REALSR_AVAILABLE:
                raise ValueError("Real-SR not available - install with: pip install realesrgan basicsr")
            
            # Choose model based on quality preference for income generation
            if quality == 'best' and scale_factor == 4 and 'x4_professional' in self.realsr_models:
                upscaler = self.realsr_models['x4_professional']
                actual_scale = 4
                method = "Real-SR Professional"
            elif quality == 'fast' and scale_factor <= 2 and 'x2_fast' in self.realsr_models:
                upscaler = self.realsr_models['x2_fast']
                actual_scale = 2
                method = "Real-SR Fast"
            elif 'x4_professional' in self.realsr_models:
                upscaler = self.realsr_models['x4_professional']
                actual_scale = 4
                method = "Real-SR Professional"
            else:
                raise ValueError("No suitable Real-SR model available")
            
            # Load and process image
            logger.info(f"Starting {method} upscaling: {quality} quality, {actual_scale}x scale for income generation")
            
            # Read image
            img = cv2.imread(image_path, cv2.IMREAD_COLOR) if OPENCV_AVAILABLE else None
            if img is None:
                # Fallback to PIL
                pil_img = Image.open(image_path)
                img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            
            # Professional AI upscaling
            output, _ = upscaler.enhance(img, outscale=actual_scale)
            
            # Save result with Real-SR branding
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            output_path = os.path.join(
                os.path.dirname(image_path).replace('Generated', 'Upscaled'),
                f"{base_name}_RealSR_{actual_scale}x_{quality}.png"
            )
            
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            cv2.imwrite(output_path, output)
            
            logger.info(f"Professional Real-SR upscaling complete: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Professional Real-SR upscaling failed: {e}")
            raise

# Initialize professional upscaler
professional_upscaler = ProfessionalUpscaler() if REALSR_AVAILABLE else None

class ImageWorkflowAutomator:
    """Enhanced Image Workflow Automator with improved error handling"""
    
    def __init__(self, api_key: str = None, batch_size: int = 3, 
                 base_prompt: str = "A beautiful landscape in artistic style",
                 gen_model: str = 'black-forest-labs/flux-schnell', 
                 vision_model: str = 'x-ai/grok-vision-beta',
                 safety_level: str = "moderate", content_filter: bool = True,
                 custom_safety_params: dict = None):
        
        self.api_key = api_key
        self.gen_model = gen_model
        self.vision_model = vision_model
        self.batch_size = min(batch_size, 5)  # Limit batch size for mobile
        self.base_prompt = base_prompt
        self.safety_level = safety_level  # "strict", "moderate", "permissive", "off"
        self.content_filter = content_filter
        self.custom_safety_params = custom_safety_params or {}
        
        # Setup directories
        self.base_dir = os.path.join(os.getcwd(), 'ImageWorkflow')
        self.dirs = {
            'generated': os.path.join(self.base_dir, 'Generated'),
            'top': os.path.join(self.base_dir, 'Top_Notch'),
            'mids': os.path.join(self.base_dir, 'Mids'),
            'meh': os.path.join(self.base_dir, 'Meh'),
            'upscaled': os.path.join(self.base_dir, 'Upscaled')
        }
        
        # Create directories
        for d in self.dirs.values():
            try:
                os.makedirs(d, exist_ok=True)
                logger.info(f"Created directory: {d}")
            except Exception as e:
                logger.error(f"Failed to create directory {d}: {e}")
        
        # Initialize OpenRouter client if API key provided
        self.client = None
        if api_key and api_key.strip() and api_key != "your_openrouter_key_here":
            try:
                from openai import OpenAI
                self.client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")
                logger.info("OpenRouter client initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize OpenRouter client: {e}")
                self.client = None
    
    def update_status(self, progress: int, message: str, error: str = None):
        """Update global workflow status"""
        global workflow_status
        workflow_status.update({
            'progress': progress,
            'message': message,
            'error': error
        })
        logger.info(f"Status update: {progress}% - {message}")
    
    def get_safety_parameters(self):
        """Get safety parameters based on current safety level"""
        base_params = {}
        
        # OpenAI/OpenRouter doesn't use these safety parameters in the API
        # Safety is handled by the models themselves
        if self.safety_level == "strict":
            base_params.update({
                "content_filter": True,
                "safe_mode": True
            })
        elif self.safety_level == "moderate":
            base_params.update({
                "content_filter": self.content_filter
            })
        elif self.safety_level == "permissive":
            base_params.update({
                "content_filter": False
            })
        elif self.safety_level == "off":
            base_params.update({
                "content_filter": False,
                "safe_mode": False
            })
        
        # Apply custom overrides
        base_params.update(self.custom_safety_params)
        
        # Filter out None values and parameters not supported by the OpenAI API
        # OpenAI Images API only supports: model, prompt, n, size, response_format, user
        # Safety is handled by the models themselves, not via API parameters
        supported_params = ['model', 'prompt', 'n', 'size', 'response_format', 'user']
        return {}
    
    def generate_batch(self) -> List[str]:
        """Generate batch of images with error handling"""
        if not self.client:
            raise ValueError("OpenRouter API client not initialized. Please check your API key.")
        
        paths = []
        self.update_status(10, "Starting image generation...")
        
        for i in range(self.batch_size):
            try:
                self.update_status(10 + (i * 30 // self.batch_size), 
                                 f"Generating image {i+1}/{self.batch_size}...")
                
                # Generate image with safety parameters
                safety_params = self.get_safety_parameters()
                generation_params = {
                    "model": self.gen_model,
                    "prompt": f"{self.base_prompt} - variation {i+1}",
                    "n": 1,
                    "size": "1024x1024"
                }
                
                # Add safety parameters if they exist
                generation_params.update(safety_params)
                
                response = self.client.images.generate(**generation_params)
                
                if not response.data:
                    logger.warning(f"No image data received for generation {i+1}")
                    continue
                
                image_url = response.data[0].url
                file_path = os.path.join(self.dirs['generated'], f"gen_{int(time.time())}_{i}.png")
                
                # Download image with timeout
                img_response = requests.get(image_url, timeout=30)
                img_response.raise_for_status()
                
                with open(file_path, 'wb') as f:
                    f.write(img_response.content)
                
                paths.append(file_path)
                logger.info(f"Generated image saved: {file_path}")
                
            except requests.exceptions.RequestException as e:
                logger.error(f"Network error generating image {i+1}: {e}")
                continue
            except Exception as e:
                logger.error(f"Error generating image {i+1}: {e}")
                continue
        
        if not paths:
            raise RuntimeError("Failed to generate any images")
        
        self.update_status(40, f"Generated {len(paths)} images successfully")
        return paths
    
    def analyze_and_sort(self, image_paths: List[str]) -> Dict[str, List[str]]:
        """Analyze images and sort with improved error handling"""
        if not self.client:
            raise ValueError("OpenRouter API client not initialized")
        
        results = {'top': [], 'mids': [], 'meh': [], 'errors': []}
        
        for idx, path in enumerate(image_paths):
            try:
                self.update_status(40 + (idx * 40 // len(image_paths)), 
                                 f"Analyzing image {idx+1}/{len(image_paths)}...")
                
                # Encode image to base64
                with open(path, "rb") as img_file:
                    base64_img = base64.b64encode(img_file.read()).decode('utf-8')
                
                # Analyze with vision model
                response = self.client.chat.completions.create(
                    model=self.vision_model,
                    messages=[{
                        "role": "user",
                        "content": [
                            {
                                "type": "text", 
                                "text": "Analyze this image for artistic quality, composition, and visual appeal. Provide a brief 2-3 word description and rate it 1-10. Format: 'Description: [words]. Rating: [number]'"
                            },
                            {
                                "type": "image_url", 
                                "image_url": {"url": f"data:image/png;base64,{base64_img}"}
                            }
                        ]
                    }],
                    max_tokens=100
                )
                
                analysis = response.choices[0].message.content.strip()
                logger.info(f"Analysis for {os.path.basename(path)}: {analysis}")
                
                # Parse response with fallback
                try:
                    if "Description:" in analysis and "Rating:" in analysis:
                        desc_part = analysis.split("Description:")[1].split("Rating:")[0].strip()
                        rating_part = analysis.split("Rating:")[1].strip()
                        
                        # Clean description
                        desc = desc_part.replace(".", "").replace(",", "").replace(" ", "_").lower()[:20]
                        
                        # Extract rating
                        rating = int(''.join(filter(str.isdigit, rating_part))[:1] or '5')
                        rating = max(1, min(10, rating))  # Clamp to 1-10
                    else:
                        desc = "analyzed_image"
                        rating = 5  # Default rating
                except:
                    desc = "analyzed_image"
                    rating = 5
                
                # Create new filename
                timestamp = int(time.time())
                new_name = f"{desc}_{timestamp}_({rating}).png"
                
                # Determine destination folder
                if rating >= 8:
                    dest_dir = self.dirs['top']
                    category = 'top'
                elif rating >= 6:
                    dest_dir = self.dirs['mids']
                    category = 'mids'
                else:
                    dest_dir = self.dirs['meh']
                    category = 'meh'
                
                new_path = os.path.join(dest_dir, new_name)
                
                # Move file
                os.rename(path, new_path)
                results[category].append({
                    'path': new_path,
                    'filename': new_name,
                    'rating': rating,
                    'description': desc.replace('_', ' ').title()
                })
                
                logger.info(f"Sorted {new_name} to {category} (rating: {rating})")
                
            except Exception as e:
                logger.error(f"Error analyzing {path}: {e}")
                results['errors'].append({
                    'path': path,
                    'error': str(e)
                })
                continue
        
        self.update_status(80, f"Analysis complete. Sorted {len(image_paths) - len(results['errors'])} images")
        return results
    
    def professional_upscale_image(self, image_path: str, scale_factor: int = 4, quality: str = 'best') -> str:
        """Professional AI upscaling with smart fallback system"""
        try:
            # Method 1: Try Professional Real-SR first (best quality, but heavy dependencies)
            if REALSR_AVAILABLE and professional_upscaler:
                try:
                    logger.info(f"Using Professional AI upscaling (Real-SR) - {quality} quality")
                    return professional_upscaler.professional_upscale(image_path, scale_factor, quality)
                except Exception as e:
                    logger.warning(f"Real-SR failed, trying API upscaler: {e}")
            
            # Method 2: Try API-based professional upscaling (lightweight, good quality)
            if API_UPSCALER_AVAILABLE and api_upscaler:
                try:
                    logger.info(f"Using API-based professional upscaling - {quality} quality")
                    return api_upscaler.professional_upscale(image_path, scale_factor, quality)
                except Exception as e:
                    logger.warning(f"API upscaler failed, falling back to traditional methods: {e}")
            
            # Method 3: Fallback to OpenCV/PIL
            logger.info("Using traditional upscaling methods")
            return self._fallback_upscale(image_path, scale_factor)
            
        except Exception as e:
            logger.error(f"All upscaling methods failed: {e}")
            raise
    
    def _fallback_upscale(self, image_path: str, scale_factor: int = 2) -> str:
        """Fallback upscaling using OpenCV or PIL"""
        try:
            if OPENCV_AVAILABLE:
                # Use OpenCV when available
                img = cv2.imread(image_path)
                if img is None:
                    raise ValueError(f"Could not load image: {image_path}")
                
                height, width = img.shape[:2]
                new_width = int(width * scale_factor)
                new_height = int(height * scale_factor)
                
                # Use INTER_CUBIC for better quality
                upscaled = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
                
                # Save upscaled image
                base_name = os.path.splitext(os.path.basename(image_path))[0]
                upscaled_path = os.path.join(self.dirs['upscaled'], f"{base_name}_OpenCV_{scale_factor}x.png")
                
                cv2.imwrite(upscaled_path, upscaled)
                logger.info(f"OpenCV fallback upscale complete: {upscaled_path}")
                return upscaled_path
            else:
                # Use PIL for upscaling when OpenCV is not available
                with Image.open(image_path) as img:
                    width, height = img.size
                    new_width = int(width * scale_factor)
                    new_height = int(height * scale_factor)
                    
                    # Use LANCZOS for better quality upscaling
                    upscaled = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                    
                    # Save upscaled image
                    base_name = os.path.splitext(os.path.basename(image_path))[0]
                    upscaled_path = os.path.join(self.dirs['upscaled'], f"{base_name}_PIL_{scale_factor}x.png")
                    
                    upscaled.save(upscaled_path, 'PNG')
                    logger.info(f"PIL fallback upscale complete: {upscaled_path}")
                    return upscaled_path
            
        except Exception as e:
            logger.error(f"Error in fallback upscaling: {e}")
            raise
    
    # Keep old method name for compatibility
    def simple_upscale(self, image_path: str, scale_factor: int = 2) -> str:
        """Legacy method - redirects to professional upscaling"""
        return self.professional_upscale_image(image_path, scale_factor, 'fast')
    
    def run_workflow(self, progress_callback=None) -> Dict:
        """Run complete workflow with progress tracking"""
        try:
            self.update_status(0, "Starting workflow...")
            
            # Generate images
            image_paths = self.generate_batch()
            
            # Analyze and sort
            results = self.analyze_and_sort(image_paths)
            
            self.update_status(100, "Workflow completed successfully!")
            
            return {
                'success': True,
                'results': results,
                'total_generated': len(image_paths),
                'directories': self.dirs
            }
            
        except Exception as e:
            error_msg = f"Workflow failed: {str(e)}"
            logger.error(error_msg)
            self.update_status(0, "Workflow failed", error_msg)
            return {
                'success': False,
                'error': error_msg,
                'traceback': traceback.format_exc()
            }

# Global workflow instance
workflow_automator = None

class ImageEditor:
    """Advanced image editing and inpainting using OpenRouter APIs"""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key
        self.client = None
        self.edit_dir = os.path.join(os.getcwd(), 'ImageWorkflow', 'Edited')
        os.makedirs(self.edit_dir, exist_ok=True)
        
        if api_key and api_key.strip() and api_key != "your_openrouter_key_here":
            try:
                from openai import OpenAI
                self.client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")
                logger.info("Image Editor OpenRouter client initialized")
            except Exception as e:
                logger.error(f"Failed to initialize Image Editor client: {e}")
                self.client = None
    
    def create_mask_from_coordinates(self, image_path: str, mask_data: List[Dict]) -> str:
        """Create a mask image from drawing coordinates"""
        try:
            # Load original image to get dimensions
            with Image.open(image_path) as img:
                width, height = img.size
            
            # Create mask image (white background, black for areas to edit)
            mask = Image.new('RGB', (width, height), 'white')
            draw = ImageDraw.Draw(mask)
            
            # Draw mask areas in black
            for stroke in mask_data:
                if stroke['type'] == 'brush':
                    points = [(point['x'] * width, point['y'] * height) for point in stroke['points']]
                    if len(points) > 1:
                        # Draw thick lines for brush strokes
                        for i in range(len(points) - 1):
                            draw.line([points[i], points[i + 1]], fill='black', width=stroke.get('size', 20))
                    elif len(points) == 1:
                        # Draw circle for single points
                        x, y = points[0]
                        r = stroke.get('size', 20) // 2
                        draw.ellipse([x-r, y-r, x+r, y+r], fill='black')
            
            # Save mask
            mask_path = os.path.join(self.edit_dir, f"mask_{int(time.time())}.png")
            mask.save(mask_path)
            logger.info(f"Created mask: {mask_path}")
            return mask_path
            
        except Exception as e:
            logger.error(f"Error creating mask: {e}")
            raise
    
    def inpaint_image(self, image_path: str, mask_path: str, prompt: str, 
                     model: str = "stability-ai/stable-diffusion-xl-base-1.0") -> str:
        """Perform inpainting using OpenRouter API"""
        if not self.client:
            raise ValueError("OpenRouter client not initialized")
        
        try:
            # Convert images to base64
            with open(image_path, "rb") as img_file:
                image_b64 = base64.b64encode(img_file.read()).decode('utf-8')
            
            with open(mask_path, "rb") as mask_file:
                mask_b64 = base64.b64encode(mask_file.read()).decode('utf-8')
            
            # Make API call for inpainting
            response = self.client.images.edit(
                image=f"data:image/png;base64,{image_b64}",
                mask=f"data:image/png;base64,{mask_b64}",
                prompt=prompt,
                n=1,
                size="1024x1024"
            )
            
            if not response.data:
                raise RuntimeError("No image data received from API")
            
            # Download result
            result_url = response.data[0].url
            img_response = requests.get(result_url, timeout=30)
            img_response.raise_for_status()
            
            # Save result
            result_path = os.path.join(self.edit_dir, f"inpaint_{int(time.time())}.png")
            with open(result_path, 'wb') as f:
                f.write(img_response.content)
            
            logger.info(f"Inpainting complete: {result_path}")
            return result_path
            
        except Exception as e:
            logger.error(f"Error in inpainting: {e}")
            raise
    
    def image_to_image(self, image_path: str, prompt: str, strength: float = 0.7,
                      model: str = "stability-ai/stable-diffusion-xl-base-1.0") -> str:
        """Perform image-to-image transformation"""
        if not self.client:
            raise ValueError("OpenRouter client not initialized")
        
        try:
            with open(image_path, "rb") as img_file:
                image_b64 = base64.b64encode(img_file.read()).decode('utf-8')
            
            # Use chat completion with vision for image-to-image style editing
            response = self.client.chat.completions.create(
                model="x-ai/grok-vision-beta",
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"Transform this image according to this description: {prompt}. Maintain the overall composition but apply the requested changes. Describe the transformation you would make in detail."
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{image_b64}"}
                        }
                    ]
                }],
                max_tokens=500
            )
            
            transformation_desc = response.choices[0].message.content
            
            # Generate new image based on transformation description
            gen_response = self.client.images.generate(
                model=model,
                prompt=f"{transformation_desc}. {prompt}",
                n=1,
                size="1024x1024"
            )
            
            if not gen_response.data:
                raise RuntimeError("No image data received from generation API")
            
            # Download result
            result_url = gen_response.data[0].url
            img_response = requests.get(result_url, timeout=30)
            img_response.raise_for_status()
            
            # Save result
            result_path = os.path.join(self.edit_dir, f"img2img_{int(time.time())}.png")
            with open(result_path, 'wb') as f:
                f.write(img_response.content)
            
            logger.info(f"Image-to-image complete: {result_path}")
            return result_path
            
        except Exception as e:
            logger.error(f"Error in image-to-image: {e}")
            raise
    
    def remove_background(self, image_path: str) -> str:
        """Remove background using AI vision analysis and generation"""
        if not self.client:
            raise ValueError("OpenRouter client not initialized")
        
        try:
            with open(image_path, "rb") as img_file:
                image_b64 = base64.b64encode(img_file.read()).decode('utf-8')
            
            # Analyze image to identify main subject
            response = self.client.chat.completions.create(
                model="x-ai/grok-vision-beta",
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Describe the main subject/object in this image that should be kept when removing the background. Be very specific about what should remain."
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{image_b64}"}
                        }
                    ]
                }],
                max_tokens=200
            )
            
            subject_desc = response.choices[0].message.content
            
            # Generate image with transparent/white background
            gen_response = self.client.images.generate(
                model="stability-ai/stable-diffusion-xl-base-1.0",
                prompt=f"{subject_desc} on a pure white background, isolated object, clean cutout, professional product photo style",
                n=1,
                size="1024x1024"
            )
            
            if not gen_response.data:
                raise RuntimeError("No image data received from generation API")
            
            # Download result
            result_url = gen_response.data[0].url
            img_response = requests.get(result_url, timeout=30)
            img_response.raise_for_status()
            
            # Save result
            result_path = os.path.join(self.edit_dir, f"nobg_{int(time.time())}.png")
            with open(result_path, 'wb') as f:
                f.write(img_response.content)
            
            logger.info(f"Background removal complete: {result_path}")
            return result_path
            
        except Exception as e:
            logger.error(f"Error in background removal: {e}")
            raise

# Global image editor instance
image_editor = None

@app.route('/')
def home():
    """Main dashboard page optimized for tablet"""
    global workflow_automator, image_editor
    
    # Check if workflow is configured
    workflow_configured = workflow_automator is not None and workflow_automator.client is not None
    editor_configured = image_editor is not None and image_editor.client is not None
    
    return render_template('index.html', 
                         notes=notes, 
                         tasks=tasks,
                         current_time=datetime.now().strftime('%Y-%m-%d %H:%M'),
                         workflow_configured=workflow_configured,
                         editor_configured=editor_configured,
                         workflow_status=workflow_status)

@app.route('/add_note', methods=['POST'])
def add_note():
    """Add a new note"""
    content = request.form.get('content')
    if content:
        note = {
            'id': len(notes) + 1,
            'content': content,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        notes.append(note)
    return jsonify({'success': True, 'notes': notes})

@app.route('/add_task', methods=['POST'])
def add_task():
    """Add a new task"""
    task_name = request.form.get('task')
    if task_name:
        task = {
            'id': len(tasks) + 1,
            'name': task_name,
            'completed': False,
            'created': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        tasks.append(task)
    return jsonify({'success': True, 'tasks': tasks})

@app.route('/toggle_task/<int:task_id>', methods=['POST'])
def toggle_task(task_id):
    """Toggle task completion status"""
    for task in tasks:
        if task['id'] == task_id:
            task['completed'] = not task['completed']
            break
    return jsonify({'success': True, 'tasks': tasks})

@app.route('/delete_note/<int:note_id>', methods=['POST'])
def delete_note(note_id):
    """Delete a note"""
    global notes
    notes = [note for note in notes if note['id'] != note_id]
    return jsonify({'success': True, 'notes': notes})

@app.route('/delete_task/<int:task_id>', methods=['POST'])
def delete_task(task_id):
    """Delete a task"""
    global tasks
    tasks = [task for task in tasks if task['id'] != task_id]
    return jsonify({'success': True, 'tasks': tasks})

@app.route('/system_info')
def system_info():
    """Display system information"""
    import platform
    import sys
    import flask
    
    info = {
        'platform': platform.platform(),
        'python_version': sys.version,
        'flask_version': flask.__version__,
        'current_directory': os.getcwd(),
        'environment': dict(os.environ)
    }
    return render_template('system_info.html', info=info)

# Image Workflow Routes
@app.route('/workflow/config', methods=['GET', 'POST'])
def workflow_config():
    """Configure workflow settings"""
    global workflow_automator, image_editor
    
    if request.method == 'POST':
        try:
            # Get API key from form or environment
            api_key = request.form.get('api_key', '').strip()
            if not api_key:
                api_key = os.getenv('OPENROUTER_API_KEY', '').strip()
            
            batch_size = int(request.form.get('batch_size', 3))
            base_prompt = request.form.get('base_prompt', 'A beautiful landscape in artistic style').strip()
            gen_model = request.form.get('gen_model', os.getenv('DEFAULT_GEN_MODEL', 'black-forest-labs/flux-schnell'))
            vision_model = request.form.get('vision_model', os.getenv('DEFAULT_VISION_MODEL', 'x-ai/grok-vision-beta'))
            
            # Safety parameters
            safety_level = request.form.get('safety_level', 'moderate')
            content_filter = request.form.get('content_filter') == 'on'
            
            if not api_key:
                flash('API key is required', 'error')
                return redirect(url_for('workflow_config'))
            
            # Initialize workflow automator with safety parameters
            workflow_automator = ImageWorkflowAutomator(
                api_key=api_key,
                batch_size=batch_size,
                base_prompt=base_prompt,
                gen_model=gen_model,
                vision_model=vision_model,
                safety_level=safety_level,
                content_filter=content_filter
            )
            
            # Initialize image editor with same API key
            image_editor = ImageEditor(api_key=api_key)
            
            if workflow_automator.client:
                flash('Workflow and Image Editor configured successfully!', 'success')
                logger.info("Workflow automator and image editor configured successfully")
            else:
                flash('Failed to initialize API client. Check your API key.', 'error')
                workflow_automator = None
                image_editor = None
                
        except Exception as e:
            flash(f'Configuration error: {str(e)}', 'error')
            logger.error(f"Workflow configuration error: {e}")
            workflow_automator = None
            image_editor = None
        
        return redirect(url_for('home'))
    
    # GET request - show configuration form
    current_config = {}
    if workflow_automator:
        current_config = {
            'batch_size': workflow_automator.batch_size,
            'base_prompt': workflow_automator.base_prompt,
            'gen_model': workflow_automator.gen_model,
            'vision_model': workflow_automator.vision_model,
            'safety_level': getattr(workflow_automator, 'safety_level', 'moderate'),
            'content_filter': getattr(workflow_automator, 'content_filter', True)
        }
    
    # Add environment variables for form defaults
    env_config = {
        'env_api_key': os.getenv('OPENROUTER_API_KEY', ''),
        'env_gen_model': os.getenv('DEFAULT_GEN_MODEL', 'black-forest-labs/flux-schnell'),
        'env_vision_model': os.getenv('DEFAULT_VISION_MODEL', 'x-ai/grok-vision-beta'),
        'env_safety_level': os.getenv('DEFAULT_SAFETY_LEVEL', 'moderate')
    }
    
    return render_template('workflow_config.html', config=current_config, env_config=env_config)

@app.route('/workflow/start', methods=['POST'])
def start_workflow():
    """Start the image generation workflow"""
    global workflow_automator, workflow_status
    
    if not workflow_automator or not workflow_automator.client:
        return jsonify({'success': False, 'error': 'Workflow not configured. Please set up API key first.'})
    
    if workflow_status['running']:
        return jsonify({'success': False, 'error': 'Workflow is already running'})
    
    try:
        # Reset status
        workflow_status.update({
            'running': True,
            'progress': 0,
            'message': 'Initializing workflow...',
            'error': None,
            'results': []
        })
        
        # Run workflow in background thread
        def run_workflow_thread():
            try:
                result = workflow_automator.run_workflow()
                workflow_status.update({
                    'running': False,
                    'results': result
                })
            except Exception as e:
                workflow_status.update({
                    'running': False,
                    'error': str(e)
                })
        
        thread = threading.Thread(target=run_workflow_thread)
        thread.daemon = True
        thread.start()
        
        return jsonify({'success': True, 'message': 'Workflow started successfully'})
        
    except Exception as e:
        workflow_status['running'] = False
        logger.error(f"Error starting workflow: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/workflow/status')
def workflow_status_api():
    """Get current workflow status"""
    return jsonify(workflow_status)

@app.route('/workflow/gallery')
def workflow_gallery():
    """Display generated and sorted images"""
    global workflow_automator
    
    if not workflow_automator:
        flash('Workflow not configured', 'error')
        return redirect(url_for('home'))
    
    # Scan directories for images
    gallery_data = {}
    for category, dir_path in workflow_automator.dirs.items():
        if category == 'generated':  # Skip generated folder in gallery
            continue
            
        images = []
        if os.path.exists(dir_path):
            for filename in os.listdir(dir_path):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    images.append({
                        'filename': filename,
                        'path': os.path.join(dir_path, filename),
                        'url': url_for('serve_image', category=category, filename=filename)
                    })
        
        gallery_data[category] = images
    
    return render_template('workflow_gallery.html', gallery=gallery_data)

@app.route('/workflow/images/<category>/<filename>')
def serve_image(category, filename):
    """Serve images from workflow directories"""
    global workflow_automator
    
    if not workflow_automator or category not in workflow_automator.dirs:
        return "Image not found", 404
    
    directory = workflow_automator.dirs[category]
    return send_from_directory(directory, filename)

@app.route('/workflow/upscale', methods=['POST'])
def upscale_image():
    """Professional AI upscaling with Real-SR for income generation"""
    global workflow_automator
    
    if not workflow_automator:
        return jsonify({'success': False, 'error': 'Workflow not configured'})
    
    try:
        image_path = request.form.get('image_path')
        scale_factor = int(request.form.get('scale_factor', 4))  # Default to 4x for professional quality
        quality = request.form.get('quality', 'best')  # 'best', 'fast', 'balanced'
        
        if not image_path or not os.path.exists(image_path):
            return jsonify({'success': False, 'error': 'Image not found'})
        
        # Professional AI upscaling with Real-SR
        upscaled_path = workflow_automator.professional_upscale_image(image_path, scale_factor, quality)
        
        # Determine method used for client feedback
        method = "Real-SR AI" if REALSR_AVAILABLE and professional_upscaler else "Traditional"
        
        return jsonify({
            'success': True, 
            'message': f'Image upscaled successfully using {method} ({quality} quality)',
            'upscaled_path': upscaled_path,
            'method': method,
            'quality': quality,
            'scale_factor': scale_factor,
            'professional_mode': REALSR_AVAILABLE,
            'upscaled_url': url_for('serve_image', 
                                  category='upscaled', 
                                  filename=os.path.basename(upscaled_path))
        })
        
    except Exception as e:
        logger.error(f"Error upscaling image: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/workflow/upscale/status', methods=['GET'])
def upscale_status():
    """Get professional upscaling capabilities status"""
    return jsonify({
        'realsr_available': REALSR_AVAILABLE,
        'api_upscaler_available': API_UPSCALER_AVAILABLE,
        'opencv_available': OPENCV_AVAILABLE,
        'professional_mode': REALSR_AVAILABLE or API_UPSCALER_AVAILABLE,
        'lightweight_professional': API_UPSCALER_AVAILABLE and not REALSR_AVAILABLE,
        'supported_qualities': ['best', 'fast', 'balanced'] if (REALSR_AVAILABLE or API_UPSCALER_AVAILABLE) else ['standard'],
        'max_scale_factor': 4 if (REALSR_AVAILABLE or API_UPSCALER_AVAILABLE) else 8,
        'recommended_for_income': REALSR_AVAILABLE or API_UPSCALER_AVAILABLE,
        'dependency_info': {
            'heavy_ml_deps': REALSR_AVAILABLE,  # PyTorch, torchvision, etc.
            'lightweight_api': API_UPSCALER_AVAILABLE,  # No heavy dependencies
            'basic_only': not (REALSR_AVAILABLE or API_UPSCALER_AVAILABLE)
        },
        'model_status': {
            'x4_professional': 'x4_professional' in (professional_upscaler.realsr_models if professional_upscaler else {}),
            'x2_fast': 'x2_fast' in (professional_upscaler.realsr_models if professional_upscaler else {}),
            'api_waifu2x': API_UPSCALER_AVAILABLE,
            'enhanced_local': API_UPSCALER_AVAILABLE
        }
    })

@app.route('/workflow/delete', methods=['POST'])
def delete_image():
    """Delete selected image"""
    try:
        image_path = request.form.get('image_path')
        
        if not image_path or not os.path.exists(image_path):
            return jsonify({'success': False, 'error': 'Image not found'})
        
        os.remove(image_path)
        logger.info(f"Deleted image: {image_path}")
        
        return jsonify({'success': True, 'message': 'Image deleted successfully'})
        
    except Exception as e:
        logger.error(f"Error deleting image: {e}")
        return jsonify({'success': False, 'error': str(e)})

# Image Editing Routes
@app.route('/editor')
def image_editor_page():
    """Image editing interface"""
    global image_editor, workflow_automator
    
    if not image_editor or not image_editor.client:
        flash('Image editor not configured. Please set up your API key first.', 'error')
        return redirect(url_for('workflow_config'))
    
    # Get available images from workflow directories
    available_images = []
    if workflow_automator:
        for category, dir_path in workflow_automator.dirs.items():
            if category == 'generated':
                continue
            if os.path.exists(dir_path):
                for filename in os.listdir(dir_path):
                    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                        available_images.append({
                            'filename': filename,
                            'path': os.path.join(dir_path, filename),
                            'category': category,
                            'url': url_for('serve_image', category=category, filename=filename)
                        })
    
    # Also check edited directory
    if os.path.exists(image_editor.edit_dir):
        for filename in os.listdir(image_editor.edit_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')) and not filename.startswith('mask_'):
                available_images.append({
                    'filename': filename,
                    'path': os.path.join(image_editor.edit_dir, filename),
                    'category': 'edited',
                    'url': url_for('serve_edited_image', filename=filename)
                })
    
    return render_template('image_editor.html', images=available_images)

@app.route('/editor/upload', methods=['POST'])
def upload_image():
    """Upload image for editing"""
    global image_editor
    
    if not image_editor:
        return jsonify({'success': False, 'error': 'Image editor not configured'})
    
    try:
        if 'image' not in request.files:
            return jsonify({'success': False, 'error': 'No image file provided'})
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'})
        
        if file and file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            filename = secure_filename(file.filename)
            timestamp = int(time.time())
            filename = f"upload_{timestamp}_{filename}"
            filepath = os.path.join(image_editor.edit_dir, filename)
            file.save(filepath)
            
            return jsonify({
                'success': True,
                'filename': filename,
                'url': url_for('serve_edited_image', filename=filename),
                'path': filepath
            })
        else:
            return jsonify({'success': False, 'error': 'Invalid file type. Please upload PNG, JPG, or JPEG files.'})
            
    except Exception as e:
        logger.error(f"Error uploading image: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/editor/inpaint', methods=['POST'])
def inpaint_image_route():
    """Perform inpainting on image"""
    global image_editor
    
    if not image_editor or not image_editor.client:
        return jsonify({'success': False, 'error': 'Image editor not configured'})
    
    try:
        data = request.get_json()
        image_path = data.get('image_path')
        mask_data = data.get('mask_data', [])
        prompt = data.get('prompt', '').strip()
        
        if not image_path or not os.path.exists(image_path):
            return jsonify({'success': False, 'error': 'Image not found'})
        
        if not prompt:
            return jsonify({'success': False, 'error': 'Prompt is required for inpainting'})
        
        if not mask_data:
            return jsonify({'success': False, 'error': 'Please draw on the image to select areas to edit'})
        
        # Create mask from drawing data
        mask_path = image_editor.create_mask_from_coordinates(image_path, mask_data)
        
        # Perform inpainting
        result_path = image_editor.inpaint_image(image_path, mask_path, prompt)
        
        return jsonify({
            'success': True,
            'result_url': url_for('serve_edited_image', filename=os.path.basename(result_path)),
            'result_path': result_path
        })
        
    except Exception as e:
        logger.error(f"Error in inpainting: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/editor/img2img', methods=['POST'])
def image_to_image_route():
    """Perform image-to-image transformation"""
    global image_editor
    
    if not image_editor or not image_editor.client:
        return jsonify({'success': False, 'error': 'Image editor not configured'})
    
    try:
        data = request.get_json()
        image_path = data.get('image_path')
        prompt = data.get('prompt', '').strip()
        strength = float(data.get('strength', 0.7))
        
        if not image_path or not os.path.exists(image_path):
            return jsonify({'success': False, 'error': 'Image not found'})
        
        if not prompt:
            return jsonify({'success': False, 'error': 'Prompt is required for transformation'})
        
        # Perform image-to-image transformation
        result_path = image_editor.image_to_image(image_path, prompt, strength)
        
        return jsonify({
            'success': True,
            'result_url': url_for('serve_edited_image', filename=os.path.basename(result_path)),
            'result_path': result_path
        })
        
    except Exception as e:
        logger.error(f"Error in image-to-image: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/editor/remove_bg', methods=['POST'])
def remove_background_route():
    """Remove background from image"""
    global image_editor
    
    if not image_editor or not image_editor.client:
        return jsonify({'success': False, 'error': 'Image editor not configured'})
    
    try:
        data = request.get_json()
        image_path = data.get('image_path')
        
        if not image_path or not os.path.exists(image_path):
            return jsonify({'success': False, 'error': 'Image not found'})
        
        # Remove background
        result_path = image_editor.remove_background(image_path)
        
        return jsonify({
            'success': True,
            'result_url': url_for('serve_edited_image', filename=os.path.basename(result_path)),
            'result_path': result_path
        })
        
    except Exception as e:
        logger.error(f"Error removing background: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/editor/images/<filename>')
def serve_edited_image(filename):
    """Serve images from edited directory"""
    global image_editor
    
    if not image_editor:
        return "Image editor not configured", 404
    
    return send_from_directory(image_editor.edit_dir, filename)

# Image Generator Routes
@app.route('/generator')
def image_generator_page():
    """Advanced image generation interface"""
    global workflow_automator
    
    if not workflow_automator or not workflow_automator.client:
        flash('Image generator not configured. Please set up your API key first.', 'error')
        return redirect(url_for('workflow_config'))
    
    return render_template('image_generator.html')

@app.route('/generator/single', methods=['POST'])
def generate_single_image():
    """Generate a single image with custom settings"""
    global workflow_automator
    
    if not workflow_automator or not workflow_automator.client:
        return jsonify({'success': False, 'error': 'Generator not configured'})
    
    try:
        prompt = request.form.get('prompt', '').strip()
        model = request.form.get('model', 'black-forest-labs/flux-schnell')
        size = request.form.get('size', '1024x1024')
        safety_level = request.form.get('safety_level', 'moderate')
        content_filter = request.form.get('content_filter') == 'on'
        
        if not prompt:
            return jsonify({'success': False, 'error': 'Prompt is required'})
        
        # Create temporary automator with custom settings
        temp_automator = ImageWorkflowAutomator(
            api_key=workflow_automator.api_key,
            batch_size=1,
            base_prompt=prompt,
            gen_model=model,
            vision_model=workflow_automator.vision_model,
            safety_level=safety_level,
            content_filter=content_filter
        )
        
        # Generate single image
        safety_params = temp_automator.get_safety_parameters()
        generation_params = {
            "model": model,
            "prompt": prompt,
            "n": 1,
            "size": size
        }
        generation_params.update(safety_params)
        
        response = temp_automator.client.images.generate(**generation_params)
        
        if not response.data:
            return jsonify({'success': False, 'error': 'No image data received'})
        
        # Download and save image
        image_url = response.data[0].url
        timestamp = int(time.time())
        filename = f"single_gen_{timestamp}.png"
        file_path = os.path.join(temp_automator.dirs['generated'], filename)
        
        img_response = requests.get(image_url, timeout=30)
        img_response.raise_for_status()
        
        with open(file_path, 'wb') as f:
            f.write(img_response.content)
        
        return jsonify({
            'success': True,
            'result': {
                'url': url_for('serve_image', category='generated', filename=filename),
                'path': file_path,
                'prompt': prompt,
                'model': model,
                'safety_level': safety_level
            }
        })
        
    except Exception as e:
        logger.error(f"Error in single image generation: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/generator/batch', methods=['POST'])
def generate_batch_images():
    """Generate multiple images with variations"""
    global workflow_automator
    
    if not workflow_automator or not workflow_automator.client:
        return jsonify({'success': False, 'error': 'Generator not configured'})
    
    try:
        base_prompt = request.form.get('base_prompt', '').strip()
        variations_text = request.form.get('variations', '').strip()
        count = int(request.form.get('count', 3))
        model = request.form.get('model', 'black-forest-labs/flux-schnell')
        safety_level = request.form.get('safety_level', 'moderate')
        auto_analyze = request.form.get('auto_analyze') == 'on'
        
        if not base_prompt:
            return jsonify({'success': False, 'error': 'Base prompt is required'})
        
        # Parse variations
        variations = []
        if variations_text:
            variations = [v.strip().lstrip('- ') for v in variations_text.split('\n') if v.strip()]
        
        # Generate default variations if none provided
        if not variations:
            variations = [
                "in artistic style",
                "with dramatic lighting",
                "in photorealistic style",
                "with vibrant colors",
                "in digital art style"
            ]
        
        # Limit to requested count
        variations = variations[:count]
        while len(variations) < count:
            variations.append(f"variation {len(variations) + 1}")
        
        # Create temporary automator
        temp_automator = ImageWorkflowAutomator(
            api_key=workflow_automator.api_key,
            batch_size=count,
            base_prompt=base_prompt,
            gen_model=model,
            vision_model=workflow_automator.vision_model,
            safety_level=safety_level,
            content_filter=True
        )
        
        results = []
        safety_params = temp_automator.get_safety_parameters()
        
        for i, variation in enumerate(variations):
            try:
                full_prompt = f"{base_prompt} {variation}"
                generation_params = {
                    "model": model,
                    "prompt": full_prompt,
                    "n": 1,
                    "size": "1024x1024"
                }
                generation_params.update(safety_params)
                
                response = temp_automator.client.images.generate(**generation_params)
                
                if response.data:
                    image_url = response.data[0].url
                    timestamp = int(time.time())
                    filename = f"batch_gen_{timestamp}_{i}.png"
                    file_path = os.path.join(temp_automator.dirs['generated'], filename)
                    
                    img_response = requests.get(image_url, timeout=30)
                    img_response.raise_for_status()
                    
                    with open(file_path, 'wb') as f:
                        f.write(img_response.content)
                    
                    result = {
                        'url': url_for('serve_image', category='generated', filename=filename),
                        'path': file_path,
                        'prompt': full_prompt,
                        'variation': variation
                    }
                    
                    # Auto-analyze if requested
                    if auto_analyze:
                        try:
                            rating = temp_automator.analyze_image(file_path)
                            result['rating'] = rating
                        except:
                            result['rating'] = 'N/A'
                    
                    results.append(result)
                    
            except Exception as e:
                logger.error(f"Error generating batch image {i}: {e}")
                continue
        
        return jsonify({
            'success': True,
            'results': results,
            'count': len(results)
        })
        
    except Exception as e:
        logger.error(f"Error in batch generation: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/generator/advanced', methods=['POST'])
def generate_advanced_image():
    """Generate image with advanced parameters"""
    global workflow_automator
    
    if not workflow_automator or not workflow_automator.client:
        return jsonify({'success': False, 'error': 'Generator not configured'})
    
    try:
        prompt = request.form.get('prompt', '').strip()
        negative_prompt = request.form.get('negative_prompt', '').strip()
        model = request.form.get('model', 'black-forest-labs/flux-schnell')
        size = request.form.get('size', '1024x1024')
        guidance_scale = float(request.form.get('guidance_scale', 7.5))
        steps = int(request.form.get('steps', 30))
        seed = request.form.get('seed', '').strip()
        safety_level = request.form.get('safety_level', 'moderate')
        content_filter = request.form.get('content_filter') == 'on'
        safe_mode = request.form.get('safe_mode') == 'on'
        
        if not prompt:
            return jsonify({'success': False, 'error': 'Prompt is required'})
        
        # Create temporary automator
        custom_safety_params = {}
        if safe_mode:
            custom_safety_params['safe_mode'] = True
        
        temp_automator = ImageWorkflowAutomator(
            api_key=workflow_automator.api_key,
            batch_size=1,
            base_prompt=prompt,
            gen_model=model,
            vision_model=workflow_automator.vision_model,
            safety_level=safety_level,
            content_filter=content_filter,
            custom_safety_params=custom_safety_params
        )
        
        # Build generation parameters
        safety_params = temp_automator.get_safety_parameters()
        generation_params = {
            "model": model,
            "prompt": prompt,
            "n": 1,
            "size": size
        }
        
        # Add advanced parameters (note: not all models support all parameters)
        if negative_prompt:
            generation_params["negative_prompt"] = negative_prompt
        if guidance_scale != 7.5:
            generation_params["guidance_scale"] = guidance_scale
        if steps != 30:
            generation_params["num_inference_steps"] = steps
        if seed:
            generation_params["seed"] = int(seed)
        
        generation_params.update(safety_params)
        
        response = temp_automator.client.images.generate(**generation_params)
        
        if not response.data:
            return jsonify({'success': False, 'error': 'No image data received'})
        
        # Download and save image
        image_url = response.data[0].url
        timestamp = int(time.time())
        filename = f"advanced_gen_{timestamp}.png"
        file_path = os.path.join(temp_automator.dirs['generated'], filename)
        
        img_response = requests.get(image_url, timeout=30)
        img_response.raise_for_status()
        
        with open(file_path, 'wb') as f:
            f.write(img_response.content)
        
        return jsonify({
            'success': True,
            'result': {
                'url': url_for('serve_image', category='generated', filename=filename),
                'path': file_path,
                'prompt': prompt,
                'negative_prompt': negative_prompt,
                'model': model,
                'safety_level': safety_level,
                'guidance_scale': guidance_scale,
                'steps': steps,
                'seed': seed if seed else 'random'
            }
        })
        
    except Exception as e:
        logger.error(f"Error in advanced generation: {e}")
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    # Configuration for Termux and tablet use
    port = int(os.environ.get('PORT', 5000))
    
    print(f"Starting Flask app on port {port}")
    print("Access your app at:")
    print(f"  Local: http://localhost:{port}")
    print(f"  Network: http://0.0.0.0:{port}")
    print("\nPress Ctrl+C to stop the server")
    
    app.run(
        host='0.0.0.0',  # Allow access from any IP
        port=port,
        debug=True,      # Enable debug mode for development
        threaded=True    # Handle multiple requests
    )