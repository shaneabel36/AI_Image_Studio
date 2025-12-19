"""
API-Based Professional Upscaling
Lightweight alternative to Real-SR for income generation
No PyTorch/heavy dependencies required
"""

import os
import requests
import base64
import logging
from PIL import Image
import io
from typing import Optional

logger = logging.getLogger(__name__)

class APIUpscaler:
    """Professional upscaling using external APIs - no heavy dependencies"""
    
    def __init__(self):
        self.apis = {
            'waifu2x': {
                'url': 'https://api.waifu2x.udp.jp/api',
                'free': True,
                'max_size': 1500,
                'formats': ['jpg', 'png', 'webp']
            },
            'upscayl': {
                'url': 'https://upscayl.org/api',
                'free': True,
                'max_size': 2000,
                'formats': ['jpg', 'png']
            }
        }
    
    def upscale_with_waifu2x(self, image_path: str, scale_factor: int = 2, noise_reduction: int = 1) -> Optional[str]:
        """
        Upscale using Waifu2x API (free, good for anime/illustrations)
        """
        try:
            # Check image size
            with Image.open(image_path) as img:
                width, height = img.size
                if max(width, height) > self.apis['waifu2x']['max_size']:
                    logger.warning(f"Image too large for Waifu2x API: {width}x{height}")
                    return None
                
                # Convert to supported format if needed
                if img.format.lower() not in self.apis['waifu2x']['formats']:
                    img = img.convert('RGB')
                
                # Convert to base64
                buffer = io.BytesIO()
                img.save(buffer, format='PNG')
                img_base64 = base64.b64encode(buffer.getvalue()).decode()
            
            # API request
            data = {
                'image': img_base64,
                'scale': scale_factor,
                'noise': noise_reduction,
                'style': 'art'  # or 'photo'
            }
            
            response = requests.post(
                f"{self.apis['waifu2x']['url']}/upscale",
                json=data,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                if 'image' in result:
                    # Decode and save result
                    upscaled_data = base64.b64decode(result['image'])
                    
                    base_name = os.path.splitext(os.path.basename(image_path))[0]
                    output_path = os.path.join(
                        os.path.dirname(image_path).replace('Generated', 'Upscaled'),
                        f"{base_name}_Waifu2x_{scale_factor}x.png"
                    )
                    
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)
                    
                    with open(output_path, 'wb') as f:
                        f.write(upscaled_data)
                    
                    logger.info(f"Waifu2x upscaling complete: {output_path}")
                    return output_path
            
            logger.error(f"Waifu2x API error: {response.status_code}")
            return None
            
        except Exception as e:
            logger.error(f"Waifu2x upscaling failed: {e}")
            return None
    
    def upscale_with_local_algorithms(self, image_path: str, scale_factor: int = 2, algorithm: str = 'lanczos') -> str:
        """
        High-quality local upscaling using advanced PIL algorithms
        No external dependencies, works offline
        """
        try:
            with Image.open(image_path) as img:
                width, height = img.size
                new_width = int(width * scale_factor)
                new_height = int(height * scale_factor)
                
                # Choose resampling algorithm
                algorithms = {
                    'lanczos': Image.Resampling.LANCZOS,
                    'bicubic': Image.Resampling.BICUBIC,
                    'hamming': Image.Resampling.HAMMING,
                    'box': Image.Resampling.BOX
                }
                
                resampling = algorithms.get(algorithm, Image.Resampling.LANCZOS)
                
                # Apply sharpening for better results
                from PIL import ImageFilter, ImageEnhance
                
                # Pre-process: slight sharpening
                enhanced = img.filter(ImageFilter.UnsharpMask(radius=0.5, percent=150, threshold=3))
                
                # Upscale
                upscaled = enhanced.resize((new_width, new_height), resampling)
                
                # Post-process: enhance contrast slightly
                enhancer = ImageEnhance.Contrast(upscaled)
                upscaled = enhancer.enhance(1.1)
                
                # Save result
                base_name = os.path.splitext(os.path.basename(image_path))[0]
                output_path = os.path.join(
                    os.path.dirname(image_path).replace('Generated', 'Upscaled'),
                    f"{base_name}_Enhanced_{algorithm}_{scale_factor}x.png"
                )
                
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                upscaled.save(output_path, 'PNG', optimize=True)
                
                logger.info(f"Enhanced local upscaling complete: {output_path}")
                return output_path
                
        except Exception as e:
            logger.error(f"Enhanced local upscaling failed: {e}")
            raise
    
    def professional_upscale(self, image_path: str, scale_factor: int = 2, quality: str = 'best') -> str:
        """
        Professional upscaling with multiple fallback methods
        Optimized for income generation without heavy dependencies
        """
        try:
            # Method 1: Try API upscaling for best quality
            if quality == 'best':
                api_result = self.upscale_with_waifu2x(image_path, scale_factor)
                if api_result:
                    return api_result
                logger.info("API upscaling failed, falling back to enhanced local")
            
            # Method 2: Enhanced local algorithms
            if quality in ['best', 'balanced']:
                return self.upscale_with_local_algorithms(image_path, scale_factor, 'lanczos')
            else:
                return self.upscale_with_local_algorithms(image_path, scale_factor, 'bicubic')
                
        except Exception as e:
            logger.error(f"All professional upscaling methods failed: {e}")
            raise

# Global instance
api_upscaler = APIUpscaler()