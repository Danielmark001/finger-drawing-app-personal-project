import cv2
import os
import numpy as np
from datetime import datetime
import time

def ensure_save_directory(directory="drawings"):
    """Ensure the save directory exists."""
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")
    return directory

def generate_filename(prefix="drawing", extension="png"):
    """Generate a filename with timestamp."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{timestamp}.{extension}"

def save_as_png(canvas, directory="drawings", filename=None):
    """Save the canvas as a PNG file with transparency."""
    if filename is None:
        filename = generate_filename("drawing", "png")
    
    # Ensure directory exists
    directory = ensure_save_directory(directory)
    filepath = os.path.join(directory, filename)
    
    # Convert BGRA to RGBA for proper saving
    if canvas.shape[2] == 4:  # Has alpha channel
        # OpenCV uses BGRA, but we need RGBA for proper PNG saving
        canvas_rgba = cv2.cvtColor(canvas, cv2.COLOR_BGRA2RGBA)
        success = cv2.imwrite(filepath, canvas_rgba)
    else:
        # If no alpha channel, just save as is
        success = cv2.imwrite(filepath, canvas)
    
    if success:
        print(f"Saved PNG file: {filepath}")
        return filepath
    else:
        print(f"Failed to save PNG file: {filepath}")
        return None

def save_as_jpg(canvas, directory="drawings", filename=None, quality=95):
    """Save the canvas as a JPG file (no transparency)."""
    if filename is None:
        filename = generate_filename("drawing", "jpg")
    
    # Ensure directory exists
    directory = ensure_save_directory(directory)
    filepath = os.path.join(directory, filename)
    
    # For JPG, we need to remove alpha channel and use white background
    if canvas.shape[2] == 4:  # Has alpha channel
        # Extract alpha channel
        alpha = canvas[:, :, 3] / 255.0
        
        # Create white background
        white_bg = np.ones_like(canvas[:, :, :3]) * 255
        
        # Blend with white background
        bgr = canvas[:, :, :3]
        canvas_rgb = bgr * alpha[:, :, np.newaxis] + white_bg * (1 - alpha[:, :, np.newaxis])
        canvas_rgb = canvas_rgb.astype(np.uint8)
    else:
        canvas_rgb = canvas
    
    # Define JPG quality
    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    
    # Save as JPG
    success = cv2.imwrite(filepath, canvas_rgb, encode_params)
    
    if success:
        print(f"Saved JPG file: {filepath}")
        return filepath
    else:
        print(f"Failed to save JPG file: {filepath}")
        return None

def create_thumbnail(canvas, size=(200, 200)):
    """Create a thumbnail version of the canvas."""
    if canvas.shape[2] == 4:  # Has alpha channel
        # Extract alpha channel
        alpha = canvas[:, :, 3] / 255.0
        
        # Create white background
        white_bg = np.ones_like(canvas[:, :, :3]) * 255
        
        # Blend with white background
        bgr = canvas[:, :, :3]
        canvas_rgb = bgr * alpha[:, :, np.newaxis] + white_bg * (1 - alpha[:, :, np.newaxis])
        canvas_rgb = canvas_rgb.astype(np.uint8)
    else:
        canvas_rgb = canvas
    
    # Resize to thumbnail size
    thumbnail = cv2.resize(canvas_rgb, size, interpolation=cv2.INTER_AREA)
    
    return thumbnail

def apply_image_filter(canvas, filter_type="sharpen"):
    """Apply image filters to the canvas."""
    # Make a copy to avoid modifying the original
    result = canvas.copy()
    
    # Extract alpha channel if present
    has_alpha = (canvas.shape[2] == 4)
    if has_alpha:
        alpha = canvas[:, :, 3].copy()
        rgb = canvas[:, :, :3].copy()
    else:
        rgb = canvas.copy()
    
    if filter_type == "sharpen":
        # Apply sharpening kernel
        kernel = np.array([[-1, -1, -1], 
                           [-1,  9, -1], 
                           [-1, -1, -1]])
        rgb = cv2.filter2D(rgb, -1, kernel)
    
    elif filter_type == "blur":
        # Apply Gaussian blur
        rgb = cv2.GaussianBlur(rgb, (5, 5), 0)
    
    elif filter_type == "emboss":
        # Apply emboss effect
        kernel = np.array([[-2, -1, 0], 
                           [-1,  1, 1], 
                           [ 0,  1, 2]])
        rgb = cv2.filter2D(rgb, -1, kernel)
    
    elif filter_type == "edge":
        # Edge detection
        rgb = cv2.Canny(rgb, 100, 200)
        # Convert back to 3 channels
        rgb = cv2.cvtColor(rgb, cv2.COLOR_GRAY2BGR)
    
    elif filter_type == "sepia":
        # Apply sepia tone
        sepia_kernel = np.array([[0.272, 0.534, 0.131],
                                 [0.349, 0.686, 0.168],
                                 [0.393, 0.769, 0.189]])
        rgb = cv2.transform(rgb, sepia_kernel)
    
    # Recombine with alpha channel if needed
    if has_alpha:
        result = np.zeros_like(canvas)
        result[:, :, :3] = rgb
        result[:, :, 3] = alpha
        return result
    else:
        return rgb

def setup_autosave(canvas, save_interval=5, directory="drawings"):
    """Set up an autosave function that runs every save_interval minutes."""
    def autosave_function():
        last_save_time = time.time()
        
        while True:
            current_time = time.time()
            elapsed_minutes = (current_time - last_save_time) / 60
            
            if elapsed_minutes >= save_interval:
                # Generate autosave filename
                filename = f"autosave_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                save_as_png(canvas, directory, filename)
                last_save_time = current_time
            
            # Check every 10 seconds
            time.sleep(10)
    
    # Return the function for the app to call in a separate thread
    return autosave_function
