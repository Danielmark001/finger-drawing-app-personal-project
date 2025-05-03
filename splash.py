import cv2
import numpy as np
import time
import os
import threading

def create_splash_screen(width=800, height=600, duration=3):
    """Display a splash screen for the specified duration in seconds."""
    # Create a gradient background
    splash = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Create a blue to purple gradient
    for y in range(height):
        for x in range(width):
            # Gradient calculation
            blue = int(40 + (x * 80 / width))
            green = int(10 + (y * 40 / height))
            red = int(60 + ((x + y) * 60 / (width + height)))
            splash[y, x] = [blue, green, red]
    
    # Add app title
    title = "Finger Drawing App"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.8
    thickness = 3
    
    # Calculate text position to center it
    text_size = cv2.getTextSize(title, font, font_scale, thickness)[0]
    text_x = (width - text_size[0]) // 2
    text_y = height // 3
    
    # Add shadow and text
    cv2.putText(splash, title, (text_x + 3, text_y + 3), font, font_scale, (0, 0, 0), thickness)
    cv2.putText(splash, title, (text_x, text_y), font, font_scale, (255, 255, 255), thickness)
    
    # Add tagline
    tagline = "Draw with your hands, create with gestures"
    tagline_scale = 0.8
    tagline_size = cv2.getTextSize(tagline, font, tagline_scale, 2)[0]
    tagline_x = (width - tagline_size[0]) // 2
    tagline_y = text_y + 60
    
    cv2.putText(splash, tagline, (tagline_x, tagline_y), font, tagline_scale, (220, 220, 255), 2)
    
    # Add version
    version = "v1.4.2"
    cv2.putText(splash, version, (width - 100, height - 30), font, 0.6, (180, 180, 220), 1)
    
    # Add loading animation
    loading_bar_width = 400
    loading_bar_height = 8
    loading_bar_x = (width - loading_bar_width) // 2
    loading_bar_y = height - 80
    
    # Create window
    window_name = "Finger Drawing App"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
    
    # Show animated loading bar
    start_time = time.time()
    while time.time() - start_time < duration:
        # Create a copy of the splash screen
        frame = splash.copy()
        
        # Calculate progress (0 to 1)
        progress = min(1.0, (time.time() - start_time) / duration)
        
        # Draw background of loading bar
        cv2.rectangle(frame, 
                     (loading_bar_x, loading_bar_y), 
                     (loading_bar_x + loading_bar_width, loading_bar_y + loading_bar_height), 
                     (60, 60, 80), -1)
        
        # Draw fill part of loading bar
        fill_width = int(loading_bar_width * progress)
        cv2.rectangle(frame, 
                     (loading_bar_x, loading_bar_y), 
                     (loading_bar_x + fill_width, loading_bar_y + loading_bar_height), 
                     (120, 120, 255), -1)
        
        # Add "Loading..." text
        loading_text = "Loading..."
        cv2.putText(frame, loading_text, (loading_bar_x, loading_bar_y - 10), 
                   font, 0.6, (200, 200, 200), 1)
        
        # Show the splash screen
        cv2.imshow(window_name, frame)
        
        # Break if ESC pressed
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
    
    # Clean up
    cv2.destroyWindow(window_name)

def check_first_run():
    """Check if this is the first time running the app."""
    flag_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".app_initialized")
    
    if not os.path.exists(flag_file):
        # This is the first run
        # Create the flag file
        with open(flag_file, "w") as f:
            f.write("App initialized")
        return True
    
    return False

def show_tips(width=800, height=600):
    """Show tips for first-time users."""
    # Create a dark background
    tips_bg = np.zeros((height, width, 3), dtype=np.uint8)
    tips_bg[:, :] = (30, 30, 40)  # Dark blue-gray background
    
    # Add title
    title = "Quick Start Guide"
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(tips_bg, title, (width//2 - 150, 60), font, 1.2, (255, 255, 255), 2)
    
    # Add divider line
    cv2.line(tips_bg, (50, 80), (width-50, 80), (100, 100, 150), 2)
    
    # Define tips
    tips = [
        "1. Use your index finger to draw on the screen",
        "2. Show your open palm to erase content",
        "3. Make a peace sign (index + middle finger) to open the color picker",
        "4. Use pinch gesture (thumb + index finger) to select a color",
        "5. Extend three fingers (index, middle, ring) to open the toolbar",
        "6. Make a fist gesture to undo your last action",
        "",
        "Keyboard shortcuts:",
        "  • Press 'r', 'g', 'b', 'w', or 'k' to change colors",
        "  • Press '+' or '-' to adjust brush size",
        "  • Press 'c' to clear the canvas",
        "  • Press 's' to save your drawing",
        "  • Press 'h' anytime to show the help screen"
    ]
    
    # Add tips text
    y_offset = 120
    line_height = 30
    
    for tip in tips:
        cv2.putText(tips_bg, tip, (60, y_offset), font, 0.6, (200, 200, 220), 1)
        y_offset += line_height
    
    # Add note at bottom
    note = "Press any key to continue to the app..."
    cv2.putText(tips_bg, note, (width//2 - 180, height - 40), font, 0.7, (150, 150, 255), 1)
    
    # Show tips window
    window_name = "Quick Start Guide"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, tips_bg)
    
    # Wait for key press
    cv2.waitKey(0)
    cv2.destroyWindow(window_name)

def run_splash_screen():
    """Run the splash screen and any first-run screens."""
    try:
        # Run splash screen
        create_splash_screen()
        
        # Check if this is the first run
        if check_first_run():
            # Show tips for first-time users
            show_tips()
            
    except Exception as e:
        print(f"Error in splash screen: {e}")
        # If there's an error, wait a bit to show the error message
        time.sleep(1)

if __name__ == "__main__":
    # Test the splash screen
    run_splash_screen()
