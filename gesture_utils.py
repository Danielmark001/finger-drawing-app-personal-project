import numpy as np
import math
import cv2
import time

def calculate_distance(point1, point2):
    """Calculate Euclidean distance between two points."""
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def get_finger_position(hand_landmarks, finger_idx, image_shape):
    """Get the x, y coordinates of a finger landmark in pixel space."""
    h, w, _ = image_shape
    x = int(hand_landmarks.landmark[finger_idx].x * w)
    y = int(hand_landmarks.landmark[finger_idx].y * h)
    return (x, y)

def detect_pinch_gesture(hand_landmarks, mp_hands, image_shape, threshold=0.05):
    """Detect if thumb and index finger are pinched together."""
    thumb_tip = get_finger_position(hand_landmarks, mp_hands.HandLandmark.THUMB_TIP, image_shape)
    index_tip = get_finger_position(hand_landmarks, mp_hands.HandLandmark.INDEX_FINGER_TIP, image_shape)
    
    # Calculate distance between thumb and index finger
    distance = calculate_distance(thumb_tip, index_tip)
    
    # Get diagonal of the frame for normalization
    h, w, _ = image_shape
    diagonal = math.sqrt(h**2 + w**2)
    
    # Normalize distance relative to the frame
    normalized_distance = distance / diagonal
    
    return normalized_distance < threshold, normalized_distance, thumb_tip

def create_color_picker(width, height):
    """Create a color picker image with a color gradient."""
    # Create a color wheel or HSV gradient
    color_picker = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Create HSV gradient
    for y in range(height):
        for x in range(width):
            # Map x to H (0-180 for OpenCV)
            h = int(x * 180 / width)
            # Map y inversely to S (0-255)
            s = 255
            # V is always 255 (full brightness)
            v = 255
            
            # Set HSV value
            color_picker[y, x] = (h, s, v)
    
    # Convert HSV to BGR for display
    color_picker = cv2.cvtColor(color_picker, cv2.COLOR_HSV2BGR)
    
    return color_picker

def create_advanced_color_picker(width, height):
    """Create an advanced color picker with both hue and saturation/value controls."""
    # Create a main HSV gradient for hue
    hue_picker = np.zeros((height // 2, width, 3), dtype=np.uint8)
    
    # Create saturation-value picker
    sv_picker = np.zeros((height // 2, width, 3), dtype=np.uint8)
    
    # Create HSV gradient for hue picker
    for y in range(height // 2):
        for x in range(width):
            # Map x to H (0-180 for OpenCV)
            h = int(x * 180 / width)
            # Fixed saturation and value for hue picker
            s = 255
            v = 255
            
            # Set HSV value
            hue_picker[y, x] = (h, s, v)
    
    # Create saturation-value grid for the selected hue
    # Default hue is red (0)
    selected_hue = 0
    
    for y in range(height // 2):
        for x in range(width):
            # Map x to saturation (0-255)
            s = int(x * 255 / width)
            # Map y inversely to value (0-255)
            v = int((1.0 - y / (height // 2)) * 255)
            
            # Set HSV value
            sv_picker[y, x] = (selected_hue, s, v)
    
    # Convert HSV to BGR for display
    hue_picker = cv2.cvtColor(hue_picker, cv2.COLOR_HSV2BGR)
    sv_picker = cv2.cvtColor(sv_picker, cv2.COLOR_HSV2BGR)
    
    # Combine the two pickers
    color_picker = np.vstack((hue_picker, sv_picker))
    
    return color_picker

def get_color_from_position(color_picker, position):
    """Get color from a position on the color picker."""
    x, y = position
    h, w, _ = color_picker.shape
    
    # Ensure position is within bounds
    x = max(0, min(x, w-1))
    y = max(0, min(y, h-1))
    
    # Get color at position (BGR format)
    color = color_picker[y, x]
    
    # Convert from numpy array to tuple
    return (int(color[0]), int(color[1]), int(color[2]))

def draw_brush_preview(frame, position, color, thickness):
    """Draw a preview of the brush at the given position with improved visual."""
    # Draw a shadow
    cv2.circle(frame, (position[0]+2, position[1]+2), thickness, (20, 20, 20), 1)
    
    # Draw the main circle
    cv2.circle(frame, position, thickness, color, 2)
    
    # Draw a small dot in the center
    cv2.circle(frame, position, 2, (255, 255, 255), -1)
    
    return frame

def draw_advanced_brush_preview(frame, position, color, thickness, brush_style):
    """Draw an advanced preview of the brush based on style."""
    if brush_style == "solid":
        # Simple circle with border
        cv2.circle(frame, (position[0]+2, position[1]+2), thickness, (20, 20, 20), 1)
        cv2.circle(frame, position, thickness, color, 2)
        cv2.circle(frame, position, 2, (255, 255, 255), -1)
    
    elif brush_style == "airbrush":
        # Spray pattern preview
        cv2.circle(frame, position, thickness, color, 1)
        
        # Draw some random dots to show spray pattern
        for _ in range(thickness * 2):
            offset_x = np.random.randint(-thickness, thickness+1)
            offset_y = np.random.randint(-thickness, thickness+1)
            
            # Distance from center determines opacity
            dist = math.sqrt(offset_x**2 + offset_y**2)
            if dist < thickness:
                opacity = int(255 * (1 - dist / thickness))
                dot_size = max(1, thickness // 10)
                cv2.circle(frame, 
                          (position[0] + offset_x, position[1] + offset_y), 
                          dot_size, (*color, opacity), -1)
    
    elif brush_style == "marker":
        # Semi-transparent rectangle to show marker effect
        marker_width = thickness
        marker_height = thickness * 2
        
        # Create a transparent overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, 
                     (position[0] - marker_width//2, position[1] - marker_height//2),
                     (position[0] + marker_width//2, position[1] + marker_height//2),
                     color, -1)
        
        # Blend with reduced opacity
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
    
    elif brush_style == "pencil":
        # Thin lines with texture for pencil preview
        for i in range(5):
            offset_x = np.random.randint(-1, 2)
            offset_y = np.random.randint(-1, 2)
            cv2.circle(frame, 
                      (position[0] + offset_x, position[1] + offset_y), 
                      max(1, thickness // 4), color, -1)
        
        # Draw center point
        cv2.circle(frame, position, 1, (255, 255, 255), -1)
    
    return frame
    
def draw_gesture_guide(frame, hand_landmarks, mp_hands, mode, image_shape):
    """Draw visual guides for different gestures."""
    h, w, _ = image_shape
    
    if mode == "draw":
        # Highlight the index finger
        index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
        index_x, index_y = int(index_tip.x * w), int(index_tip.y * h)
        
        # Draw a small circle around the index fingertip
        cv2.circle(frame, (index_x, index_y), 15, (0, 255, 0), 2)
        
    elif mode == "erase":
        # Highlight the palm area
        wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
        middle_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
        
        wrist_x, wrist_y = int(wrist.x * w), int(wrist.y * h)
        mcp_x, mcp_y = int(middle_mcp.x * w), int(middle_mcp.y * h)
        
        # Draw a circle around the palm center
        palm_center_x = (wrist_x + mcp_x) // 2
        palm_center_y = (wrist_y + mcp_y) // 2
        
        palm_radius = int(calculate_distance((wrist_x, wrist_y), (mcp_x, mcp_y)))
        cv2.circle(frame, (palm_center_x, palm_center_y), palm_radius, (0, 0, 255), 2)
        
    elif mode == "color_pick":
        # Highlight index and middle fingers
        index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
        middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
        
        index_x, index_y = int(index_tip.x * w), int(index_tip.y * h)
        middle_x, middle_y = int(middle_tip.x * w), int(middle_tip.y * h)
        
        # Draw lines connecting the fingertips
        cv2.line(frame, (index_x, index_y), (middle_x, middle_y), (255, 255, 0), 2)
        
        # Draw circles around the fingertips
        cv2.circle(frame, (index_x, index_y), 10, (255, 255, 0), 2)
        cv2.circle(frame, (middle_x, middle_y), 10, (255, 255, 0), 2)
    
    elif mode == "shape":
        # Highlight index finger and draw shape guide
        index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
        index_x, index_y = int(index_tip.x * w), int(index_tip.y * h)
        
        # Draw a crosshair at the finger position
        cv2.line(frame, (index_x - 15, index_y), (index_x + 15, index_y), (0, 255, 255), 2)
        cv2.line(frame, (index_x, index_y - 15), (index_x, index_y + 15), (0, 255, 255), 2)
        cv2.circle(frame, (index_x, index_y), 20, (0, 255, 255), 1)
    
    elif mode == "select":
        # Highlight three fingers (tool selection mode)
        index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
        middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
        ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
        
        index_x, index_y = int(index_tip.x * w), int(index_tip.y * h)
        middle_x, middle_y = int(middle_tip.x * w), int(middle_tip.y * h)
        ring_x, ring_y = int(ring_tip.x * w), int(ring_tip.y * h)
        
        # Connect the finger tips
        cv2.line(frame, (index_x, index_y), (middle_x, middle_y), (255, 0, 255), 2)
        cv2.line(frame, (middle_x, middle_y), (ring_x, ring_y), (255, 0, 255), 2)
        
        # Draw circles at fingertips
        cv2.circle(frame, (index_x, index_y), 8, (255, 0, 255), -1)
        cv2.circle(frame, (middle_x, middle_y), 8, (255, 0, 255), -1)
        cv2.circle(frame, (ring_x, ring_y), 8, (255, 0, 255), -1)
    
    return frame

def create_startup_screen(width, height):
    """Create a welcome screen for the application."""
    # Create a dark background
    startup_screen = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Create a gradient background
    for y in range(height):
        for x in range(width):
            startup_screen[y, x] = [
                int(30 + (x * 20 / width)),
                int(30 + ((x + y) * 20 / (width + height))),
                int(40 + (y * 30 / height))
            ]
    
    # Add a title
    title = "Finger Drawing App"
    font_scale = 1.5
    font_thickness = 3
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # Get text size
    text_size = cv2.getTextSize(title, font, font_scale, font_thickness)[0]
    text_x = (width - text_size[0]) // 2
    text_y = height // 3
    
    # Draw title text with shadow effect
    cv2.putText(startup_screen, title, (text_x + 2, text_y + 2), font, font_scale, (0, 0, 0), font_thickness)
    cv2.putText(startup_screen, title, (text_x, text_y), font, font_scale, (255, 255, 255), font_thickness)
    
    # Add subtitle
    subtitle = "Draw with your finger, erase with your palm"
    sub_font_scale = 0.8
    sub_text_size = cv2.getTextSize(subtitle, font, sub_font_scale, 2)[0]
    sub_x = (width - sub_text_size[0]) // 2
    sub_y = text_y + 50
    
    cv2.putText(startup_screen, subtitle, (sub_x, sub_y), font, sub_font_scale, (200, 200, 255), 2)
    
    # Add hand illustration
    hand_center_x = width // 2
    hand_center_y = height // 2 + 50
    hand_radius = 50
    
    # Draw palm
    cv2.circle(startup_screen, (hand_center_x, hand_center_y), hand_radius, (100, 100, 150), -1)
    cv2.circle(startup_screen, (hand_center_x, hand_center_y), hand_radius, (150, 150, 200), 2)
    
    # Draw fingers
    finger_length = 80
    finger_width = 15
    
    for i in range(5):
        angle = np.pi * 0.7 - (i * np.pi * 0.2)
        end_x = int(hand_center_x + finger_length * np.cos(angle))
        end_y = int(hand_center_y - finger_length * np.sin(angle))
        
        # Draw finger
        cv2.line(startup_screen, (hand_center_x, hand_center_y), (end_x, end_y), (100, 100, 150), finger_width)
        cv2.circle(startup_screen, (end_x, end_y), finger_width // 2, (100, 100, 150), -1)
        
        # Draw outline
        cv2.line(startup_screen, (hand_center_x, hand_center_y), (end_x, end_y), (150, 150, 200), 2)
        cv2.circle(startup_screen, (end_x, end_y), finger_width // 2, (150, 150, 200), 2)
    
    # Add instruction to continue
    instruction = "Press any key to start"
    inst_font_scale = 0.7
    inst_text_size = cv2.getTextSize(instruction, font, inst_font_scale, 2)[0]
    inst_x = (width - inst_text_size[0]) // 2
    inst_y = height - 50
    
    cv2.putText(startup_screen, instruction, (inst_x, inst_y), font, inst_font_scale, (255, 255, 255), 2)
    
    return startup_screen

def calculate_brightness(color):
    """Calculate perceived brightness of a color (BGR format)."""
    # Convert BGR to perceived brightness (0-255)
    # Using the formula: 0.299*R + 0.587*G + 0.114*B
    return 0.299 * color[2] + 0.587 * color[1] + 0.114 * color[0]

def detect_hand_presence(landmarks, mp_hands, margin=0.1):
    """Detect if hand is fully in the frame with a margin."""
    # Check if all landmarks are within frame bounds with margin
    for landmark in landmarks.landmark:
        if (landmark.x < margin or landmark.x > 1 - margin or
            landmark.y < margin or landmark.y > 1 - margin):
            return False
    
    return True

def draw_animated_gesture_guide(frame, current_time):
    """Draw an animated guide showing hand gestures and their functions."""
    h, w, _ = frame.shape
    
    # Background panel
    cv2.rectangle(frame, (w//2 - 150, h//2 - 100), (w//2 + 150, h//2 + 100), (30, 30, 40), -1)
    cv2.rectangle(frame, (w//2 - 150, h//2 - 100), (w//2 + 150, h//2 + 100), (100, 100, 150), 2)
    
    # Which gesture to show (cycle through gestures)
    gesture_period = 3.0  # seconds per gesture
    gesture_index = int((current_time % (gesture_period * 5)) / gesture_period)
    
    gestures = [
        ("Index finger", "Draw", (0, 255, 0)),
        ("Open palm", "Erase", (0, 0, 255)),
        ("Peace sign", "Color picker", (255, 255, 0)),
        ("Three fingers", "Tool menu", (255, 0, 255)),
        ("Pinch", "Select", (0, 255, 255))
    ]
    
    name, action, color = gestures[gesture_index]
    
    # Draw gesture name
    cv2.putText(frame, name, (w//2 - 100, h//2 - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    
    # Draw action
    cv2.putText(frame, action, (w//2 - 100, h//2 + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # Animated progress bar
    progress = (current_time % gesture_period) / gesture_period
    bar_width = 250
    filled_width = int(bar_width * progress)
    
    cv2.rectangle(frame, (w//2 - 125, h//2 + 80), (w//2 + 125, h//2 + 90), (50, 50, 50), -1)
    cv2.rectangle(frame, (w//2 - 125, h//2 + 80), (w//2 - 125 + filled_width, h//2 + 90), color, -1)
    
    return frame

def get_max_layer_index(layers):
    """Get the index of the layer with the most non-transparent pixels."""
    max_pixels = 0
    max_index = 0
    
    for i, layer in enumerate(layers):
        # Count non-transparent pixels
        non_transparent = np.sum(layer[:, :, 3] > 0)
        if non_transparent > max_pixels:
            max_pixels = non_transparent
            max_index = i
    
    return max_index

def apply_texture(image, texture_type="canvas"):
    """Apply a texture effect to the image."""
    h, w = image.shape[:2]
    
    # Extract RGB and alpha channels
    rgb = image[:, :, :3].copy()
    alpha = image[:, :, 3].copy() if image.shape[2] == 4 else None
    
    if texture_type == "canvas":
        # Create a canvas-like texture
        texture = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Generate noise
        noise = np.random.randint(0, 15, (h, w))
        
        # Create a repeating pattern
        pattern_size = 20
        for y in range(0, h, pattern_size):
            for x in range(0, w, pattern_size):
                # Alternate light and dark patches
                shade = 240 if (x // pattern_size + y // pattern_size) % 2 == 0 else 220
                
                # Set patch with some variation
                yend = min(y + pattern_size, h)
                xend = min(x + pattern_size, w)
                texture[y:yend, x:xend] = (shade, shade, shade)
        
        # Add noise
        texture = np.clip(texture + noise[:, :, np.newaxis], 0, 255).astype(np.uint8)
        
        # Blend with original image
        result = cv2.addWeighted(rgb, 0.9, texture, 0.1, 0)
    
    elif texture_type == "paper":
        # Create a paper-like texture
        texture = np.ones((h, w, 3), dtype=np.uint8) * 240
        
        # Generate noise
        noise = np.random.randint(-10, 10, (h, w, 3))
        texture = np.clip(texture + noise, 200, 255).astype(np.uint8)
        
        # Add some random lines for paper fibers
        for _ in range(100):
            x1, y1 = np.random.randint(0, w), np.random.randint(0, h)
            x2, y2 = x1 + np.random.randint(-10, 10), y1 + np.random.randint(-10, 10)
            
            if 0 <= x2 < w and 0 <= y2 < h:
                cv2.line(texture, (x1, y1), (x2, y2), (230, 230, 230), 1)
        
        # Blend with original image
        result = cv2.addWeighted(rgb, 0.85, texture, 0.15, 0)
    
    else:  # No texture
        result = rgb
    
    # Recombine with alpha channel
    if alpha is not None:
        result_with_alpha = np.zeros((h, w, 4), dtype=np.uint8)
        result_with_alpha[:, :, :3] = result
        result_with_alpha[:, :, 3] = alpha
        return result_with_alpha
    
    return result

def create_brush_preview_image(brush_style, color, thickness):
    """Create a preview image for a brush style."""
    preview_size = 100
    preview = np.zeros((preview_size, preview_size, 4), dtype=np.uint8)
    
    center = preview_size // 2
    
    if brush_style == "solid":
        # Draw a simple line
        cv2.line(preview, (20, center), (80, center), (*color, 255), thickness)
    
    elif brush_style == "airbrush":
        # Draw spray pattern
        for _ in range(100):
            x = np.random.randint(center - 30, center + 30)
            y = np.random.randint(center - 30, center + 30)
            
            # Distance from center affects opacity
            dist = math.sqrt((x - center)**2 + (y - center)**2)
            if dist < 30:
                opacity = int(255 * (1 - dist / 30))
                dot_size = max(1, thickness // 8)
                cv2.circle(preview, (x, y), dot_size, (*color, opacity), -1)
    
    elif brush_style == "marker":
        # Draw marker strokes
        for offset in range(-thickness//3, thickness//3 + 1):
            cv2.line(preview, (20, center + offset), (80, center + offset), (*color, 100), thickness//2)
    
    elif brush_style == "pencil":
        # Draw pencil strokes with texture
        for _ in range(30):
            x1 = np.random.randint(20, 40)
            y1 = np.random.randint(center - 5, center + 5)
            x2 = np.random.randint(60, 80)
            y2 = np.random.randint(center - 5, center + 5)
            
            # Vary opacity for pencil texture
            opacity = np.random.randint(150, 256)
            cv2.line(preview, (x1, y1), (x2, y2), (*color, opacity), max(1, thickness // 4))
    
    return preview
