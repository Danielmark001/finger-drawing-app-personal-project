import numpy as np
import math
import cv2

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
