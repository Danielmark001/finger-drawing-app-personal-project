import cv2
import numpy as np
import mediapipe as mp
import time
import math
import os
import pickle
from collections import deque
from gesture_utils import (
    detect_pinch_gesture, 
    create_color_picker, 
    get_color_from_position,
    draw_brush_preview,
    calculate_distance,
    get_finger_position
)

# Import config if available
try:
    from config import *
except ImportError:
    # Default settings if config.py is not found
    FULLSCREEN = False
    WINDOW_TITLE = "Finger Drawing App"

class FingerDrawingApp:
    def __init__(self):
        # Initialize mediapipe hands module
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Drawing parameters
        self.canvas = None
        self.layers = []
        self.active_layer = 0
        self.prev_finger_pos = None
        self.drawing_color = (0, 0, 255)  # Red in BGR format
        self.brush_thickness = 5
        self.is_drawing = False
        self.is_erasing = False
        
        # Window state
        self.fullscreen = FULLSCREEN if 'FULLSCREEN' in globals() else False
        self.window_title = WINDOW_TITLE if 'WINDOW_TITLE' in globals() else "Finger Drawing App"
        
        # Undo/redo history
        self.undo_stack = deque(maxlen=20)
        self.redo_stack = deque(maxlen=20)
        
        # UI elements
        self.show_color_picker = False
        self.color_picker = None
        self.color_palette = []
        self.saved_colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 255), (0, 0, 0)]
        self.color_picker_pos = (20, 100)
        self.color_picker_size = (200, 30)
        
        # Tools
        self.tools = ["brush", "eraser", "line", "rectangle", "circle", "fill"]
        self.current_tool = "brush"
        self.shape_start = None
        
        # Effects
        self.effects = ["none", "blur", "sharpen", "grayscale", "invert"]
        self.current_effect = "none"
        
        # Brush styles
        self.brush_styles = ["solid", "airbrush", "marker", "pencil"]
        self.current_brush = "solid"
        
        # Mode control
        self.current_mode = "draw"
        
        # Help and UI state
        self.show_help = False
        self.show_toolbar = True
        self.show_layer_panel = False
        
        # Initialize webcam
        self.cap = cv2.VideoCapture(0)
        
        # Try multiple times to get a frame
        max_attempts = 5
        for attempt in range(max_attempts):
            success, frame = self.cap.read()
            if success and frame is not None and frame.size > 0:
                print(f"Got webcam on attempt {attempt+1}")
                h, w, _ = frame.shape
                
                # Initialize first layer
                new_layer = np.zeros((h, w, 4), dtype=np.uint8)
                self.layers.append(new_layer)
                
                # Set canvas as combination of all layers
                self.canvas = np.zeros((h, w, 4), dtype=np.uint8)
                
                self.color_picker = create_color_picker(self.color_picker_size[0], self.color_picker_size[1])
                
                # Try to load saved colors
                self.load_saved_colors()
                break
            elif attempt < max_attempts - 1:
                print(f"Attempt {attempt+1} failed, trying again...")
                time.sleep(0.5)
                self.cap.release()
                self.cap = cv2.VideoCapture(0)
            else:
                print(f"Couldn't get webcam after {max_attempts} tries")
                # Default sizes as fallback
                self.canvas = np.zeros((480, 640, 4), dtype=np.uint8)
                new_layer = np.zeros((480, 640, 4), dtype=np.uint8)
                self.layers.append(new_layer)
                self.color_picker = create_color_picker(self.color_picker_size[0], self.color_picker_size[1])
    
    def save_current_state(self):
        """Save current state for undo/redo."""
        layers_copy = [layer.copy() for layer in self.layers]
        self.undo_stack.append(layers_copy)
        self.redo_stack.clear()  # Clear redo when we do a new action
    
    def undo(self):
        """Undo the last drawing action."""
        if len(self.undo_stack) > 0:
            # Save current state to redo stack
            current_layers = [layer.copy() for layer in self.layers]
            self.redo_stack.append(current_layers)
            
            # Restore previous state
            self.layers = self.undo_stack.pop()
            self.update_canvas()
            return True
        return False
    
    def redo(self):
        """Redo the last undone action."""
        if len(self.redo_stack) > 0:
            # Save current state to undo stack
            current_layers = [layer.copy() for layer in self.layers]
            self.undo_stack.append(current_layers)
            
            # Restore redo state
            self.layers = self.redo_stack.pop()
            self.update_canvas()
            return True
        return False
    
    def add_layer(self):
        """Add a new transparent layer."""
        if self.canvas is not None:
            h, w = self.canvas.shape[:2]
            new_layer = np.zeros((h, w, 4), dtype=np.uint8)
            self.layers.append(new_layer)
            self.active_layer = len(self.layers) - 1
            self.update_canvas()
    
    def delete_layer(self):
        """Delete the current active layer."""
        if len(self.layers) > 1:  # Keep at least one layer
            self.layers.pop(self.active_layer)
            self.active_layer = min(self.active_layer, len(self.layers) - 1)
            self.update_canvas()
    
    def move_layer_up(self):
        """Move the active layer up in the stack."""
        if self.active_layer < len(self.layers) - 1:
            self.layers[self.active_layer], self.layers[self.active_layer + 1] = \
                self.layers[self.active_layer + 1], self.layers[self.active_layer]
            self.active_layer += 1
            self.update_canvas()
    
    def move_layer_down(self):
        """Move the active layer down in the stack."""
        if self.active_layer > 0:
            self.layers[self.active_layer], self.layers[self.active_layer - 1] = \
                self.layers[self.active_layer - 1], self.layers[self.active_layer]
            self.active_layer -= 1
            self.update_canvas()
    
    def update_canvas(self):
        """Update the canvas by combining all layers."""
        if len(self.layers) == 0:
            return
            
        # Start with a transparent canvas
        h, w = self.layers[0].shape[:2]
        self.canvas = np.zeros((h, w, 4), dtype=np.uint8)
        
        # Combine all layers (bottom to top)
        for i, layer in enumerate(self.layers):
            # Skip empty layers
            if np.sum(layer[:, :, 3]) == 0:
                continue
                
            # Get the alpha channel of the current layer
            alpha = layer[:, :, 3] / 255.0
            
            # Expand alpha to 3 channels for RGB
            alpha_rgb = np.stack([alpha, alpha, alpha], axis=2)
            
            # Get RGB channels
            rgb_layer = layer[:, :, :3]
            rgb_canvas = self.canvas[:, :, :3]
            
            # Blend RGB channels
            new_rgb = rgb_canvas * (1 - alpha_rgb) + rgb_layer * alpha_rgb
            
            # Get the alpha channel of the canvas
            canvas_alpha = self.canvas[:, :, 3] / 255.0
            
            # Calculate new alpha
            new_alpha = canvas_alpha + alpha * (1 - canvas_alpha)
            
            # Update the canvas
            self.canvas[:, :, :3] = new_rgb
            self.canvas[:, :, 3] = new_alpha * 255
    
    def load_saved_colors(self):
        """Load saved color palette from file."""
        try:
            palette_file = "color_palette.pkl"
            if os.path.exists(palette_file):
                with open(palette_file, 'rb') as f:
                    self.saved_colors = pickle.load(f)
                print("Loaded color palette")
        except Exception as e:
            print(f"Error loading palette: {e}")
    
    def save_color_palette(self):
        """Save the current color palette to file."""
        try:
            palette_file = "color_palette.pkl"
            with open(palette_file, 'wb') as f:
                pickle.dump(self.saved_colors, f)
            print("Saved color palette")
        except Exception as e:
            print(f"Error saving palette: {e}")
    
    def add_to_palette(self, color):
        """Add current color to saved palette."""
        if color not in self.saved_colors:
            self.saved_colors.append(color)
            if len(self.saved_colors) > 10:  # Limit to 10 saved colors
                self.saved_colors.pop(0)
            self.save_color_palette()
    
    def detect_hand_landmarks(self, frame):
        """Detect hand landmarks using MediaPipe."""
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame with mediapipe
        results = self.hands.process(rgb_frame)
        
        return results
    
    def is_finger_extended(self, landmarks, finger_tip_idx, finger_pip_idx):
        """Check if a finger is extended (pointing up)."""
        tip = landmarks.landmark[finger_tip_idx]
        pip = landmarks.landmark[finger_pip_idx]
        
        # Check if tip is higher than pip (y is smaller)
        return tip.y < pip.y
    
    def is_palm_showing(self, landmarks):
        """Check if palm is showing (for eraser)."""
        # 1. Check if fingers are extended
        fingers_extended = [
            self.is_finger_extended(landmarks, self.mp_hands.HandLandmark.THUMB_TIP, self.mp_hands.HandLandmark.THUMB_IP),
            self.is_finger_extended(landmarks, self.mp_hands.HandLandmark.INDEX_FINGER_TIP, self.mp_hands.HandLandmark.INDEX_FINGER_PIP),
            self.is_finger_extended(landmarks, self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP, self.mp_hands.HandLandmark.MIDDLE_FINGER_PIP),
            self.is_finger_extended(landmarks, self.mp_hands.HandLandmark.RING_FINGER_TIP, self.mp_hands.HandLandmark.RING_FINGER_PIP),
            self.is_finger_extended(landmarks, self.mp_hands.HandLandmark.PINKY_TIP, self.mp_hands.HandLandmark.PINKY_PIP)
        ]
        
        # Count extended fingers (excluding thumb which can be less reliable)
        extended_count = sum(fingers_extended[1:])
        
        # 2. Check palm orientation (palm facing the camera)
        wrist = landmarks.landmark[self.mp_hands.HandLandmark.WRIST]
        middle_mcp = landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
        middle_tip = landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
        
        # If palm is facing the camera, the middle_tip's z should be closer to camera than the middle_mcp's z
        palm_facing_camera = (middle_tip.z - middle_mcp.z) < -0.05
        
        # 3. Check finger spread (fingers should be apart, not clenched)
        index_tip = landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
        pinky_tip = landmarks.landmark[self.mp_hands.HandLandmark.PINKY_TIP]
        
        # Calculate horizontal distance between index and pinky fingertips
        finger_spread = abs(index_tip.x - pinky_tip.x)
        fingers_spread_apart = finger_spread > 0.2  # Threshold for spread fingers
        
        # Combine all conditions:
        # - At least 3 fingers must be extended (excluding thumb)
        # - Palm should be facing camera
        # - Fingers should be spread apart
        return extended_count >= 3 and palm_facing_camera and fingers_spread_apart
    
    def detect_fist_gesture(self, landmarks):
        """Detect if hand is making a fist."""
        fingers_extended = [
            self.is_finger_extended(landmarks, self.mp_hands.HandLandmark.THUMB_TIP, self.mp_hands.HandLandmark.THUMB_IP),
            self.is_finger_extended(landmarks, self.mp_hands.HandLandmark.INDEX_FINGER_TIP, self.mp_hands.HandLandmark.INDEX_FINGER_PIP),
            self.is_finger_extended(landmarks, self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP, self.mp_hands.HandLandmark.MIDDLE_FINGER_PIP),
            self.is_finger_extended(landmarks, self.mp_hands.HandLandmark.RING_FINGER_TIP, self.mp_hands.HandLandmark.RING_FINGER_PIP),
            self.is_finger_extended(landmarks, self.mp_hands.HandLandmark.PINKY_TIP, self.mp_hands.HandLandmark.PINKY_PIP)
        ]
        
        # Count extended fingers (should be very few for a fist)
        extended_count = sum(fingers_extended)
        return extended_count <= 1  # Fist if at most one finger is extended
    
    def detect_three_finger_gesture(self, landmarks):
        """Detect if index, middle, and ring fingers are extended."""
        fingers_extended = [
            self.is_finger_extended(landmarks, self.mp_hands.HandLandmark.INDEX_FINGER_TIP, self.mp_hands.HandLandmark.INDEX_FINGER_PIP),
            self.is_finger_extended(landmarks, self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP, self.mp_hands.HandLandmark.MIDDLE_FINGER_PIP),
            self.is_finger_extended(landmarks, self.mp_hands.HandLandmark.RING_FINGER_TIP, self.mp_hands.HandLandmark.RING_FINGER_PIP)
        ]
        
        return all(fingers_extended) and not self.is_finger_extended(
            landmarks, self.mp_hands.HandLandmark.PINKY_TIP, self.mp_hands.HandLandmark.PINKY_PIP)
    
    def detect_ok_gesture(self, landmarks, image_shape):
        """Detect if making an OK sign (thumb and index finger forming a circle)."""
        thumb_tip = get_finger_position(landmarks, self.mp_hands.HandLandmark.THUMB_TIP, image_shape)
        index_tip = get_finger_position(landmarks, self.mp_hands.HandLandmark.INDEX_FINGER_TIP, image_shape)
        
        # Check if thumb and index are close together
        distance = calculate_distance(thumb_tip, index_tip)
        
        # Get diagonal of the frame for normalization
        h, w, _ = image_shape
        diagonal = math.sqrt(h**2 + w**2)
        
        # Normalize distance relative to the frame
        normalized_distance = distance / diagonal
        
        # Check if other fingers are extended
        other_fingers_extended = [
            self.is_finger_extended(landmarks, self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP, 
                                 self.mp_hands.HandLandmark.MIDDLE_FINGER_PIP),
            self.is_finger_extended(landmarks, self.mp_hands.HandLandmark.RING_FINGER_TIP, 
                                 self.mp_hands.HandLandmark.RING_FINGER_PIP),
            self.is_finger_extended(landmarks, self.mp_hands.HandLandmark.PINKY_TIP, 
                                 self.mp_hands.HandLandmark.PINKY_PIP)
        ]
        
        # OK gesture: thumb and index form circle, other fingers extended
        return normalized_distance < 0.03 and sum(other_fingers_extended) >= 2
    
    def detect_tool_selection_gesture(self, landmarks):
        """Detect tool selection gesture (index, middle, and ring extended)."""
        return (self.is_finger_extended(landmarks, self.mp_hands.HandLandmark.INDEX_FINGER_TIP, 
                                     self.mp_hands.HandLandmark.INDEX_FINGER_PIP) and
                self.is_finger_extended(landmarks, self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP, 
                                     self.mp_hands.HandLandmark.MIDDLE_FINGER_PIP) and
                self.is_finger_extended(landmarks, self.mp_hands.HandLandmark.RING_FINGER_TIP, 
                                     self.mp_hands.HandLandmark.RING_FINGER_PIP) and
                not self.is_finger_extended(landmarks, self.mp_hands.HandLandmark.PINKY_TIP, 
                                         self.mp_hands.HandLandmark.PINKY_PIP))
    
    def calculate_palm_width(self, landmarks):
        """Calculate palm width to determine distance from camera."""
        h, w, _ = self.canvas.shape
        
        # Use the distance between the pinky_mcp and thumb_mcp as palm width
        pinky_mcp = landmarks.landmark[self.mp_hands.HandLandmark.PINKY_MCP]
        thumb_mcp = landmarks.landmark[self.mp_hands.HandLandmark.THUMB_MCP]
        
        pinky_x, pinky_y = int(pinky_mcp.x * w), int(pinky_mcp.y * h)
        thumb_x, thumb_y = int(thumb_mcp.x * w), int(thumb_mcp.y * h)
        
        # Calculate Euclidean distance
        palm_width = math.sqrt((pinky_x - thumb_x)**2 + (pinky_y - thumb_y)**2)
        
        return palm_width
    
    def apply_brush_style(self, canvas, start_pos, end_pos, color, thickness):
        """Apply different brush styles."""
        if self.current_brush == "solid":
            # Simple solid line
            cv2.line(canvas, start_pos, end_pos, (*color, 255), thickness)
        
        elif self.current_brush == "airbrush":
            # Airbrush effect - spray pattern
            dx = end_pos[0] - start_pos[0]
            dy = end_pos[1] - start_pos[1]
            distance = max(1, math.sqrt(dx*dx + dy*dy))
            
            # Draw multiple small circles along the line
            steps = int(distance)
            for i in range(steps + 1):
                t = i / steps if steps > 0 else 0
                x = int(start_pos[0] + dx * t)
                y = int(start_pos[1] + dy * t)
                
                # Draw main dot
                cv2.circle(canvas, (x, y), thickness // 2, (*color, 255), -1)
                
                # Draw random spray dots around the main line
                num_spray_dots = thickness * 2
                for _ in range(num_spray_dots):
                    spray_x = x + np.random.randint(-thickness*2, thickness*2+1)
                    spray_y = y + np.random.randint(-thickness*2, thickness*2+1)
                    
                    # Calculate distance from center
                    spray_dist = math.sqrt((spray_x - x)**2 + (spray_y - y)**2)
                    
                    # Spray opacity decreases with distance
                    if spray_dist < thickness * 2:
                        opacity = int(255 * (1 - spray_dist / (thickness * 2)))
                        spray_radius = max(1, thickness // 3)
                        cv2.circle(canvas, (spray_x, spray_y), spray_radius, (*color, opacity), -1)
        
        elif self.current_brush == "marker":
            # Marker effect with semi-transparency
            alpha = 100  # Semi-transparent
            for offset in range(-thickness//3, thickness//3 + 1):
                cv2.line(canvas, 
                        (start_pos[0], start_pos[1] + offset), 
                        (end_pos[0], end_pos[1] + offset), 
                        (*color, alpha), thickness)
        
        elif self.current_brush == "pencil":
            # Pencil effect - thin line with texture
            dx = end_pos[0] - start_pos[0]
            dy = end_pos[1] - start_pos[1]
            distance = max(1, math.sqrt(dx*dx + dy*dy))
            
            # Draw multiple thin lines with slight jitter
            steps = int(distance * 2)
            for i in range(steps + 1):
                t = i / steps if steps > 0 else 0
                x = int(start_pos[0] + dx * t)
                y = int(start_pos[1] + dy * t)
                
                # Add subtle jitter for pencil texture
                jitter_x = x + np.random.randint(-1, 2)
                jitter_y = y + np.random.randint(-1, 2)
                
                # Vary opacity slightly for texture
                opacity = np.random.randint(180, 256)
                cv2.circle(canvas, (jitter_x, jitter_y), max(1, thickness // 4), (*color, opacity), -1)
    
    def fill_area(self, x, y, target_color, fill_color):
        """Fill a connected area with the given color (flood fill)."""
        h, w = self.layers[self.active_layer].shape[:2]
        
        # Create a mask for visited pixels
        visited = np.zeros((h, w), dtype=bool)
        
        # Convert target_color to include alpha
        target_color_rgba = tuple(list(target_color) + [255])
        fill_color_rgba = tuple(list(fill_color) + [255])
        
        # Skip if already the fill color
        if tuple(self.layers[self.active_layer][y, x]) == fill_color_rgba:
            return
        
        # Get target color from the clicked pixel
        target_color_rgba = tuple(self.layers[self.active_layer][y, x])
        
        # Define a tolerance for color matching (adjust as needed)
        tolerance = 30
        
        def color_match(c1, c2, tol):
            """Check if colors are within tolerance."""
            return (abs(int(c1[0]) - int(c2[0])) <= tol and
                    abs(int(c1[1]) - int(c2[1])) <= tol and
                    abs(int(c1[2]) - int(c2[2])) <= tol)
        
        # Use Queue-based approach for flood fill to avoid recursion limits
        from collections import deque
        queue = deque([(y, x)])
        visited[y, x] = True
        
        while queue:
            cy, cx = queue.popleft()
            
            # Fill the current pixel
            self.layers[self.active_layer][cy, cx] = fill_color_rgba
            
            # Check 4-connected neighbors
            for dy, dx in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                ny, nx = cy + dy, cx + dx
                
                # Check if within bounds
                if 0 <= ny < h and 0 <= nx < w:
                    # Check if not visited and color matches
                    if not visited[ny, nx] and color_match(self.layers[self.active_layer][ny, nx], target_color_rgba, tolerance):
                        queue.append((ny, nx))
                        visited[ny, nx] = True
    
    def apply_effect(self, effect_type):
        """Apply visual effect to the active layer."""
        if len(self.layers) == 0 or self.active_layer >= len(self.layers):
            return
            
        layer = self.layers[self.active_layer]
        
        # Extract RGB and alpha channels
        rgb = layer[:, :, :3].copy()
        alpha = layer[:, :, 3].copy()
        
        if effect_type == "blur":
            # Apply Gaussian blur
            rgb = cv2.GaussianBlur(rgb, (15, 15), 0)
        
        elif effect_type == "sharpen":
            # Apply sharpening using a kernel
            kernel = np.array([[-1, -1, -1],
                               [-1,  9, -1],
                               [-1, -1, -1]])
            rgb = cv2.filter2D(rgb, -1, kernel)
        
        elif effect_type == "grayscale":
            # Convert to grayscale
            gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
            rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        
        elif effect_type == "invert":
            # Invert colors
            rgb = 255 - rgb
        
        # Recombine channels and update layer
        self.layers[self.active_layer][:, :, :3] = rgb
        self.layers[self.active_layer][:, :, 3] = alpha
        
        # Update the canvas
        self.update_canvas()
    
    def draw_toolbar(self, frame):
        """Draw toolbar with drawing tools."""
        h, w, _ = frame.shape
        
        # Create semi-transparent background
        toolbar_width = 80
        toolbar_height = h - 100
        toolbar_x = w - toolbar_width - 10
        toolbar_y = 50
        
        overlay = frame.copy()
        cv2.rectangle(overlay, (toolbar_x, toolbar_y), 
                     (toolbar_x + toolbar_width, toolbar_y + toolbar_height), 
                     (30, 30, 40), -1)
        
        # Add a subtle border
        cv2.rectangle(overlay, (toolbar_x, toolbar_y), 
                     (toolbar_x + toolbar_width, toolbar_y + toolbar_height), 
                     (100, 100, 120), 2)
        
        # Blend the overlay with the original frame
        alpha = 0.8
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        
        # Tool buttons
        button_size = 60
        button_margin = 10
        
        # Define tool icons as simple shapes/text instead of emoji
        tools_with_icons = [
            ("brush", "Draw", "Draw"),
            ("eraser", "Erase", "Erase"),
            ("line", "Line", "Line"),
            ("rectangle", "Rect", "Rectangle"),
            ("circle", "Circ", "Circle"),
            ("fill", "Fill", "Fill")
        ]
        
        for i, (tool_name, icon_text, tooltip) in enumerate(tools_with_icons):
            button_x = toolbar_x + (toolbar_width - button_size) // 2
            button_y = toolbar_y + button_margin + i * (button_size + button_margin)
            
            # Highlight selected tool
            button_color = (60, 60, 100) if tool_name == self.current_tool else (50, 50, 70)
            highlight_color = (100, 100, 255) if tool_name == self.current_tool else (80, 80, 100)
            
            # Draw button background
            cv2.rectangle(frame, (button_x, button_y), 
                         (button_x + button_size, button_y + button_size), 
                         button_color, -1)
            
            # Draw button border
            cv2.rectangle(frame, (button_x, button_y), 
                         (button_x + button_size, button_y + button_size), 
                         highlight_color, 2)
            
            # Draw button icon (simple shape instead of emoji)
            icon_center_x = button_x + button_size // 2
            icon_center_y = button_y + button_size // 2
            
            # Draw different icon for each tool
            if tool_name == "brush":
                # Draw brush icon
                cv2.line(frame, (icon_center_x-15, icon_center_y+15), (icon_center_x+5, icon_center_y-15), (255, 255, 255), 3)
                cv2.circle(frame, (icon_center_x+7, icon_center_y-17), 5, (255, 255, 255), -1)
            elif tool_name == "eraser":
                # Draw eraser icon
                cv2.rectangle(frame, (icon_center_x-10, icon_center_y-10), (icon_center_x+10, icon_center_y+5), (255, 255, 255), -1)
                cv2.rectangle(frame, (icon_center_x-15, icon_center_y+5), (icon_center_x+15, icon_center_y+15), (150, 150, 150), -1)
            elif tool_name == "line":
                # Draw line icon
                cv2.line(frame, (icon_center_x-15, icon_center_y-15), (icon_center_x+15, icon_center_y+15), (255, 255, 255), 2)
            elif tool_name == "rectangle":
                # Draw rectangle icon
                cv2.rectangle(frame, (icon_center_x-15, icon_center_y-10), (icon_center_x+15, icon_center_y+10), (255, 255, 255), 2)
            elif tool_name == "circle":
                # Draw circle icon
                cv2.circle(frame, (icon_center_x, icon_center_y), 15, (255, 255, 255), 2)
            elif tool_name == "fill":
                # Draw fill icon
                cv2.rectangle(frame, (icon_center_x-12, icon_center_y-12), (icon_center_x+12, icon_center_y+12), (255, 255, 255), 1)
                # Fill pattern
                for y in range(-8, 9, 4):
                    for x in range(-8, 9, 4):
                        cv2.circle(frame, (icon_center_x+x, icon_center_y+y), 1, (255, 255, 255), -1)
            
            # Draw tool name
            cv2.putText(frame, tooltip, (button_x, button_y + button_size + 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            
        # Brush style selector (at the bottom of toolbar)
        brush_selector_y = toolbar_y + toolbar_height - 150
        cv2.putText(frame, "Brush Style:", (toolbar_x, brush_selector_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        for i, style in enumerate(self.brush_styles):
            style_y = brush_selector_y + 25 + i * 25
            
            # Highlight selected style
            text_color = (255, 255, 255) if style == self.current_brush else (150, 150, 150)
            
            cv2.putText(frame, style.capitalize(), (toolbar_x + 10, style_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)
            
            # Add selection indicator
            if style == self.current_brush:
                cv2.circle(frame, (toolbar_x + 5, style_y - 4), 3, (100, 100, 255), -1)
    
    def draw_color_palette(self, frame):
        """Draw the saved color palette."""
        if not self.saved_colors:
            return
            
        palette_x = 20
        palette_y = 470
        swatch_size = 30
        swatch_margin = 5
        
        # Create background panel
        panel_width = len(self.saved_colors) * (swatch_size + swatch_margin) + swatch_margin
        panel_height = swatch_size + 2 * swatch_margin
        
        # Draw panel background
        cv2.rectangle(frame, (palette_x - 5, palette_y - 5), 
                     (palette_x + panel_width, palette_y + panel_height), 
                     (30, 30, 40), -1)
        cv2.rectangle(frame, (palette_x - 5, palette_y - 5), 
                     (palette_x + panel_width, palette_y + panel_height), 
                     (100, 100, 120), 1)
        
        # Title
        cv2.putText(frame, "Color Palette", (palette_x, palette_y - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Draw color swatches
        for i, color in enumerate(self.saved_colors):
            swatch_x = palette_x + i * (swatch_size + swatch_margin)
            
            # Draw color swatch
            cv2.rectangle(frame, (swatch_x, palette_y), 
                         (swatch_x + swatch_size, palette_y + swatch_size), 
                         color, -1)
            
            # Highlight current color
            if color == self.drawing_color:
                cv2.rectangle(frame, (swatch_x - 2, palette_y - 2), 
                             (swatch_x + swatch_size + 2, palette_y + swatch_size + 2), 
                             (255, 255, 255), 2)
            else:
                cv2.rectangle(frame, (swatch_x, palette_y), 
                             (swatch_x + swatch_size, palette_y + swatch_size), 
                             (50, 50, 50), 1)
    
    def draw_layers_panel(self, frame):
        """Draw the layers panel showing all layers."""
        if not self.show_layer_panel:
            return
            
        h, w, _ = frame.shape
        
        # Panel dimensions
        panel_width = 150
        panel_height = min(300, h - 100)
        panel_x = 10
        panel_y = 100
        
        # Create semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (panel_x, panel_y), 
                     (panel_x + panel_width, panel_y + panel_height), 
                     (30, 30, 40), -1)
        
        # Add border
        cv2.rectangle(overlay, (panel_x, panel_y), 
                     (panel_x + panel_width, panel_y + panel_height), 
                     (100, 100, 120), 2)
        
        # Blend overlay
        alpha = 0.8
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        
        # Panel title
        cv2.putText(frame, "Layers", (panel_x + 5, panel_y + 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Layer buttons
        layer_height = 30
        layer_margin = 5
        
        # Add layer button
        add_y = panel_y + 35
        cv2.rectangle(frame, (panel_x + 5, add_y), 
                     (panel_x + 30, add_y + 20), (50, 120, 50), -1)
        cv2.putText(frame, "+", (panel_x + 12, add_y + 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Delete layer button
        cv2.rectangle(frame, (panel_x + 40, add_y), 
                     (panel_x + 65, add_y + 20), (120, 50, 50), -1)
        cv2.putText(frame, "-", (panel_x + 47, add_y + 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Move up button
        cv2.rectangle(frame, (panel_x + 75, add_y), 
                     (panel_x + 100, add_y + 20), (50, 50, 120), -1)
        cv2.putText(frame, "↑", (panel_x + 82, add_y + 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Move down button
        cv2.rectangle(frame, (panel_x + 110, add_y), 
                     (panel_x + 135, add_y + 20), (50, 50, 120), -1)
        cv2.putText(frame, "↓", (panel_x + 117, add_y + 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Layer list - show top layer first
        start_y = add_y + 30
        for i, layer in enumerate(reversed(self.layers)):
            layer_index = len(self.layers) - 1 - i
            layer_y = start_y + i * (layer_height + layer_margin)
            
            # Skip if outside panel
            if layer_y + layer_height > panel_y + panel_height:
                break
                
            # Layer background
            bg_color = (70, 70, 100) if layer_index == self.active_layer else (50, 50, 70)
            cv2.rectangle(frame, (panel_x + 5, layer_y), 
                         (panel_x + panel_width - 10, layer_y + layer_height), 
                         bg_color, -1)
            
            # Layer border
            border_color = (150, 150, 255) if layer_index == self.active_layer else (100, 100, 120)
            cv2.rectangle(frame, (panel_x + 5, layer_y), 
                         (panel_x + panel_width - 10, layer_y + layer_height), 
                         border_color, 1)
            
            # Layer label
            cv2.putText(frame, f"Layer {layer_index + 1}", (panel_x + 10, layer_y + 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Visibility icon 
            # TODO: implement layer visibility toggle
            cv2.circle(frame, (panel_x + panel_width - 20, layer_y + layer_height//2), 5, (200, 200, 200), 1)
    
    def draw_ui(self, frame):
        """Draw the main UI elements."""
        h, w, _ = frame.shape
        
        # Create a semi-transparent overlay for header area
        ui_overlay = frame.copy()
        cv2.rectangle(ui_overlay, (0, 0), (w, 70), (30, 30, 40), -1)
        
        # Create a semi-transparent overlay for footer area
        cv2.rectangle(ui_overlay, (0, h-60), (w, h), (30, 30, 40), -1)
        
        # Blend the overlay with the original frame
        alpha = 0.7
        cv2.addWeighted(ui_overlay, alpha, frame, 1 - alpha, 0, frame)
        
        # Draw top toolbar
        top_margin = 15
        
        # Add app title
        cv2.putText(frame, "Finger Drawing App", (20, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Current drawing mode and tool
        mode_text = f"Tool: {self.current_tool.capitalize()}"
        cv2.putText(frame, mode_text, (300, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        
        # Current brush settings
        brush_text = f"Brush: {self.brush_thickness}px | Style: {self.current_brush.capitalize()}"
        cv2.putText(frame, brush_text, (500, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        # Current color with swatch
        color_swatch_x = w - 180
        cv2.rectangle(frame, (color_swatch_x, 15), (color_swatch_x + 40, 45), self.drawing_color, -1)
        cv2.rectangle(frame, (color_swatch_x, 15), (color_swatch_x + 40, 45), (255, 255, 255), 1)
        
        # Add help icon
        help_icon_x = w - 50
        help_icon_y = 30
        cv2.circle(frame, (help_icon_x, help_icon_y), 15, (70, 70, 120), -1)
        cv2.putText(frame, "?", (help_icon_x - 5, help_icon_y + 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        
        # Draw color picker if enabled
        if self.show_color_picker:
            cp_x, cp_y = self.color_picker_pos
            cp_w, cp_h = self.color_picker_size
            
            # Draw background panel
            cv2.rectangle(frame, (cp_x - 10, cp_y - 40), (cp_x + cp_w + 10, cp_y + cp_h + 60), (30, 30, 40), -1)
            cv2.rectangle(frame, (cp_x - 10, cp_y - 40), (cp_x + cp_w + 10, cp_y + cp_h + 60), (100, 100, 120), 1)
            
            # Title
            cv2.putText(frame, "Color Picker", (cp_x, cp_y - 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            # Draw color picker with improved visuals
            frame[cp_y:cp_y+cp_h, cp_x:cp_x+cp_w] = self.color_picker
            
            # Draw border
            cv2.rectangle(frame, (cp_x, cp_y), (cp_x+cp_w, cp_y+cp_h), (255, 255, 255), 1)
            
            # Add instruction for color picker
            cv2.putText(frame, "Pinch to select a color", (cp_x, cp_y+cp_h+20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            # Add save to palette button
            cv2.rectangle(frame, (cp_x, cp_y+cp_h+30), (cp_x+120, cp_y+cp_h+50), (50, 100, 50), -1)
            cv2.rectangle(frame, (cp_x, cp_y+cp_h+30), (cp_x+120, cp_y+cp_h+50), (100, 150, 100), 1)
            cv2.putText(frame, "Save to palette", (cp_x+5, cp_y+cp_h+45), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Draw toolbar with tools
        if self.show_toolbar:
            self.draw_toolbar(frame)
        
        # Draw color palette
        self.draw_color_palette(frame)
        
        # Draw layers panel if enabled
        self.draw_layers_panel(frame)
        
        # Draw effect selector if in effect mode
        if self.current_mode == "effect":
            effect_x = 20
            effect_y = 200
            cv2.putText(frame, "Select Effect:", (effect_x, effect_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            for i, effect in enumerate(self.effects):
                effect_y_pos = effect_y + 30 + i * 25
                text_color = (255, 255, 255) if effect == self.current_effect else (150, 150, 150)
                cv2.putText(frame, effect.capitalize(), (effect_x + 20, effect_y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)
                
                if effect == self.current_effect:
                    cv2.circle(frame, (effect_x + 10, effect_y_pos - 5), 3, (100, 100, 255), -1)
        
        # Footer with instructions
        instructions = [
            "GESTURES: Index finger = Draw | Open palm = Erase | Peace sign = Color | 3 Fingers = Tools",
            "KEYS: [r,g,b,w,k] = Colors | [+/-] = Brush size | [c] = Clear | [s] = Save | [l] = Layers | [z] = Undo | [f] = Fullscreen"
        ]
        
        for i, instruction in enumerate(instructions):
            y_pos = h - 40 + i * 20
            cv2.putText(frame, instruction, (20, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    def handle_keyboard_input(self, key):
        """Handle keyboard inputs."""
        if key == 27:  # ESC key to exit
            return False
        elif key == ord('h'):  # 'h' key to show help
            self.show_help = True
        elif key == ord('f'):  # 'f' key to toggle fullscreen
            self.fullscreen = not self.fullscreen
            if self.fullscreen:
                cv2.setWindowProperty(self.window_title, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            else:
                cv2.setWindowProperty(self.window_title, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
        elif key == ord('c'):  # 'c' key to clear canvas
            self.save_current_state()  # Save current state for undo
            self.layers[self.active_layer] = np.zeros_like(self.layers[self.active_layer])
            self.update_canvas()
        elif key == ord('r'):  # 'r' key to set color to red
            self.drawing_color = (0, 0, 255)
            self.current_mode = "draw"
            self.current_tool = "brush"
        elif key == ord('g'):  # 'g' key to set color to green
            self.drawing_color = (0, 255, 0)
            self.current_mode = "draw"
            self.current_tool = "brush"
        elif key == ord('b'):  # 'b' key to set color to blue
            self.drawing_color = (255, 0, 0)
            self.current_mode = "draw"
            self.current_tool = "brush"
        elif key == ord('w'):  # 'w' key to set color to white
            self.drawing_color = (255, 255, 255)
            self.current_mode = "draw"
            self.current_tool = "brush"
        elif key == ord('k'):  # 'k' key to set color to black
            self.drawing_color = (0, 0, 0)
            self.current_mode = "draw"
            self.current_tool = "brush"
        elif key == ord('p'):  # 'p' key to toggle color picker
            self.show_color_picker = not self.show_color_picker
            if self.show_color_picker:
                self.current_mode = "color_pick"
            else:
                self.current_mode = "draw"
        elif key == ord('l'):  # 'l' key to toggle layer panel
            self.show_layer_panel = not self.show_layer_panel
        elif key == ord('t'):  # 't' key to toggle toolbar
            self.show_toolbar = not self.show_toolbar
        elif key == ord('z'):  # 'z' key for undo
            self.undo()
        elif key == ord('y'):  # 'y' key for redo
            self.redo()
        elif key == ord('a'):  # 'a' key to add new layer
            self.save_current_state()
            self.add_layer()
        elif key == ord('d'):  # 'd' key to delete current layer
            if len(self.layers) > 1:
                self.save_current_state()
                self.delete_layer()
        elif key == ord('+') or key == ord('='):  # '+' key to increase brush thickness
            self.brush_thickness = min(30, self.brush_thickness + 1)
        elif key == ord('-') or key == ord('_'):  # '-' key to decrease brush thickness
            self.brush_thickness = max(1, self.brush_thickness - 1)
        elif key == ord('1'):  # Number keys for brush styles
            self.current_brush = "solid"
        elif key == ord('2'):
            self.current_brush = "airbrush"
        elif key == ord('3'):
            self.current_brush = "marker"
        elif key == ord('4'):
            self.current_brush = "pencil"
        elif key == ord('s'):  # 's' key to save canvas
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            filename = f"drawing_{timestamp}.png"
            
            # Make sure drawings directory exists
            if not os.path.exists("drawings"):
                os.makedirs("drawings")
            
            # Save to drawings folder
            filepath = os.path.join("drawings", filename)
            
            # Composite all layers for saving
            self.update_canvas()
            
            # Convert BGRA to RGBA for saving
            save_image = cv2.cvtColor(self.canvas, cv2.COLOR_BGRA2RGBA)
            
            cv2.imwrite(filepath, save_image)
            print(f"Saved drawing as {filepath}")
        
        return True
    
    def render_help_screen(self, frame):
        """Render an interactive help screen."""
        h, w, _ = frame.shape
        
        # Create a dark semi-transparent overlay
        help_overlay = np.zeros_like(frame)
        
        # Draw background
        cv2.rectangle(help_overlay, (0, 0), (w, h), (20, 20, 30), -1)
        
        # Draw content panel
        panel_margin = 50
        cv2.rectangle(help_overlay, (panel_margin, panel_margin), 
                     (w - panel_margin, h - panel_margin), (40, 40, 60), -1)
        cv2.rectangle(help_overlay, (panel_margin, panel_margin), 
                     (w - panel_margin, h - panel_margin), (80, 80, 120), 2)
        
        # Title
        cv2.putText(help_overlay, "Finger Drawing App - Help", (w//2 - 200, panel_margin + 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Divide into columns
        col_width = (w - 2 * panel_margin) // 2 - 20
        
        # Left column: Gesture controls
        left_x = panel_margin + 20
        top_y = panel_margin + 80
        cv2.putText(help_overlay, "HAND GESTURES:", (left_x, top_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 255), 1)
        
        gestures = [
            "Extend index finger to draw",
            "Open palm to erase (size changes with distance)",
            "Index + middle fingers (peace sign) for color picker",
            "Pinch (thumb + index) to select color",
            "Three extended fingers (index, middle, ring) to open tools",
            "Fist gesture for undo",
            "OK sign (thumb+index circle) to select a tool"
        ]
        
        for i, gesture in enumerate(gestures):
            y = top_y + 30 + i * 25
            cv2.putText(help_overlay, gesture, (left_x, y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Middle column: Keyboard controls
        mid_x = left_x + col_width
        cv2.putText(help_overlay, "KEYBOARD CONTROLS:", (mid_x, top_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 255), 1)
        
        keyboard_controls = [
            "[r,g,b,w,k] - Change color (red, green, blue, white, black)",
            "[+/-] - Increase/decrease brush thickness",
            "[1-4] - Change brush style (solid, airbrush, marker, pencil)",
            "[c] - Clear the canvas",
            "[s] - Save your drawing",
            "[z/y] - Undo/redo",
            "[l] - Toggle layers panel",
            "[t] - Toggle toolbar",
            "[a/d] - Add/delete layer",
            "[f] - Toggle fullscreen mode",
            "[h] - Show/hide this help screen",
            "[ESC] - Exit application"
        ]
        
        for i, control in enumerate(keyboard_controls):
            y = top_y + 30 + i * 25
            cv2.putText(help_overlay, control, (mid_x, y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Draw examples for tools
        tool_section_y = top_y + 350
        cv2.putText(help_overlay, "DRAWING TOOLS:", (left_x, tool_section_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 255), 1)
        
        tools_description = [
            "Brush - Freeform drawing with different styles",
            "Eraser - Remove parts of your drawing",
            "Line - Draw straight lines (click start & end points)",
            "Rectangle - Draw rectangles (click opposite corners)",
            "Circle - Draw circles (click center & radius point)",
            "Fill - Fill connected areas with the same color"
        ]
        
        for i, desc in enumerate(tools_description):
            y = tool_section_y + 30 + i * 25
            cv2.putText(help_overlay, desc, (left_x, y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Press any key message
        cv2.putText(help_overlay, "Press any key to continue", (w//2 - 120, h - panel_margin - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 255), 1)
        
        # Blend overlay with original frame
        alpha = 0.9
        result = cv2.addWeighted(frame, 1 - alpha, help_overlay, alpha, 0)
        
        return result
    
    def process_hands(self, frame, results):
        """Process hand gestures and update drawing."""
        h, w, _ = frame.shape
        frame_with_drawing = frame.copy()
        
        # Ensure the canvas has proper dimensions
        if self.canvas is None or self.canvas.shape[0] != h or self.canvas.shape[1] != w:
            self.canvas = np.zeros((h, w, 4), dtype=np.uint8)
            
            # Also ensure all layers have the right dimensions
            for i in range(len(self.layers)):
                if self.layers[i].shape[0] != h or self.layers[i].shape[1] != w:
                    self.layers[i] = cv2.resize(self.layers[i], (w, h))
        
        # Overlay the canvas onto the frame
        if self.canvas.shape[2] == 4:  # If canvas has alpha channel
            mask = self.canvas[:, :, 3] > 0
            if np.any(mask):  # Only blend if there's something to blend
                frame_with_drawing[mask] = cv2.addWeighted(frame_with_drawing[mask], 0.3, self.canvas[:, :, :3][mask], 0.7, 0)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw hand landmarks with improved visualization
                self.mp_drawing.draw_landmarks(
                    frame_with_drawing, 
                    hand_landmarks, 
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )
                
                # Get index finger tip position
                index_finger_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
                x, y = int(index_finger_tip.x * w), int(index_finger_tip.y * h)
                
                # Check for fist gesture (undo)
                if self.detect_fist_gesture(hand_landmarks):
                    # Add delay to prevent multiple undos
                    current_time = time.time()
                    if not hasattr(self, 'last_undo_time') or current_time - self.last_undo_time > 1.0:
                        self.undo()
                        self.last_undo_time = current_time
                        # Visual feedback
                        cv2.putText(frame_with_drawing, "Undo", (x + 20, y), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    continue
                
                # Check for OK gesture (tool selection)
                is_ok_gesture = self.detect_ok_gesture(hand_landmarks, frame.shape)
                if is_ok_gesture and self.show_toolbar:
                    # Tool buttons area
                    toolbar_width = 80
                    toolbar_x = w - toolbar_width - 10
                    toolbar_y = 50
                    button_size = 60
                    button_margin = 10
                    
                    # Check if within toolbar area
                    if toolbar_x <= x <= toolbar_x + toolbar_width:
                        for i, tool in enumerate(self.tools):
                            button_y = toolbar_y + button_margin + i * (button_size + button_margin)
                            
                            # Check if pointing at this tool
                            if button_y <= y <= button_y + button_size:
                                self.current_tool = tool
                                self.current_mode = "draw" if tool == "brush" else "erase" if tool == "eraser" else "shape"
                                
                                # Visual feedback
                                cv2.putText(frame_with_drawing, f"Selected: {tool}", (x - 120, y - 20), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    
                    # Check brush style selection area
                    brush_selector_y = toolbar_y + 500
                    if toolbar_x <= x <= toolbar_x + toolbar_width and brush_selector_y <= y <= brush_selector_y + 100:
                        for i, style in enumerate(self.brush_styles):
                            style_y = brush_selector_y + 25 + i * 25
                            
                            # Check if pointing at this style
                            if style_y - 15 <= y <= style_y + 5:
                                self.current_brush = style
                                
                                # Visual feedback
                                cv2.putText(frame_with_drawing, f"Brush: {style}", (x - 120, y - 20), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    
                    continue
                
                # Check for three finger gesture (tool selection)
                if self.detect_tool_selection_gesture(hand_landmarks):
                    self.show_toolbar = True
                    self.current_mode = "select"
                    
                    # Visual feedback
                    cv2.putText(frame_with_drawing, "Tool Selection Mode", (x + 20, y - 20), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    continue
                
                # Check if palm is showing (for erasing)
                if self.is_palm_showing(hand_landmarks):
                    self.is_drawing = False
                    self.is_erasing = True
                    self.current_mode = "erase"
                    self.current_tool = "eraser"
                    
                    # Calculate eraser size based on palm distance from camera
                    palm_width = self.calculate_palm_width(hand_landmarks)
                    
                    # Map the palm width to an eraser size (larger palm width = closer to camera = larger eraser)
                    min_palm_width = 30
                    max_palm_width = 150
                    min_eraser = 15
                    max_eraser = 100
                    
                    # Normalize the palm width between 0 and 1
                    normalized_width = max(0, min(1, (palm_width - min_palm_width) / (max_palm_width - min_palm_width)))
                    
                    # Map to eraser size with easing for smoother transitions
                    eased_value = normalized_width * normalized_width
                    eraser_radius = int(min_eraser + eased_value * (max_eraser - min_eraser))
                    
                    # Create a circular eraser on the active layer
                    cv2.circle(self.layers[self.active_layer], (x, y), eraser_radius, (0, 0, 0, 0), -1)
                    self.update_canvas()
                    
                    # Draw eraser preview with visual depth cue
                    # Inner circle
                    cv2.circle(frame_with_drawing, (x, y), eraser_radius - 5, (100, 100, 100), 1)
                    # Main circle
                    cv2.circle(frame_with_drawing, (x, y), eraser_radius, (255, 255, 255), 2)
                    # Outer glow
                    cv2.circle(frame_with_drawing, (x, y), eraser_radius + 3, (200, 200, 200), 1)
                    
                    # Create a colored indicator for distance
                    distance_color = (
                        int(255 * (1 - normalized_width)),  # R - more red when farther (smaller eraser)
                        int(255 * normalized_width),        # G - more green when closer (larger eraser)
                        0                                   # B
                    )
                    
                    # Show eraser size with distance indicator
                    cv2.putText(frame_with_drawing, f"Eraser: {eraser_radius}px", 
                               (x + eraser_radius + 5, y), cv2.FONT_HERSHEY_SIMPLEX, 
                               0.6, distance_color, 2)
                    
                    # Distance hint
                    hint_text = "Closer" if normalized_width > 0.7 else "Further" if normalized_width < 0.3 else "Mid-range"
                    cv2.putText(frame_with_drawing, hint_text, 
                               (x + eraser_radius + 5, y + 25), cv2.FONT_HERSHEY_SIMPLEX, 
                               0.5, distance_color, 1)
                    
                # Check for color selection gesture (index + middle finger extended)
                elif (self.is_finger_extended(hand_landmarks, self.mp_hands.HandLandmark.INDEX_FINGER_TIP, 
                                           self.mp_hands.HandLandmark.INDEX_FINGER_PIP) and
                     self.is_finger_extended(hand_landmarks, self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP, 
                                          self.mp_hands.HandLandmark.MIDDLE_FINGER_PIP) and
                     not self.is_finger_extended(hand_landmarks, self.mp_hands.HandLandmark.RING_FINGER_TIP, 
                                              self.mp_hands.HandLandmark.RING_FINGER_PIP) and
                     not self.is_finger_extended(hand_landmarks, self.mp_hands.HandLandmark.PINKY_TIP, 
                                              self.mp_hands.HandLandmark.PINKY_PIP)):
                    
                    self.show_color_picker = True
                    self.current_mode = "color_pick"
                    self.is_drawing = False
                    self.is_erasing = False
                    self.prev_finger_pos = None
                    
                    # Visual feedback
                    cv2.putText(frame_with_drawing, "Color Selection Mode", (x + 10, y - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    
                # Check for pinch gesture (to pick color)
                elif self.show_color_picker:
                    is_pinching, pinch_distance, pinch_pos = detect_pinch_gesture(
                        hand_landmarks, self.mp_hands, frame.shape
                    )
                    
                    if is_pinching:
                        # Check if pinching in color picker area
                        cp_x, cp_y = self.color_picker_pos
                        cp_w, cp_h = self.color_picker_size
                        
                        if cp_x <= pinch_pos[0] <= cp_x + cp_w and cp_y <= pinch_pos[1] <= cp_y + cp_h:
                            # Get relative position in color picker
                            rel_x = pinch_pos[0] - cp_x
                            rel_y = pinch_pos[1] - cp_y
                            
                            # Update drawing color
                            self.drawing_color = get_color_from_position(self.color_picker, (rel_x, rel_y))
                            
                            # Visual feedback
                            cv2.circle(frame_with_drawing, pinch_pos, 15, self.drawing_color, -1)
                            cv2.circle(frame_with_drawing, pinch_pos, 15, (255, 255, 255), 2)
                            
                        # Check if pinching the "save to palette" button
                        button_y = cp_y + cp_h + 30
                        if cp_x <= pinch_pos[0] <= cp_x + 120 and button_y <= pinch_pos[1] <= button_y + 20:
                            self.add_to_palette(self.drawing_color)
                            
                            # Visual feedback
                            cv2.putText(frame_with_drawing, "Color saved to palette!", 
                                       (pinch_pos[0] - 50, pinch_pos[1] - 10), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        
                        # Check if pinching in color palette
                        palette_x = 20
                        palette_y = 470
                        swatch_size = 30
                        swatch_margin = 5
                        
                        for i, color in enumerate(self.saved_colors):
                            swatch_x = palette_x + i * (swatch_size + swatch_margin)
                            
                            if (swatch_x <= pinch_pos[0] <= swatch_x + swatch_size and 
                                palette_y <= pinch_pos[1] <= palette_y + swatch_size):
                                self.drawing_color = color
                                self.show_color_picker = False
                                self.current_mode = "draw"
                                self.current_tool = "brush"
                                
                                # Visual feedback
                                cv2.circle(frame_with_drawing, pinch_pos, 15, self.drawing_color, -1)
                                cv2.circle(frame_with_drawing, pinch_pos, 15, (255, 255, 255), 2)
                                cv2.putText(frame_with_drawing, "Color selected!", 
                                           (pinch_pos[0] - 50, pinch_pos[1] - 10), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # Handle drawing with index finger
                elif self.is_finger_extended(hand_landmarks, self.mp_hands.HandLandmark.INDEX_FINGER_TIP, 
                                          self.mp_hands.HandLandmark.INDEX_FINGER_PIP):
                    
                    # Save the drawing state before starting a new stroke
                    if self.prev_finger_pos is None:
                        self.save_current_state()
                    
                    # Different tools based on current selection
                    if self.current_tool == "brush":
                        self.is_drawing = True
                        self.is_erasing = False
                        
                        # Draw brush preview
                        draw_brush_preview(frame_with_drawing, (x, y), self.drawing_color, self.brush_thickness)
                        
                        if self.prev_finger_pos is not None:
                            # Apply the selected brush style
                            self.apply_brush_style(
                                self.layers[self.active_layer],
                                self.prev_finger_pos,
                                (x, y),
                                self.drawing_color,
                                self.brush_thickness
                            )
                            self.update_canvas()
                        
                        self.prev_finger_pos = (x, y)
                    
                    elif self.current_tool == "fill":
                        # Fill tool - perform flood fill when finger is "pressed" (held still)
                        if self.prev_finger_pos is not None:
                            # Calculate movement from previous position
                            movement = calculate_distance(self.prev_finger_pos, (x, y))
                            
                            # If finger is held relatively still, perform fill
                            if movement < 5:
                                self.fill_area(x, y, None, self.drawing_color)
                                self.update_canvas()
                                
                                # Visual feedback
                                cv2.putText(frame_with_drawing, "Fill", (x + 20, y), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.drawing_color, 2)
                                
                                # Reset to avoid multiple fills
                                self.prev_finger_pos = None
                            else:
                                self.prev_finger_pos = (x, y)
                        else:
                            self.prev_finger_pos = (x, y)
                    
                    elif self.current_tool == "line" or self.current_tool == "rectangle" or self.current_tool == "circle":
                        # Shape drawing - need two points
                        if self.shape_start is None:
                            # First point of shape
                            self.shape_start = (x, y)
                            
                            # Visual feedback for shape starting point
                            cv2.circle(frame_with_drawing, self.shape_start, 5, self.drawing_color, -1)
                            cv2.putText(frame_with_drawing, "Shape start", (x + 10, y - 10), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.drawing_color, 1)
                        else:
                            # Draw preview of shape
                            shape_preview = np.zeros_like(self.canvas)
                            
                            if self.current_tool == "line":
                                cv2.line(shape_preview, self.shape_start, (x, y), 
                                        (*self.drawing_color, 255), self.brush_thickness)
                                
                                # Show preview on frame
                                mask = shape_preview[:, :, 3] > 0
                                if np.any(mask):
                                    frame_with_drawing[mask] = cv2.addWeighted(
                                        frame_with_drawing[mask], 0.5, shape_preview[:, :, :3][mask], 0.5, 0
                                    )
                                
                                # Check if finger held still to complete shape
                                if self.prev_finger_pos is not None:
                                    movement = calculate_distance(self.prev_finger_pos, (x, y))
                                    
                                    if movement < 5:
                                        # Complete shape
                                        cv2.line(self.layers[self.active_layer], 
                                                self.shape_start, (x, y), 
                                                (*self.drawing_color, 255), self.brush_thickness)
                                        self.update_canvas()
                                        
                                        # Visual feedback
                                        cv2.putText(frame_with_drawing, "Line complete", (x + 10, y + 20), 
                                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                                        
                                        # Reset shape state
                                        self.shape_start = None
                            
                            elif self.current_tool == "rectangle":
                                cv2.rectangle(shape_preview, self.shape_start, (x, y), 
                                            (*self.drawing_color, 255), self.brush_thickness)
                                
                                # Show preview on frame
                                mask = shape_preview[:, :, 3] > 0
                                if np.any(mask):
                                    frame_with_drawing[mask] = cv2.addWeighted(
                                        frame_with_drawing[mask], 0.5, shape_preview[:, :, :3][mask], 0.5, 0
                                    )
                                
                                # Check if finger held still to complete shape
                                if self.prev_finger_pos is not None:
                                    movement = calculate_distance(self.prev_finger_pos, (x, y))
                                    
                                    if movement < 5:
                                        # Complete shape
                                        cv2.rectangle(self.layers[self.active_layer], 
                                                    self.shape_start, (x, y), 
                                                    (*self.drawing_color, 255), self.brush_thickness)
                                        self.update_canvas()
                                        
                                        # Visual feedback
                                        cv2.putText(frame_with_drawing, "Rectangle complete", (x + 10, y + 20), 
                                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                                        
                                        # Reset shape state
                                        self.shape_start = None
                            
                            elif self.current_tool == "circle":
                                # Calculate radius from start point to current point
                                radius = int(math.sqrt((x - self.shape_start[0])**2 + (y - self.shape_start[1])**2))
                                
                                cv2.circle(shape_preview, self.shape_start, radius, 
                                          (*self.drawing_color, 255), self.brush_thickness)
                                
                                # Show preview on frame
                                mask = shape_preview[:, :, 3] > 0
                                if np.any(mask):
                                    frame_with_drawing[mask] = cv2.addWeighted(
                                        frame_with_drawing[mask], 0.5, shape_preview[:, :, :3][mask], 0.5, 0
                                    )
                                
                                # Check if finger held still to complete shape
                                if self.prev_finger_pos is not None:
                                    movement = calculate_distance(self.prev_finger_pos, (x, y))
                                    
                                    if movement < 5:
                                        # Complete shape
                                        cv2.circle(self.layers[self.active_layer], 
                                                 self.shape_start, radius, 
                                                 (*self.drawing_color, 255), self.brush_thickness)
                                        self.update_canvas()
                                        
                                        # Visual feedback
                                        cv2.putText(frame_with_drawing, "Circle complete", (x + 10, y + 20), 
                                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                                        
                                        # Reset shape state
                                        self.shape_start = None
                    
                    # Store current position for movement detection
                    self.prev_finger_pos = (x, y)
                else:
                    self.is_drawing = False
                    self.prev_finger_pos = None
        else:
            self.is_drawing = False
            self.is_erasing = False
            self.prev_finger_pos = None
        
        return frame_with_drawing
    
    def create_startup_screen(self, width, height):
        """Create a startup screen."""
        # Create a dark background with gradient
        startup = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Create a gradient background
        for y in range(height):
            for x in range(width):
                # Create a blue-purple gradient
                startup[y, x] = [
                    int(50 + (x * 50 / width)),  # B
                    int(20 + (y * 20 / height)),  # G
                    int(50 + ((x + y) * 50 / (width + height)))  # R
                ]
        
        # Add a title
        title = "Finger Drawing App"
        font_scale = 1.5
        font_thickness = 3
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Get text size
        text_size = cv2.getTextSize(title, font, font_scale, font_thickness)[0]
        text_x = (width - text_size[0]) // 2
        text_y = height // 4
        
        # Draw title text with shadow effect
        cv2.putText(startup, title, (text_x + 3, text_y + 3), font, font_scale, (0, 0, 0), font_thickness)
        cv2.putText(startup, title, (text_x, text_y), font, font_scale, (255, 255, 255), font_thickness)
        
        # Add subtitle
        subtitle = "Draw with gestures, create with your hands"
        sub_font_scale = 0.8
        sub_text_size = cv2.getTextSize(subtitle, font, sub_font_scale, 2)[0]
        sub_x = (width - sub_text_size[0]) // 2
        sub_y = text_y + 50
        
        cv2.putText(startup, subtitle, (sub_x, sub_y), font, sub_font_scale, (200, 200, 255), 2)
        
        # Add feature list
        features = [
            "✓ Different brush styles",
            "✓ Multiple drawing tools (brush, shapes, fill)",
            "✓ Layer support",
            "✓ Color picker with saved palette",
            "✓ Undo/redo",
            "✓ Visual effects", 
            "✓ Natural gesture controls"
        ]
        
        feature_y = sub_y + 80
        for feature in features:
            cv2.putText(startup, feature, (width//2 - 200, feature_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 220, 220), 1)
            feature_y += 35
        
        # Add version info
        version = "v1.4.2"
        cv2.putText(startup, version, (20, height - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 200), 1)
        
        # Add instruction to continue
        instruction = "Press any key to start"
        inst_font_scale = 0.8
        inst_text_size = cv2.getTextSize(instruction, font, inst_font_scale, 2)[0]
        inst_x = (width - inst_text_size[0]) // 2
        inst_y = height - 50
        
        # Create pulsing effect
        for opacity in np.linspace(0, 1, 3):
            startup_copy = startup.copy()
            cv2.putText(startup_copy, instruction, (inst_x, inst_y), font, 
                       inst_font_scale, (int(255*opacity), int(255*opacity), int(255)), 2)
            
            # Display with delay for pulsing effect
            cv2.imshow(self.window_title, startup_copy)
            cv2.waitKey(500)
        
        return startup
    
    def run(self):
        """Main application loop."""
        if self.canvas is None or self.canvas.size == 0:
            print("Could not initialize webcam. Exiting...")
            return
        
        # Make sure canvas has right number of channels
        if len(self.canvas.shape) < 3 or self.canvas.shape[2] != 4:
            h, w = self.canvas.shape[:2] if len(self.canvas.shape) > 1 else (480, 640)
            self.canvas = np.zeros((h, w, 4), dtype=np.uint8)
            print("Reinitializing canvas with proper format")
        
        # Init UI flags
        self.show_help = False
        
        # Create window with proper properties
        cv2.namedWindow(self.window_title, cv2.WINDOW_NORMAL)
        
        # Set fullscreen mode if configured
        if self.fullscreen:
            cv2.setWindowProperty(self.window_title, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        
        # Show startup screen
        success, frame = self.cap.read()
        if success:
            width, height = frame.shape[1], frame.shape[0]
            startup_screen = self.create_startup_screen(width, height)
            
            # Wait for any key to dismiss startup screen
            key = cv2.waitKey(0) & 0xFF
            
            # If ESC was pressed during startup, exit
            if key == 27:
                self.cap.release()
                cv2.destroyAllWindows()
                self.hands.close()
                return
            
            # Show help after startup
            self.show_help = True
        
        while True:
            success, frame = self.cap.read()
            if not success:
                print("Failed to get frame from webcam")
                break
            
            # Flip the frame horizontally for a more natural interaction
            frame = cv2.flip(frame, 1)
            
            # If help is enabled, show the help screen
            if self.show_help:
                help_screen = self.render_help_screen(frame)
                cv2.imshow(self.window_title, help_screen)
                
                # Wait for any key to dismiss help
                key = cv2.waitKey(0) & 0xFF
                self.show_help = False
                
                # If ESC was pressed during help, exit
                if key == 27:
                    break
                continue
            
            # Process the frame normally
            results = self.detect_hand_landmarks(frame)
            output_frame = self.process_hands(frame, results)
            
            # Draw UI elements on top
            self.draw_ui(output_frame)
            
            # Display the result
            cv2.imshow(self.window_title, output_frame)
            
            # Handle keyboard inputs
            key = cv2.waitKey(1) & 0xFF
            if not self.handle_keyboard_input(key):
                break
        
        # Release resources
        self.cap.release()
        cv2.destroyAllWindows()
        self.hands.close()

if __name__ == "__main__":
    app = FingerDrawingApp()
    app.run()
