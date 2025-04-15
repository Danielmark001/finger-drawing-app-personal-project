import cv2
import numpy as np
import mediapipe as mp
import time
import math
from gesture_utils import (
    detect_pinch_gesture, 
    create_color_picker, 
    get_color_from_position,
    draw_brush_preview,
    calculate_distance
)

class EnhancedFingerDrawingApp:
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
        
        # Drawing parameters
        self.canvas = None
        self.prev_finger_pos = None
        self.drawing_color = (0, 0, 255)  # Red by default
        self.brush_thickness = 5
        self.is_drawing = False
        self.is_erasing = False
        
        # UI elements
        self.show_color_picker = False
        self.color_picker = None
        self.color_picker_pos = (20, 100)  # (x, y) position of the color picker
        self.color_picker_size = (200, 30)  # (width, height) of the color picker
        
        # Mode control
        self.current_mode = "draw"  # Options: "draw", "erase", "color_pick"
        
        # Initialize webcam
        self.cap = cv2.VideoCapture(0)
        
        # Try multiple times to get a frame
        max_attempts = 5
        for attempt in range(max_attempts):
            success, frame = self.cap.read()
            if success and frame is not None and frame.size > 0:
                print(f"Successfully initialized webcam on attempt {attempt+1}")
                h, w, _ = frame.shape
                self.canvas = np.zeros((h, w, 4), dtype=np.uint8)
                self.color_picker = create_color_picker(self.color_picker_size[0], self.color_picker_size[1])
                break
            elif attempt < max_attempts - 1:
                print(f"Attempt {attempt+1} failed, retrying...")
                time.sleep(0.5)  # Wait before trying again
                self.cap.release()
                self.cap = cv2.VideoCapture(0)
            else:
                print(f"Failed to initialize webcam after {max_attempts} attempts")
                # Set default sizes
                self.canvas = np.zeros((480, 640, 4), dtype=np.uint8)
                self.color_picker = create_color_picker(self.color_picker_size[0], self.color_picker_size[1])
        
    def detect_hand_landmarks(self, frame):
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame with mediapipe
        results = self.hands.process(rgb_frame)
        
        return results
    
    def is_finger_extended(self, landmarks, finger_tip_idx, finger_pip_idx):
        tip = landmarks.landmark[finger_tip_idx]
        pip = landmarks.landmark[finger_pip_idx]
        
        # Check if tip is higher than pip (y is smaller)
        return tip.y < pip.y
    
    def is_palm_showing(self, landmarks):
        """
        Improved palm detection that considers both finger extension and palm orientation.
        """
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
        # We'll use the relationship between wrist, middle_mcp, and middle_tip to determine orientation
        wrist = landmarks.landmark[self.mp_hands.HandLandmark.WRIST]
        middle_mcp = landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
        middle_tip = landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
        
        # If palm is facing the camera, the middle_tip's z should be significantly less than the middle_mcp's z
        # (meaning the fingertip is closer to the camera than the knuckle)
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
        
    def calculate_palm_width(self, landmarks):
        """
        Calculate the width of the palm to determine distance from camera.
        Wider palm in the image = closer to camera.
        """
        # Get pixel coordinates for the sides of the palm
        h, w, _ = self.canvas.shape
        
        # Use the distance between the pinky_mcp and thumb_mcp as palm width
        pinky_mcp = landmarks.landmark[self.mp_hands.HandLandmark.PINKY_MCP]
        thumb_mcp = landmarks.landmark[self.mp_hands.HandLandmark.THUMB_MCP]
        
        pinky_x, pinky_y = int(pinky_mcp.x * w), int(pinky_mcp.y * h)
        thumb_x, thumb_y = int(thumb_mcp.x * w), int(thumb_mcp.y * h)
        
        # Calculate Euclidean distance
        palm_width = math.sqrt((pinky_x - thumb_x)**2 + (pinky_y - thumb_y)**2)
        
        return palm_width
    
    def process_hands(self, frame, results):
        h, w, _ = frame.shape
        frame_with_drawing = frame.copy()
        
        # Ensure canvas has proper dimensions and channels
        if self.canvas is None or self.canvas.shape[0] != h or self.canvas.shape[1] != w:
            self.canvas = np.zeros((h, w, 4), dtype=np.uint8)
        
        # Overlay the canvas onto the frame
        if self.canvas.shape[2] == 4:  # If canvas has alpha channel
            mask = self.canvas[:, :, 3] > 0
            if np.any(mask):  # Only blend if there's something to blend
                frame_with_drawing[mask] = cv2.addWeighted(frame_with_drawing[mask], 0.5, self.canvas[:, :, :3][mask], 0.5, 0)
        
        # Draw mode UI
        self.draw_ui(frame_with_drawing)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw hand landmarks for visualization
                self.mp_drawing.draw_landmarks(
                    frame_with_drawing, 
                    hand_landmarks, 
                    self.mp_hands.HAND_CONNECTIONS
                )
                
                # Get index finger tip position for drawing
                index_finger_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
                x, y = int(index_finger_tip.x * w), int(index_finger_tip.y * h)
                
                # Check for pinch gesture (to pick color)
                is_pinching, pinch_distance, pinch_pos = detect_pinch_gesture(
                    hand_landmarks, self.mp_hands, frame.shape
                )
                
                # Handle different modes based on gestures
                if is_pinching and self.show_color_picker:
                    # Check if pinching in color picker area
                    color_picker_x_range = range(self.color_picker_pos[0], self.color_picker_pos[0] + self.color_picker_size[0])
                    color_picker_y_range = range(self.color_picker_pos[1], self.color_picker_pos[1] + self.color_picker_size[1])
                    
                    if pinch_pos[0] in color_picker_x_range and pinch_pos[1] in color_picker_y_range:
                        # Get relative position in color picker
                        rel_x = pinch_pos[0] - self.color_picker_pos[0]
                        rel_y = pinch_pos[1] - self.color_picker_pos[1]
                        
                        # Update drawing color
                        self.drawing_color = get_color_from_position(self.color_picker, (rel_x, rel_y))
                        self.current_mode = "draw"
                        self.show_color_picker = False
                    
                # Check for palm (eraser)
                elif self.is_palm_showing(hand_landmarks):
                    self.is_drawing = False
                    self.is_erasing = True
                    self.current_mode = "erase"
                    
                    # Calculate eraser size based on palm distance from camera
                    # We use the size of the hand in the image as a proxy for distance
                    palm_width = self.calculate_palm_width(hand_landmarks)
                    
                    # Map the palm width to an eraser size (larger palm width = closer to camera = larger eraser)
                    # Min eraser size: 15, Max eraser size: 80
                    min_palm_width = 30  # Approximate minimum palm width in pixels
                    max_palm_width = 150  # Approximate maximum palm width in pixels
                    min_eraser = 15
                    max_eraser = 80
                    
                    # Normalize the palm width between 0 and 1
                    normalized_width = max(0, min(1, (palm_width - min_palm_width) / (max_palm_width - min_palm_width)))
                    
                    # Map to eraser size with easing function for smoother transitions
                    # Using a quadratic easing function: f(x) = x^2
                    eased_value = normalized_width * normalized_width
                    eraser_radius = int(min_eraser + eased_value * (max_eraser - min_eraser))
                    
                    # Create a circular eraser
                    cv2.circle(self.canvas, (x, y), eraser_radius, (0, 0, 0, 0), -1)
                    
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
                
                # Check for index finger extended (drawing)
                elif self.is_finger_extended(hand_landmarks, self.mp_hands.HandLandmark.INDEX_FINGER_TIP, 
                                          self.mp_hands.HandLandmark.INDEX_FINGER_PIP):
                    if self.current_mode == "draw":
                        self.is_drawing = True
                        self.is_erasing = False
                        
                        # Draw brush preview
                        draw_brush_preview(frame_with_drawing, (x, y), self.drawing_color, self.brush_thickness)
                        
                        if self.prev_finger_pos is not None:
                            # Draw line between previous and current position
                            cv2.line(self.canvas, self.prev_finger_pos, (x, y), 
                                    (*self.drawing_color, 255), self.brush_thickness)
                        
                        self.prev_finger_pos = (x, y)
                    
                # Gesture for showing color picker: Index + Middle finger extended
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
                else:
                    self.is_drawing = False
                    self.prev_finger_pos = None
        else:
            self.is_drawing = False
            self.is_erasing = False
            self.prev_finger_pos = None
        
        return frame_with_drawing
    
    def draw_ui(self, frame):
        h, w, _ = frame.shape
        
        # Create a semi-transparent overlay for UI elements
        ui_overlay = frame.copy()
        
        # Draw a dark semi-transparent header area
        cv2.rectangle(ui_overlay, (0, 0), (w, 80), (30, 30, 30), -1)
        
        # Draw a dark semi-transparent footer area for instructions
        cv2.rectangle(ui_overlay, (0, h-100), (w, h), (30, 30, 30), -1)
        
        # Blend the overlay with the original frame
        alpha = 0.7
        frame = cv2.addWeighted(ui_overlay, alpha, frame, 1 - alpha, 0)
        
        # Draw color picker if enabled
        if self.show_color_picker:
            cp_x, cp_y = self.color_picker_pos
            cp_w, cp_h = self.color_picker_size
            
            # Draw color picker with improved visuals
            frame[cp_y:cp_y+cp_h, cp_x:cp_x+cp_w] = self.color_picker
            
            # Draw border with drop shadow effect
            cv2.rectangle(frame, (cp_x-3, cp_y-3), (cp_x+cp_w+3, cp_y+cp_h+3), (40, 40, 40), 4)
            cv2.rectangle(frame, (cp_x-2, cp_y-2), (cp_x+cp_w+2, cp_y+cp_h+2), (255, 255, 255), 2)
            
            # Add instruction for color picker
            cv2.putText(frame, "Pinch to select a color", (cp_x, cp_y+cp_h+20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Add status text and icons to the frame
        mode_x = 20
        
        if self.current_mode == "draw":
            # Draw mode icon and text
            cv2.circle(frame, (mode_x, 30), 10, self.drawing_color, -1)
            cv2.putText(frame, "Drawing Mode", (mode_x + 20, 35), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # Show color swatch with label
            cv2.rectangle(frame, (mode_x + 200, 20), (mode_x + 240, 40), self.drawing_color, -1)
            cv2.rectangle(frame, (mode_x + 200, 20), (mode_x + 240, 40), (255, 255, 255), 1)
            cv2.putText(frame, "Color:", (mode_x + 150, 35), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
        elif self.current_mode == "erase":
            # Eraser mode icon and text
            cv2.circle(frame, (mode_x, 30), 15, (200, 200, 200), 1)
            cv2.circle(frame, (mode_x, 30), 5, (200, 200, 200), -1)
            cv2.putText(frame, "Erasing Mode", (mode_x + 20, 35), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(frame, "Move palm closer/farther to resize eraser", 
                       (mode_x + 220, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            
        elif self.current_mode == "color_pick":
            # Color pick mode icon and text
            cv2.rectangle(frame, (mode_x - 5, 25), (mode_x + 5, 35), (255, 0, 0), -1)
            cv2.rectangle(frame, (mode_x + 8, 25), (mode_x + 18, 35), (0, 255, 0), -1)
            cv2.rectangle(frame, (mode_x - 5, 38), (mode_x + 18, 48), (0, 0, 255), -1)
            cv2.putText(frame, "Color Selection Mode", (mode_x + 25, 35), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Show brush thickness with a visual indicator
        thickness_x = mode_x
        thickness_y = 60
        cv2.putText(frame, f"Brush: {self.brush_thickness}px", (thickness_x, thickness_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Draw a brush preview
        cv2.circle(frame, (thickness_x + 140, thickness_y - 5), self.brush_thickness // 2, 
                  self.drawing_color, -1)
        
        # Add instructions in the footer
        instructions = [
            "GESTURES: Index finger = Draw | Open palm = Erase | Peace sign = Color picker",
            "KEYS: [r,g,b,w,k] = Colors | [+/-] = Brush size | [c] = Clear | [s] = Save | [ESC] = Exit"
        ]
        
        for i, instruction in enumerate(instructions):
            y_pos = h - 70 + i * 25
            cv2.putText(frame, instruction, (20, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
        # Add a small help icon with tooltip
        help_x, help_y = w - 50, 30
        cv2.circle(frame, (help_x, help_y), 15, (100, 100, 255), -1)
        cv2.putText(frame, "?", (help_x - 5, help_y + 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1)
    
    def show_help_overlay(self, frame):
        """Display a help screen with detailed instructions"""
        h, w, _ = frame.shape
        
        # Create a semi-transparent overlay
        help_overlay = np.zeros_like(frame)
        
        # Draw a dark background
        cv2.rectangle(help_overlay, (0, 0), (w, h), (30, 30, 50), -1)
        
        # Create a light panel in the center
        panel_margin = 50
        cv2.rectangle(help_overlay, (panel_margin, panel_margin), 
                     (w - panel_margin, h - panel_margin), (50, 50, 70), -1)
        cv2.rectangle(help_overlay, (panel_margin + 2, panel_margin + 2), 
                     (w - panel_margin - 2, h - panel_margin - 2), (70, 70, 90), 2)
        
        # Title
        cv2.putText(help_overlay, "Finger Drawing App - Help", (w//2 - 180, panel_margin + 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Instructions
        instructions = [
            "HAND GESTURES:",
            " - Extend only index finger to draw",
            " - Show open palm (all fingers) to erase",
            " - Bring palm closer to create larger eraser, further for smaller",
            " - Make peace sign (index+middle finger) to show color picker",
            " - Pinch (thumb+index) to select color from picker",
            "",
            "KEYBOARD CONTROLS:",
            " - [r] Change to red color",
            " - [g] Change to green color",
            " - [b] Change to blue color",
            " - [w] Change to white color",
            " - [k] Change to black color",
            " - [p] Toggle color picker",
            " - [+/=] Increase brush thickness",
            " - [-/_] Decrease brush thickness",
            " - [c] Clear the canvas",
            " - [s] Save your drawing",
            " - [h] Show/hide this help screen",
            " - [ESC] Exit application"
        ]
        
        y_offset = panel_margin + 80
        line_height = 22
        
        for i, line in enumerate(instructions):
            indent = 30 if line.startswith(" -") else 0
            cv2.putText(help_overlay, line, (panel_margin + 20 + indent, y_offset + i * line_height), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Press any key to continue
        cv2.putText(help_overlay, "Press any key to continue", (w//2 - 120, h - panel_margin - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 255), 1)
        
        # Blend with the original frame
        result = cv2.addWeighted(frame, 0.1, help_overlay, 0.9, 0)
        
        return result
        
    def run(self):
        if self.canvas is None or self.canvas.size == 0:
            print("Could not initialize webcam. Exiting...")
            return
        
        # Ensure canvas is properly initialized with 4 channels
        if len(self.canvas.shape) < 3 or self.canvas.shape[2] != 4:
            h, w = self.canvas.shape[:2] if len(self.canvas.shape) > 1 else (480, 640)
            self.canvas = np.zeros((h, w, 4), dtype=np.uint8)
            print("Reinitializing canvas with proper format")
            
        # Initialize UI flags
        show_help = False  # Don't show help immediately
        show_startup = True  # Show startup screen first
        
        # Show startup screen if enabled
        if show_startup:
            from gesture_utils import create_startup_screen
            
            # Get frame dimensions
            if self.canvas is not None and self.canvas.shape[0] > 0:
                height, width = self.canvas.shape[:2]
            else:
                # Default dimensions if canvas isn't initialized
                width, height = 640, 480
                
            startup_screen = create_startup_screen(width, height)
            cv2.imshow('Finger Drawing App', startup_screen)
            
            # Wait for any key to dismiss startup screen
            key = cv2.waitKey(0) & 0xFF
            show_startup = False
            
            # If ESC was pressed during startup, exit
            if key == 27:
                self.cap.release()
                cv2.destroyAllWindows()
                self.hands.close()
                return
            
            # Show help after startup
            show_help = True
        
        while True:
            success, frame = self.cap.read()
            if not success:
                print("Failed to get frame from webcam")
                break
            
            # Flip the frame horizontally for a more natural interaction
            frame = cv2.flip(frame, 1)
            
            # Process the frame normally
            results = self.detect_hand_landmarks(frame)
            output_frame = self.process_hands(frame, results)
            
            # Add gesture guides if hands are detected
            if results.multi_hand_landmarks:
                from gesture_utils import draw_gesture_guide
                for hand_landmarks in results.multi_hand_landmarks:
                    output_frame = draw_gesture_guide(
                        output_frame, hand_landmarks, self.mp_hands, 
                        self.current_mode, frame.shape
                    )
            
            # If help is enabled, show the help overlay
            if show_help:
                output_frame = self.show_help_overlay(output_frame)
                cv2.imshow('Finger Drawing App', output_frame)
                
                # Wait for any key to dismiss help
                key = cv2.waitKey(0) & 0xFF
                show_help = False
                
                # If ESC was pressed during help, exit
                if key == 27:
                    break
                continue
            
            # Display the result
            cv2.imshow('Finger Drawing App', output_frame)
            
            # Handle keyboard inputs
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC key to exit
                break
            elif key == ord('h'):  # 'h' key to show help
                show_help = True
            elif key == ord('c'):  # 'c' key to clear canvas
                # Add visual feedback when clearing
                clear_overlay = frame.copy()
                cv2.putText(clear_overlay, "Canvas Cleared!", (frame.shape[1]//2 - 100, frame.shape[0]//2), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.imshow('Finger Drawing App', clear_overlay)
                cv2.waitKey(300)  # Brief delay to show the message
                
                self.canvas = np.zeros_like(self.canvas)
            elif key == ord('r'):  # 'r' key to set color to red
                self.drawing_color = (0, 0, 255)
                self.current_mode = "draw"
            elif key == ord('g'):  # 'g' key to set color to green
                self.drawing_color = (0, 255, 0)
                self.current_mode = "draw"
            elif key == ord('b'):  # 'b' key to set color to blue
                self.drawing_color = (255, 0, 0)
                self.current_mode = "draw"
            elif key == ord('w'):  # 'w' key to set color to white
                self.drawing_color = (255, 255, 255)
                self.current_mode = "draw"
            elif key == ord('k'):  # 'k' key to set color to black
                self.drawing_color = (0, 0, 0)
                self.current_mode = "draw"
            elif key == ord('p'):  # 'p' key to toggle color picker
                self.show_color_picker = not self.show_color_picker
                if self.show_color_picker:
                    self.current_mode = "color_pick"
                else:
                    self.current_mode = "draw"
            elif key == ord('+') or key == ord('='):  # '+' key to increase brush thickness
                old_thickness = self.brush_thickness
                self.brush_thickness = min(30, self.brush_thickness + 1)
                
                # Show feedback if thickness changed
                if self.brush_thickness != old_thickness:
                    thick_overlay = frame.copy()
                    cv2.putText(thick_overlay, f"Brush: {self.brush_thickness}px", (frame.shape[1]//2 - 80, frame.shape[0]//2), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    cv2.circle(thick_overlay, (frame.shape[1]//2 + 100, frame.shape[0]//2 - 5), 
                              self.brush_thickness//2, self.drawing_color, -1)
                    cv2.imshow('Finger Drawing App', thick_overlay)
                    cv2.waitKey(200)  # Brief delay to show the message
                
            elif key == ord('-') or key == ord('_'):  # '-' key to decrease brush thickness
                old_thickness = self.brush_thickness
                self.brush_thickness = max(1, self.brush_thickness - 1)
                
                # Show feedback if thickness changed
                if self.brush_thickness != old_thickness:
                    thick_overlay = frame.copy()
                    cv2.putText(thick_overlay, f"Brush: {self.brush_thickness}px", (frame.shape[1]//2 - 80, frame.shape[0]//2), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    cv2.circle(thick_overlay, (frame.shape[1]//2 + 100, frame.shape[0]//2 - 5), 
                              self.brush_thickness//2, self.drawing_color, -1)
                    cv2.imshow('Finger Drawing App', thick_overlay)
                    cv2.waitKey(200)  # Brief delay to show the message
                
            elif key == ord('s'):  # 's' key to save canvas
                timestamp = cv2.getTickCount()
                filename = f"drawing_{timestamp}.png"
                
                # Save the canvas
                cv2.imwrite(filename, cv2.cvtColor(self.canvas, cv2.COLOR_BGRA2RGBA))
                
                # Show save confirmation
                save_overlay = frame.copy()
                cv2.putText(save_overlay, f"Saved as {filename}", (frame.shape[1]//2 - 150, frame.shape[0]//2), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow('Finger Drawing App', save_overlay)
                cv2.waitKey(1000)  # Longer delay for save confirmation
                
                print(f"Saved drawing as {filename}")
        
        # Release resources
        self.cap.release()
        cv2.destroyAllWindows()
        self.hands.close()

if __name__ == "__main__":
    app = EnhancedFingerDrawingApp()
    app.run()
