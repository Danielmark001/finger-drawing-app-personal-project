import cv2
import numpy as np
import time
import os

class Tutorial:
    def __init__(self, width=1024, height=768):
        self.width = width
        self.height = height
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Create window
        self.window_name = "Finger Drawing Tutorial"
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, width, height)
    
    def create_base_frame(self, title):
        """Create a base frame with title."""
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        frame[:, :] = (30, 30, 40)  # Dark blue-gray background
        
        # Add header bar
        cv2.rectangle(frame, (0, 0), (self.width, 80), (50, 50, 70), -1)
        
        # Add title
        cv2.putText(frame, title, (30, 50), self.font, 1.2, (255, 255, 255), 2)
        
        # Add footer bar
        cv2.rectangle(frame, (0, self.height-60), (self.width, self.height), (50, 50, 70), -1)
        
        # Add navigation hint
        nav_text = "Press SPACEBAR to continue | ESC to exit"
        text_size = cv2.getTextSize(nav_text, self.font, 0.7, 1)[0]
        cv2.putText(frame, nav_text, (self.width - text_size[0] - 20, self.height - 25), 
                   self.font, 0.7, (200, 200, 200), 1)
        
        return frame
    
    def show_frame(self, frame, wait_key=True):
        """Show a frame and wait for key press if needed."""
        cv2.imshow(self.window_name, frame)
        
        if wait_key:
            while True:
                key = cv2.waitKey(100) & 0xFF
                if key == 27:  # ESC
                    return False  # Exit
                elif key == 32:  # Space
                    return True  # Continue
        else:
            cv2.waitKey(1)
            return True
    
    def draw_hand_gesture(self, frame, gesture_type, position):
        """Draw a hand gesture illustration."""
        x, y = position
        size = 120  # Hand size
        
        # Base hand shape - palm
        cv2.circle(frame, (x, y), size//2, (100, 100, 150), -1)
        cv2.circle(frame, (x, y), size//2, (150, 150, 200), 2)
        
        # Draw fingers based on gesture type
        finger_length = size * 0.8
        finger_width = size // 8
        
        if gesture_type == "draw":
            # Only index finger extended
            angles = [0.9, 0.6, 1.1, 1.3, 1.5]  # Radians
            lengths = [0.3, 1.0, 0.4, 0.3, 0.3]  # Relative to finger_length
            
        elif gesture_type == "erase":
            # All fingers extended (open palm)
            angles = [0.9, 0.7, 0.9, 1.1, 1.3]
            lengths = [0.7, 1.0, 0.9, 0.8, 0.7]
            
        elif gesture_type == "color_pick":
            # Peace sign (index and middle fingers)
            angles = [0.9, 0.6, 0.8, 1.3, 1.5]
            lengths = [0.3, 1.0, 1.0, 0.4, 0.3]
            
        elif gesture_type == "tool_select":
            # 3 fingers extended
            angles = [0.9, 0.6, 0.8, 1.0, 1.5]
            lengths = [0.3, 1.0, 1.0, 1.0, 0.3]
            
        elif gesture_type == "pinch":
            # Pinch gesture (index and thumb touching)
            angles = [0.5, 0.6, 1.1, 1.3, 1.5]
            lengths = [0.8, 0.8, 0.4, 0.3, 0.3]
            # Draw pinch circle
            thumb_tip_x = int(x + finger_length * lengths[0] * np.cos(angles[0]))
            thumb_tip_y = int(y - finger_length * lengths[0] * np.sin(angles[0]))
            index_tip_x = int(x + finger_length * lengths[1] * np.cos(angles[1]))
            index_tip_y = int(y - finger_length * lengths[1] * np.sin(angles[1]))
            # Draw circle at pinch location
            pinch_x = (thumb_tip_x + index_tip_x) // 2
            pinch_y = (thumb_tip_y + index_tip_y) // 2
            cv2.circle(frame, (pinch_x, pinch_y), 10, (255, 200, 200), -1)
            
        else:  # Default - relaxed hand
            angles = [0.9, 0.7, 0.9, 1.1, 1.3]
            lengths = [0.6, 0.7, 0.6, 0.5, 0.4]
        
        # Draw fingers
        for i in range(5):
            angle = angles[i]
            length = lengths[i] * finger_length
            
            end_x = int(x + length * np.cos(angle))
            end_y = int(y - length * np.sin(angle))
            
            # Draw finger
            cv2.line(frame, (x, y), (end_x, end_y), (100, 100, 150), finger_width)
            cv2.circle(frame, (end_x, end_y), finger_width // 2, (100, 100, 150), -1)
            
            # Draw outline
            cv2.line(frame, (x, y), (end_x, end_y), (150, 150, 200), 2)
            cv2.circle(frame, (end_x, end_y), finger_width // 2, (150, 150, 200), 2)
        
        # Add gesture label
        label_y = y + size // 2 + 40
        cv2.putText(frame, gesture_type.upper(), (x - 40, label_y), self.font, 0.7, (200, 200, 255), 2)
    
    def run_tutorial(self):
        """Run the complete tutorial."""
        # Welcome screen
        welcome = self.create_base_frame("Welcome to Finger Drawing App")
        welcome_text = [
            "This tutorial will guide you through the basic gestures",
            "and features of the Finger Drawing App.",
            "",
            "You'll learn how to:",
            "• Draw and erase with hand gestures",
            "• Select colors and tools",
            "• Use keyboard shortcuts",
            "• Work with layers and effects",
            "",
            "Make sure your webcam is positioned so your hand is clearly visible.",
        ]
        
        for i, line in enumerate(welcome_text):
            y = 140 + i * 30
            cv2.putText(welcome, line, (100, y), self.font, 0.7, (220, 220, 220), 1)
        
        if not self.show_frame(welcome):
            return
        
        # Basic drawing gesture
        drawing = self.create_base_frame("Drawing Gesture")
        self.draw_hand_gesture(drawing, "draw", (300, 300))
        
        draw_text = [
            "To draw, extend only your index finger",
            "and move it around in front of the camera.",
            "",
            "Keep your other fingers curled down.",
            "",
            "The tip of your index finger controls",
            "where you draw on the screen."
        ]
        
        for i, line in enumerate(draw_text):
            y = 180 + i * 30
            cv2.putText(drawing, line, (450, y), self.font, 0.7, (220, 220, 220), 1)
        
        if not self.show_frame(drawing):
            return
        
        # Erasing gesture
        erasing = self.create_base_frame("Erasing Gesture")
        self.draw_hand_gesture(erasing, "erase", (300, 300))
        
        erase_text = [
            "To erase, show your open palm to the camera",
            "with all fingers extended.",
            "",
            "The eraser follows the center of your palm.",
            "",
            "TIP: Move your palm closer to the camera",
            "for a larger eraser, or further away",
            "for a smaller, more precise eraser."
        ]
        
        for i, line in enumerate(erase_text):
            y = 180 + i * 30
            cv2.putText(erasing, line, (450, y), self.font, 0.7, (220, 220, 220), 1)
        
        if not self.show_frame(erasing):
            return
        
        # Color picker gesture
        color_pick = self.create_base_frame("Color Selection Gesture")
        self.draw_hand_gesture(color_pick, "color_pick", (300, 300))
        
        color_text = [
            "To open the color picker, use a peace sign",
            "(extend your index and middle fingers).",
            "",
            "Once the color picker appears, make a pinch",
            "gesture (touch thumb and index finger together)",
            "over your desired color to select it.",
            "",
            "You can also press R, G, B, W, or K keys",
            "to quickly select red, green, blue, white,",
            "or black colors."
        ]
        
        for i, line in enumerate(color_text):
            y = 180 + i * 30
            cv2.putText(color_pick, line, (450, y), self.font, 0.7, (220, 220, 220), 1)
        
        if not self.show_frame(color_pick):
            return
        
        # Tool selection
        tools = self.create_base_frame("Tool Selection")
        self.draw_hand_gesture(tools, "tool_select", (300, 300))
        
        tools_text = [
            "To open the tools panel, extend three fingers",
            "(index, middle, and ring) while keeping your",
            "pinky and thumb curled.",
            "",
            "Once the toolbar appears, make an OK sign",
            "(circle with thumb and index finger)",
            "over a tool to select it.",
            "",
            "Available tools: Brush, Eraser, Line,",
            "Rectangle, Circle, and Fill"
        ]
        
        for i, line in enumerate(tools_text):
            y = 180 + i * 30
            cv2.putText(tools, line, (450, y), self.font, 0.7, (220, 220, 220), 1)
        
        if not self.show_frame(tools):
            return
        
        # Keyboard shortcuts
        shortcuts = self.create_base_frame("Keyboard Shortcuts")
        
        shortcuts_text = [
            "COLORS:",
            "  R - Red",
            "  G - Green", 
            "  B - Blue",
            "  W - White",
            "  K - Black",
            "",
            "BRUSH STYLES (1-4):",
            "  1 - Solid",
            "  2 - Airbrush",
            "  3 - Marker",
            "  4 - Pencil",
            "",
            "TOOLS & ACTIONS:",
            "  +/- - Adjust brush size",
            "  C - Clear canvas",
            "  S - Save drawing",
            "  Z - Undo",
            "  Y - Redo",
            "  L - Toggle layers panel",
            "  H - Show help screen",
            "  ESC - Exit"
        ]
        
        # Split in two columns
        col_width = 250
        col1_x, col2_x = 180, 500
        
        for i, line in enumerate(shortcuts_text):
            # Calculate column and position
            if i < 13:  # First column
                x, y = col1_x, 180 + (i % 13) * 30
            else:  # Second column
                x, y = col2_x, 180 + (i - 13) * 30
                
            # Highlight keys in a different color
            if " - " in line:
                parts = line.split(" - ", 1)
                key_part = parts[0]
                desc_part = " - " + parts[1] if len(parts) > 1 else ""
                
                # Draw key part in highlight color
                cv2.putText(shortcuts, key_part, (x, y), self.font, 0.7, (150, 150, 255), 1)
                
                # Calculate width of key part to position description
                key_width = cv2.getTextSize(key_part, self.font, 0.7, 1)[0][0]
                
                # Draw description part
                cv2.putText(shortcuts, desc_part, (x + key_width, y), self.font, 0.7, (220, 220, 220), 1)
            else:
                # Section header
                cv2.putText(shortcuts, line, (x, y), self.font, 0.7, (255, 200, 150), 1)
        
        if not self.show_frame(shortcuts):
            return
        
        # Final tips
        tips = self.create_base_frame("Tips & Tricks")
        
        tips_text = [
            "• For precise erasing, move your palm further from the camera",
            "",
            "• To complete drawing a shape, hold your finger still for a moment",
            "",
            "• Use layers for complex drawings - press L to show the layers panel",
            "",
            "• Save your work frequently with the S key",
            "",
            "• If hand detection is unstable, ensure good lighting and a",
            "  plain background",
            "",
            "• When using the fill tool, hold your finger still over the area",
            "  you want to fill",
            "",
            "• Press H anytime to show the help screen"
        ]
        
        for i, line in enumerate(tips_text):
            y = 150 + i * 30
            cv2.putText(tips, line, (150, y), self.font, 0.7, (220, 220, 220), 1)
        
        if not self.show_frame(tips):
            return
        
        # Conclusion
        conclusion = self.create_base_frame("Ready to Draw!")
        
        conclusion_text = [
            "You've completed the tutorial!",
            "",
            "Now you know all the basic gestures and features",
            "of the Finger Drawing App.",
            "",
            "Remember that practice makes perfect with gesture control.",
            "If you need a refresher on any gestures or features,",
            "press the H key anytime to bring up the help screen.",
            "",
            "Press SPACEBAR to close this tutorial and start drawing!"
        ]
        
        for i, line in enumerate(conclusion_text):
            y = 200 + i * 30
            cv2.putText(conclusion, line, (150, y), self.font, 0.7, (220, 220, 220), 1)
        
        self.show_frame(conclusion)
        
        # Close window
        cv2.destroyWindow(self.window_name)

def run_tutorial():
    """Run the tutorial."""
    tutorial = Tutorial()
    tutorial.run_tutorial()

if __name__ == "__main__":
    # Run tutorial directly if this script is executed
    run_tutorial()
