import cv2
import numpy as np
import mediapipe as mp
import math
import time

class SimulatedFingerDrawingApp:
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
        
        # Create a simulated video feed (a black canvas with moving dots)
        self.frame_width = 640
        self.frame_height = 480
        self.canvas = np.zeros((self.frame_height, self.frame_width, 4), dtype=np.uint8)
        
        # Simulated hand position
        self.simulated_index_finger_pos = [320, 240]
        self.simulated_hand_state = "drawing"  # "drawing" or "erasing"
        self.simulated_movement_radius = 100
        self.simulated_movement_speed = 0.05
        self.simulated_time = 0
        
    def generate_simulated_frame(self):
        # Create a blank frame
        frame = np.zeros((self.frame_height, self.frame_width, 3), dtype=np.uint8)
        
        # Add a gradient background
        for y in range(self.frame_height):
            for x in range(self.frame_width):
                frame[y, x] = [
                    int(x * 255 / self.frame_width), 
                    int((x + y) * 255 / (self.frame_width + self.frame_height)),
                    int(y * 255 / self.frame_height)
                ]
        
        # Update simulated finger position
        self.simulated_time += self.simulated_movement_speed
        
        # Change state every 5 seconds
        if int(self.simulated_time) % 10 < 5:
            self.simulated_hand_state = "drawing"
        else:
            self.simulated_hand_state = "erasing"
        
        # Move the finger in a figure-8 pattern
        center_x = self.frame_width // 2
        center_y = self.frame_height // 2
        
        self.simulated_index_finger_pos = [
            center_x + int(self.simulated_movement_radius * math.sin(self.simulated_time)),
            center_y + int(self.simulated_movement_radius * math.sin(self.simulated_time * 2) / 2)
        ]
        
        # Draw a simulated hand
        if self.simulated_hand_state == "drawing":
            # Draw a circle for index finger
            cv2.circle(frame, tuple(self.simulated_index_finger_pos), 5, (0, 255, 0), -1)
            cv2.circle(frame, (self.simulated_index_finger_pos[0] - 30, self.simulated_index_finger_pos[1] + 30), 3, (0, 200, 0), -1)  # thumb
            cv2.putText(frame, "Drawing Mode", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        else:
            # Draw an open palm
            palm_center = self.simulated_index_finger_pos
            cv2.circle(frame, tuple(palm_center), 20, (0, 0, 255), 2)  # Palm
            # Fingers
            for i in range(5):
                angle = math.pi / 2 + (i - 2) * math.pi / 6
                finger_end = [
                    palm_center[0] + int(40 * math.cos(angle)),
                    palm_center[1] - int(40 * math.sin(angle))
                ]
                cv2.line(frame, tuple(palm_center), tuple(finger_end), (0, 0, 255), 2)
            cv2.putText(frame, "Erasing Mode", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        return frame
    
    def process_simulated_hand(self, frame):
        h, w, _ = frame.shape
        frame_with_drawing = frame.copy()
        
        # Overlay the canvas onto the frame
        mask = self.canvas[:, :, 3] > 0
        frame_with_drawing[mask] = cv2.addWeighted(frame_with_drawing[mask], 0.5, self.canvas[:, :, :3][mask], 0.5, 0)
        
        x, y = self.simulated_index_finger_pos
        
        # Handle drawing or erasing based on simulated state
        if self.simulated_hand_state == "erasing":
            self.is_drawing = False
            self.is_erasing = True
            # Create a circular eraser
            eraser_radius = 30
            cv2.circle(self.canvas, (x, y), eraser_radius, (0, 0, 0, 0), -1)
            # Draw eraser preview
            cv2.circle(frame_with_drawing, (x, y), eraser_radius, (255, 255, 