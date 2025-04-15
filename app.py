import cv2
import numpy as np
import mediapipe as mp
import math

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
        
        # Drawing parameters
        self.canvas = None
        self.prev_finger_pos = None
        self.drawing_color = (0, 0, 255)  # Red by default
        self.brush_thickness = 5
        self.is_drawing = False
        self.is_erasing = False
        
        # Initialize webcam
        self.cap = cv2.VideoCapture(0)
        success, frame = self.cap.read()
        if success:
            h, w = frame.shape[:2]
            self.canvas = np.zeros((h, w, 4), dtype=np.uint8)
        else:
            # Default canvas size if webcam fails
            self.canvas = np.zeros((480, 640, 4), dtype=np.uint8)
            print("Warning: Could not initialize webcam. Using default canvas size.")
        
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
        # Check if all fingers are extended (open palm)
        fingers_extended = [
            self.is_finger_extended(landmarks, self.mp_hands.HandLandmark.THUMB_TIP, self.mp_hands.HandLandmark.THUMB_IP),
            self.is_finger_extended(landmarks, self.mp_hands.HandLandmark.INDEX_FINGER_TIP, self.mp_hands.HandLandmark.INDEX_FINGER_PIP),
            self.is_finger_extended(landmarks, self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP, self.mp_hands.HandLandmark.MIDDLE_FINGER_PIP),
            self.is_finger_extended(landmarks, self.mp_hands.HandLandmark.RING_FINGER_TIP, self.mp_hands.HandLandmark.RING_FINGER_PIP),
            self.is_finger_extended(landmarks, self.mp_hands.HandLandmark.PINKY_TIP, self.mp_hands.HandLandmark.PINKY_PIP)
        ]
        
        # Count extended fingers
        extended_count = sum(fingers_extended)
        
        # Palm is showing if at least 4 fingers are extended
        return extended_count >= 4
    
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
                
                # Check if palm is showing (for erasing)
                if self.is_palm_showing(hand_landmarks):
                    self.is_drawing = False
                    self.is_erasing = True
                    # Create a circular eraser
                    eraser_radius = 30
                    cv2.circle(self.canvas, (x, y), eraser_radius, (0, 0, 0, 0), -1)
                else:
                    # Check if index finger is extended for drawing
                    if self.is_finger_extended(hand_landmarks, self.mp_hands.HandLandmark.INDEX_FINGER_TIP, 
                                              self.mp_hands.HandLandmark.INDEX_FINGER_PIP):
                        self.is_drawing = True
                        self.is_erasing = False
                        
                        if self.prev_finger_pos is not None:
                            # Draw line between previous and current position
                            cv2.line(self.canvas, self.prev_finger_pos, (x, y), 
                                    (*self.drawing_color, 255), self.brush_thickness)
                        
                        self.prev_finger_pos = (x, y)
                    else:
                        self.is_drawing = False
                        self.prev_finger_pos = None
        else:
            self.is_drawing = False
            self.is_erasing = False
            self.prev_finger_pos = None
        
        # Add status text to the frame
        if self.is_drawing:
            cv2.putText(frame_with_drawing, "Drawing", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        elif self.is_erasing:
            cv2.putText(frame_with_drawing, "Erasing", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        return frame_with_drawing
    
    def run(self):
        if self.canvas is None:
            print("Could not initialize webcam. Exiting...")
            return
        
        # Make sure canvas is properly initialized with alpha channel
        if self.canvas.shape[2] != 4:
            h, w = self.canvas.shape[:2] if len(self.canvas.shape) > 2 else (480, 640)
            self.canvas = np.zeros((h, w, 4), dtype=np.uint8)
        
        while True:
            success, frame = self.cap.read()
            if not success:
                print("Failed to get frame from webcam")
                break
            
            # Flip the frame horizontally for a more natural interaction
            frame = cv2.flip(frame, 1)
            
            # Detect hand landmarks
            results = self.detect_hand_landmarks(frame)
            
            # Process hands and update canvas
            output_frame = self.process_hands(frame, results)
            
            # Display the result
            cv2.imshow('Finger Drawing App', output_frame)
            
            # Handle keyboard inputs
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC key to exit
                break
            elif key == ord('c'):  # 'c' key to clear canvas
                self.canvas = np.zeros_like(self.canvas)
            elif key == ord('r'):  # 'r' key to set color to red
                self.drawing_color = (0, 0, 255)
            elif key == ord('g'):  # 'g' key to set color to green
                self.drawing_color = (0, 255, 0)
            elif key == ord('b'):  # 'b' key to set color to blue
                self.drawing_color = (255, 0, 0)
            elif key == ord('+'):  # '+' key to increase brush thickness
                self.brush_thickness = min(30, self.brush_thickness + 1)
            elif key == ord('-'):  # '-' key to decrease brush thickness
                self.brush_thickness = max(1, self.brush_thickness - 1)
        
        # Release resources
        self.cap.release()
        cv2.destroyAllWindows()
        self.hands.close()

if __name__ == "__main__":
    app = FingerDrawingApp()
    app.run()
