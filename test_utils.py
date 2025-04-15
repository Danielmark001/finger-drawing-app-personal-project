import cv2
import numpy as np
import mediapipe as mp
import time

def test_webcam():
    """Test if webcam is working correctly."""
    print("Testing webcam connection...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("ERROR: Could not open webcam. Please check your camera connection.")
        return False
    
    # Read a frame
    ret, frame = cap.read()
    if not ret:
        print("ERROR: Could not read from webcam.")
        cap.release()
        return False
    
    # Display webcam feed for 5 seconds
    start_time = time.time()
    while time.time() - start_time < 5:
        ret, frame = cap.read()
        if not ret:
            break
        
        cv2.putText(frame, "Webcam Test - Press 'q' to quit", (20, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("Webcam Test", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("Webcam test completed successfully.")
    return True

def test_hand_detection():
    """Test if MediaPipe hand detection is working correctly."""
    print("Testing hand detection...")
    
    # Initialize MediaPipe Hands
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open webcam.")
        return False
    
    # Display hand detection for 15 seconds
    start_time = time.time()
    detection_count = 0
    frames_processed = 0
    
    while time.time() - start_time < 15:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Flip the frame horizontally for a more intuitive mirror view
        frame = cv2.flip(frame, 1)
        frames_processed += 1
        
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame with MediaPipe Hands
        results = hands.process(rgb_frame)
        
        # Draw hand landmarks
        if results.multi_hand_landmarks:
            detection_count += 1
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        
        # Display instructions and status
        cv2.putText(frame, "Hand Detection Test - Show your hand", (20, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        status = "Detected" if results.multi_hand_landmarks else "Not Detected"
        color = (0, 255, 0) if results.multi_hand_landmarks else (0, 0, 255)
        cv2.putText(frame, f"Status: {status}", (20, 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        cv2.putText(frame, f"Time left: {int(15 - (time.time() - start_time))}s", (20, 110), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        
        cv2.imshow("Hand Detection Test", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Calculate detection rate
    detection_rate = detection_count / frames_processed if frames_processed > 0 else 0
    
    cap.release()
    hands.close()
    cv2.destroyAllWindows()
    
    print(f"Hand detection test completed with {detection_rate*100:.1f}% detection rate.")
    
    if detection_rate < 0.1:
        print("WARNING: Poor hand detection rate. Make sure you showed your hand and you have adequate lighting.")
        return False
    
    return True

def test_drawing_canvas():
    """Test if drawing on a canvas works correctly."""
    print("Testing drawing canvas...")
    
    # Create a black canvas
    canvas = np.zeros((480, 640, 3), dtype=np.uint8)
    drawing = False
    last_x, last_y = -1, -1
    color = (0, 255, 0)  # Green
    
    def draw_line(event, x, y, flags, param):
        nonlocal drawing, last_x, last_y
        
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            last_x, last_y = x, y
        elif event == cv2.EVENT_MOUSEMOVE:
            if drawing:
                cv2.line(canvas, (last_x, last_y), (x, y), color, 5)
                last_x, last_y = x, y
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
    
    # Create window and assign callback
    cv2.namedWindow("Drawing Test")
    cv2.setMouseCallback("Drawing Test", draw_line)
    
    print("Draw with your mouse to test the canvas. Press 'q' to quit.")
    
    while True:
        display_canvas = canvas.copy()
        
        # Add instructions
        cv2.putText(display_canvas, "Click and drag to draw. Press 'q' to quit.", (20, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow("Drawing Test", display_canvas)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cv2.destroyAllWindows()
    print("Drawing canvas test completed successfully.")
    return True

def run_all_tests():
    """Run all the tests in sequence."""
    print("Starting test sequence for Finger Drawing App...")
    
    webcam_ok = test_webcam()
    if not webcam_ok:
        print("Webcam test failed. Please check your camera and try again.")
        return False
    
    hand_detection_ok = test_hand_detection()
    if not hand_detection_ok:
        print("Hand detection test had issues. The app may still work, but performance might be degraded.")
    
    canvas_ok = test_drawing_canvas()
    if not canvas_ok:
        print("Canvas drawing test failed. There might be issues with the application.")
        return False
    
    print("All tests completed. You should be ready to use the Finger Drawing App!")
    return True

if __name__ == "__main__":
    run_all_tests()
