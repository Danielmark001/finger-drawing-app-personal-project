import cv2
import numpy as np
import mediapipe as mp
import time
import os
import sys

def test_webcam():
    """Test if webcam is functioning properly."""
    print("Testing webcam...")
    
    # Try to open the webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ ERROR: Could not open webcam.")
        return False
    
    # Try to read a frame
    ret, frame = cap.read()
    if not ret or frame is None or frame.size == 0:
        print("❌ ERROR: Could not read frame from webcam.")
        cap.release()
        return False
    
    # Show webcam feed with instructions
    print("✅ Webcam is working!")
    print("You should see your webcam feed in a new window.")
    print("Press 'q' to continue to the next test.")
    
    window_name = "Webcam Test"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    # Display webcam feed for a few seconds
    start_time = time.time()
    frame_count = 0
    
    while time.time() - start_time < 10:  # Run for 10 seconds max
        ret, frame = cap.read()
        if not ret:
            break
        
        # Show FPS
        frame_count += 1
        elapsed = time.time() - start_time
        fps = frame_count / elapsed if elapsed > 0 else 0
        
        # Add text instructions
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, "Press 'q' to continue", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Show frame
        cv2.imshow(window_name, frame)
        
        # Check for 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Clean up
    cap.release()
    cv2.destroyWindow(window_name)
    
    return True

def test_hand_detection():
    """Test if hand detection is working properly."""
    print("\nTesting hand detection...")
    
    # Initialize hand detection
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    
    try:
        hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
    except Exception as e:
        print(f"❌ ERROR: Could not initialize hand detection: {e}")
        return False
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ ERROR: Could not open webcam.")
        return False
    
    # Show webcam feed with hand detection
    print("✅ Hand detection initialized!")
    print("Show your hand to the camera.")
    print("You should see hand landmarks if detection is working.")
    print("Press 'q' to continue to the next test.")
    
    window_name = "Hand Detection Test"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    # Frame counter for detection success rate
    total_frames = 0
    detected_frames = 0
    
    # Display webcam feed with hand detection
    start_time = time.time()
    
    while time.time() - start_time < 20:  # Run for 20 seconds max
        ret, frame = cap.read()
        if not ret:
            break
        
        # Flip the frame for more natural interaction
        frame = cv2.flip(frame, 1)
        
        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame
        results = hands.process(rgb_frame)
        
        total_frames += 1
        
        # Draw hand landmarks
        if results.multi_hand_landmarks:
            detected_frames += 1
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, 
                    hand_landmarks, 
                    mp_hands.HAND_CONNECTIONS
                )
        
        # Calculate detection rate
        detection_rate = (detected_frames / total_frames) * 100 if total_frames > 0 else 0
        
        # Add text instructions and stats
        cv2.putText(frame, f"Detection rate: {detection_rate:.1f}%", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        if results.multi_hand_landmarks:
            cv2.putText(frame, "Hand detected!", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "No hand detected", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        cv2.putText(frame, "Press 'q' to continue", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Show frame
        cv2.imshow(window_name, frame)
        
        # Check for 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Clean up
    cap.release()
    cv2.destroyWindow(window_name)
    hands.close()
    
    # Evaluate results
    if detected_frames == 0:
        print("❌ WARNING: No hands were detected during the test.")
        print("   Make sure your hand is visible to the camera and well lit.")
        print("   The app may still work, but hand detection might be unreliable.")
        return False
    else:
        print(f"✅ Hand detection working with {detection_rate:.1f}% detection rate!")
        if detection_rate < 50:
            print("   Detection rate is a bit low. Try improving lighting or camera position.")
        return True

def test_drawing():
    """Test basic drawing functionality."""
    print("\nTesting drawing functionality...")
    
    # Create a blank canvas
    canvas = np.zeros((480, 640, 3), dtype=np.uint8)
    canvas.fill(255)  # White background
    
    # Initialize drawing variables
    drawing = False
    last_x, last_y = -1, -1
    color = (0, 0, 255)  # Red
    thickness = 5
    
    # Mouse callback function
    def draw(event, x, y, flags, param):
        nonlocal drawing, last_x, last_y
        
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            last_x, last_y = x, y
        elif event == cv2.EVENT_MOUSEMOVE:
            if drawing:
                cv2.line(canvas, (last_x, last_y), (x, y), color, thickness)
                last_x, last_y = x, y
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
    
    # Set up window
    window_name = "Drawing Test"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, draw)
    
    # Instructions
    print("✅ Drawing test initialized!")
    print("Try drawing on the canvas with your mouse.")
    print("Press 'c' to clear, 'r/g/b' to change colors, '+/-' for thickness.")
    print("Press 'q' to finish the test.")
    
    while True:
        # Create a copy of the canvas with instructions
        display = canvas.copy()
        
        # Add text instructions
        cv2.putText(display, "Drawing Test", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.putText(display, "Mouse: Draw | c: Clear | r/g/b: Colors | +/-: Thickness | q: Quit", 
                   (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # Show canvas
        cv2.imshow(window_name, display)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('c'):
            canvas.fill(255)
        elif key == ord('r'):
            color = (0, 0, 255)  # Red
        elif key == ord('g'):
            color = (0, 255, 0)  # Green
        elif key == ord('b'):
            color = (255, 0, 0)  # Blue
        elif key == ord('+') or key == ord('='):
            thickness = min(30, thickness + 1)
        elif key == ord('-'):
            thickness = max(1, thickness - 1)
    
    # Clean up
    cv2.destroyWindow(window_name)
    
    print("✅ Drawing test completed!")
    return True

def check_requirements():
    """Check if all required packages are installed."""
    print("Checking required packages...")
    
    try:
        import numpy
        print("✅ NumPy is installed")
    except ImportError:
        print("❌ ERROR: NumPy is not installed. Run: pip install numpy")
        return False
    
    try:
        import cv2
        print(f"✅ OpenCV is installed (version {cv2.__version__})")
    except ImportError:
        print("❌ ERROR: OpenCV is not installed. Run: pip install opencv-python")
        return False
    
    try:
        import mediapipe
        print(f"✅ MediaPipe is installed (version {mediapipe.__version__})")
    except ImportError:
        print("❌ ERROR: MediaPipe is not installed. Run: pip install mediapipe")
        return False
    
    print("✅ All required packages are installed!")
    return True

def run_tests():
    """Run all tests."""
    print("=" * 50)
    print("FINGER DRAWING APP - DIAGNOSTIC TESTS")
    print("=" * 50)
    
    # Check requirements
    if not check_requirements():
        print("\n❌ Requirements check failed. Please install missing packages.")
        input("Press Enter to exit...")
        return
    
    print("\nRunning tests...")
    
    # Test webcam
    webcam_ok = test_webcam()
    
    # Test hand detection
    if webcam_ok:
        hand_detection_ok = test_hand_detection()
    else:
        print("\n⚠️ Skipping hand detection test due to webcam issues.")
        hand_detection_ok = False
    
    # Test drawing functionality
    drawing_ok = test_drawing()
    
    # Print summary
    print("\n" + "=" * 50)
    print("TEST RESULTS:")
    print("=" * 50)
    print(f"Webcam: {'✅ PASS' if webcam_ok else '❌ FAIL'}")
    print(f"Hand detection: {'✅ PASS' if hand_detection_ok else '❌ FAIL' if not webcam_ok else '⚠️ WARNING'}")
    print(f"Drawing: {'✅ PASS' if drawing_ok else '❌ FAIL'}")
    
    # Conclusion
    if webcam_ok and (hand_detection_ok or not webcam_ok) and drawing_ok:
        print("\n✅ All critical tests passed! You're ready to use the app.")
        print("Run 'python main.py' to start the application.")
    elif not webcam_ok:
        print("\n❌ The webcam test failed. The app requires a working webcam.")
        print("Please troubleshoot your webcam before using the app.")
    elif not hand_detection_ok and webcam_ok:
        print("\n⚠️ Hand detection might be unreliable, but you can still try the app.")
        print("For better results, ensure good lighting and a clean background.")
    else:
        print("\n❌ Some tests failed. Please review the issues before using the app.")
    
    input("\nPress Enter to exit...")

if __name__ == "__main__":
    run_tests()
