import os
import sys
import cv2
import time
import subprocess

# Ensure we're in the right directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Check for required packages
def check_requirements():
    """Check for required packages and install if missing."""
    try:
        import numpy
        import mediapipe
        import cv2
        return True
    except ImportError as e:
        print(f"Missing required package: {e}")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
            return True
        except:
            print("Failed to install requirements. Please install manually with:")
            print("pip install -r requirements.txt")
            return False

# Check if webcam is available
def check_webcam():
    """Check if webcam is available."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not access webcam.")
        cap.release()
        return False
    
    # Try to read a frame
    ret, frame = cap.read()
    if not ret or frame is None:
        print("Error: Could not read from webcam.")
        cap.release()
        return False
    
    cap.release()
    return True

def main():
    """Main function to run the app."""
    # Check requirements
    if not check_requirements():
        print("Missing required packages. Exiting...")
        input("Press Enter to exit...")
        return
    
    # Check webcam
    if not check_webcam():
        print("Webcam not available. The app requires a working webcam.")
        choice = input("Do you want to continue anyway? (y/n): ")
        if choice.lower() != 'y':
            return
    
    # Import splash screen if available
    try:
        from splash import run_splash_screen
        # Show splash screen
        run_splash_screen()
    except ImportError:
        print("Note: Splash screen module not found, continuing without it.")
    
    print("=" * 50)
    print("Starting Finger Drawing App...")
    print("=" * 50)
    
    # Launch the full version directly
    try:
        import finger_draw
        app_instance = finger_draw.FingerDrawingApp()
        app_instance.run()
    except Exception as e:
        print(f"Error running the app: {e}")
        input("Press Enter to exit...")

if __name__ == "__main__":
    main()
