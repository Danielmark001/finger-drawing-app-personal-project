import cv2
import time

def test_webcam():
    print("Testing webcam access...")
    
    # Try different camera indices
    for camera_idx in range(3):  # try camera indices 0, 1, 2
        print(f"Trying camera index {camera_idx}...")
        cap = cv2.VideoCapture(camera_idx)
        
        if not cap.isOpened():
            print(f"Could not open webcam with index {camera_idx}")
            cap.release()
            continue
            
        # Try to read a frame
        ret, frame = cap.read()
        if not ret or frame is None:
            print(f"Could not read from webcam with index {camera_idx}")
            cap.release()
            continue
            
        print(f"Successfully accessed webcam with index {camera_idx}")
        print(f"Frame shape: {frame.shape}")
        
        # Try to display the frame
        try:
            cv2.imshow('Webcam Test', frame)
            print("Displaying webcam frame for 5 seconds...")
            start_time = time.time()
            while time.time() - start_time < 5:
                ret, frame = cap.read()
                if ret:
                    cv2.imshow('Webcam Test', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            cv2.destroyAllWindows()
        except Exception as e:
            print(f"Error displaying frame: {e}")
        
        cap.release()
        return True
        
    print("Could not access any webcam (tried indices 0-2)")
    return False

if __name__ == "__main__":
    test_webcam()
