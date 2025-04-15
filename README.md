# Finger Drawing Webcam Application

This application uses your webcam to detect hand gestures, allowing you to draw with your index finger and erase with your palm. It features a real-time drawing canvas that responds to natural hand movements.


https://github.com/user-attachments/assets/66dd2cf0-5b3a-43f4-bef0-4100735eb271

## Features

- Real-time finger tracking using MediaPipe
- Draw on screen by extending your index finger
- Erase drawings by showing your palm (open hand gesture) - eraser size dynamically changes based on how close your palm is to the camera
- Color picker accessed via gesture (index + middle finger extended)
- Multiple color options with keyboard shortcuts
- Adjustable brush thickness
- Save your drawings to PNG files
- Clear canvas with a keystroke

## Requirements

- Python 3.7+
- Webcam

## Installation

1. Clone this repository:
```
git clone https://github.com/yourusername/finger_drawing_app.git
cd finger_drawing_app
```

2. Install the required packages:
```
pip install -r requirements.txt
```

## Usage

We provide two versions of the application:

1. Basic version: `python app.py`
2. Enhanced version with more features: `python enhanced_app.py`

You can also use the provided launcher scripts:
- On Windows: Double-click `run.bat`
- On Linux/Mac: `./run.sh` (make it executable first with `chmod +x run.sh`)

### Controls

#### Hand Gestures:
- **Index finger extended**: Draw on the canvas
- **Open palm** (all fingers extended): Erase in a circular area
- **Index + Middle finger extended** (peace sign): Open color picker
- **Pinch gesture** (thumb + index): Select color from color picker

#### Keyboard Controls:
- Press **'r'** to change color to red
- Press **'g'** to change color to green
- Press **'b'** to change color to blue
- Press **'w'** to change color to white
- Press **'k'** to change color to black
- Press **'p'** to toggle color picker
- Press **'+'** or **'='** to increase brush thickness
- Press **'-'** to decrease brush thickness
- Press **'c'** to clear the canvas
- Press **'s'** to save your drawing
- Press **ESC** to exit the application

### Testing

To verify your setup is working correctly, run the test utility:
```
python test_utils.py
python enhanced_app.py
```

This will test your webcam, hand detection, and drawing functionality separately.

## How It Works

The application uses MediaPipe's hand tracking to detect hand landmarks in real-time. It processes the webcam feed to identify specific hand gestures:

1. **Drawing Mode**: When only the index finger is extended upward, the app tracks its movement and draws lines following the fingertip.

2. **Erasing Mode**: When most fingers are extended (open palm), the app creates a circular eraser that removes any drawing in its path.

3. **Color Selection**: The enhanced version allows you to select colors using a "peace sign" gesture (index and middle fingers extended) to open the color picker, then a pinch gesture to select a color.

The drawing is overlaid onto the webcam feed using an alpha-blended canvas, creating an interactive drawing experience without the need for physical touch.

### Technical Implementation

- **Hand Landmark Detection**: Uses MediaPipe's 21-point hand landmark model to track finger positions
- **Gesture Recognition**: Custom algorithms to detect specific finger configurations
- **Canvas Management**: RGBA canvas with transparency for overlay onto the webcam feed
- **Color Management**: HSV color gradient for intuitive color selection

## Project Structure

- `app.py`: Basic finger drawing application
- `enhanced_app.py`: Extended version with color picker and additional gestures
- `gesture_utils.py`: Utility functions for gesture detection and UI
- `test_utils.py`: Diagnostic tools to test webcam, hand detection, and drawing
- `requirements.txt`: Required Python packages
- `run.sh` & `run.bat`: Launcher scripts for Linux/Mac and Windows

## Troubleshooting

- **Webcam not detected**: Ensure your webcam is properly connected and not being used by another application
- **Poor hand detection**: 
  - Ensure you have good lighting - natural light works best
  - Keep your hand within the camera frame
  - Try to minimize background movement and ensure a non-distracting background
  - Hold your hand about 1-2 feet from the camera for optimal detection
  - If detection is still unstable, try adjusting the `min_detection_confidence` parameter in the code (lower value = more sensitive)
- **Drawing lag**: Close other resource-intensive applications to improve performance
- **Installation issues**: Make sure you have Python 3.7+ and pip installed properly
- **Eraser size issues**: 
  - Move your palm slowly closer/further from the camera
  - The distance detection works best with consistent lighting
  - The palm must be fully open and facing the camera

### Common errors and solutions:

1. **MediaPipe installation error**: Try `pip install mediapipe-silicon` for Mac M1/M2 users
2. **OpenCV window doesn't respond**: Press 'q' or 'ESC' to close windows between tests
3. **Drawing doesn't appear**: Make sure you're moving your index finger with the tip visible to the camera
4. **Palm detection issues**: Make sure all your fingers are extended and your palm is facing the camera
5. **Eraser size not changing**: Try moving your palm in a range from 1 foot to 2.5 feet from the camera for best results

## Advanced Customization

You can modify several parameters in the code to customize your experience:
- Change brush styles and thickness in the `process_hands` method
- Adjust gesture sensitivity in the `is_finger_extended` and `is_palm_showing` functions
- Add custom gestures by creating new detection functions in `gesture_utils.py`

## License

MIT

## Future Enhancements

- Multiple brush styles (pencil, marker, spray)
- Shape recognition (draw a circle gesture to create perfect circle)
- Multiple users drawing simultaneously 
- Image import/export functionality
- Undo/redo functionality
