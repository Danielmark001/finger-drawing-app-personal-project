# Finger Drawing App

A webcam-based drawing application that lets you create art using hand gestures. Draw with your index finger, erase with your palm, and interact with a digital canvas - all without touching your computer.


https://github.com/user-attachments/assets/d71576cf-baa4-4a73-8711-e3b92133c618


## Features

- Draw by simply pointing your index finger at the camera
- Erase by showing your palm (eraser size changes with distance)
- Choose colors with intuitive gestures
- Multiple brush styles: solid, airbrush, marker, and pencil
- Drawing tools: lines, rectangles, circles, and fill tool
- Multiple layers for complex drawings
- Undo/redo functionality
- Save your drawings as PNG files

## Requirements

- Python 3.7+
- Webcam
- OpenCV, NumPy, and MediaPipe libraries

## Installation

1. Clone or download this repository
2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Running the App

The app offers three different versions with increasing features:

1. Run the app directly:
   ```
   python main.py
   ```
   This will show a menu where you can choose which version to run.

2. Or run a specific version directly:
   - Basic version: `python app.py`
   - Standard version: `python enhanced_app.py`
   - Full version: `python finger_draw.py`

3. First-time users should check out the tutorial:
   ```
   python tutorial.py
   ```

## Controls

### Hand Gestures

- **Index finger extended**: Draw on the canvas
- **Open palm**: Erase (size changes with distance from camera)
- **Peace sign** (index + middle finger): Open color picker
- **Pinch** (thumb + index): Select color or UI element
- **Three fingers** (index, middle, ring): Open tools menu
- **Fist**: Undo last action
- **OK sign** (circle with thumb + index): Select tool

### Keyboard Shortcuts

- **Colors**: r (red), g (green), b (blue), w (white), k (black)
- **Brush Size**: + (increase), - (decrease)
- **Brush Styles**: 1 (solid), 2 (airbrush), 3 (marker), 4 (pencil)
- **Actions**: c (clear), s (save), z (undo), y (redo), l (layers), h (help)
- **Exit**: ESC

## Tips for Best Results

- Ensure good lighting on your hands
- Use a plain, uncluttered background
- Position your hand 1-2 feet from the camera
- Make gestures clearly and hold them briefly
- Keep your hand within the camera frame

## Customization

You can customize various settings in the `config.py` file:
- Display settings (resolution, fullscreen)
- Performance options for slower computers
- Default brush and color settings
- File management preferences
- UI appearance options

## Troubleshooting

- **Webcam not detected**: Make sure no other application is using your webcam
- **Hand detection issues**: Improve lighting or try a simpler background
- **Performance problems**: Lower the resolution in config.py or disable hand landmarks display
- **Drawing lag**: Try the basic version which uses fewer resources

## License

MIT License

## Acknowledgements

- MediaPipe team for the hand tracking library
- OpenCV contributors for image processing capabilities
- All contributors and testers of this project
