# Finger Drawing App - User Configuration
# Edit this file to customize your experience

#----------------------
# Display Settings
#----------------------
WINDOW_TITLE = "Finger Drawing Studio Pro"
FULLSCREEN = True  # Set to True for fullscreen mode
DEFAULT_WIDTH = 1280
DEFAULT_HEIGHT = 720

#----------------------
# Performance Settings
#----------------------
# Lower these values on slower computers
FPS_LIMIT = 60  # Target frames per second
REDUCED_QUALITY = False  # Set to True on slower machines
SHOW_LANDMARKS = True  # Set to False to improve performance

#----------------------
# Drawing Settings
#----------------------
DEFAULT_COLOR = (0, 0, 255)  # Red in BGR format
DEFAULT_THICKNESS = 5
MAX_THICKNESS = 50  # Maximum brush thickness
DEFAULT_BRUSH = "solid"  # Options: solid, airbrush, marker, pencil, charcoal, highlighter

#----------------------
# Hand Gesture Detection
#----------------------
DETECTION_CONFIDENCE = 0.7  # 0.0-1.0, lower = more sensitive but less accurate
TRACKING_CONFIDENCE = 0.5   # 0.0-1.0, lower = smoother tracking but more errors
MAX_HANDS = 2  # Number of hands to track simultaneously

#----------------------
# File Management
#----------------------
SAVE_DIRECTORY = "drawings"  # Directory to save drawings
AUTOSAVE = True  # Set to True to automatically save every N minutes
AUTOSAVE_INTERVAL = 5  # Minutes between autosaves
SAVE_FORMATS = ["png", "jpg", "svg"]  # Available save formats

#----------------------
# User Interface
#----------------------
SHOW_TOOLBAR = True  # Show the toolbar at startup
DARK_MODE = True  # Dark color scheme
UI_TRANSPARENCY = 0.8  # 0.0-1.0, higher = more transparent
SHOW_HELP_AT_STARTUP = True  # Show help screen on first launch
MODERN_UI = True  # Use modern UI elements
SHOW_GRID = False  # Show grid overlay for precise drawing
GRID_SIZE = 20  # Size of grid cells

#----------------------
# Effect Settings
#----------------------
APPLY_ANTIALIASING = True  # Smoother lines but slightly slower
ENABLE_TEXTURES = True  # Apply paper-like texture to drawings
DEFAULT_TEXTURE = "paper"  # Options: none, paper, canvas, rough

#----------------------
# New Features
#----------------------
ENABLE_PRESSURE_SENSITIVITY = True  # Use hand distance as pressure input
ENABLE_SYMMETRY = False  # Enable symmetrical drawing
SYMMETRY_MODE = "horizontal"  # Options: horizontal, vertical, quadrant
ENABLE_SNAP_TO_GRID = False  # Enable snapping to grid points
ENABLE_VOICE_COMMANDS = False  # Enable voice command detection (if available)
ENABLE_EXPORT_TO_SOCIAL = True  # Enable direct export to social media

#----------------------
# Advanced Settings
#----------------------
ENABLE_BACKGROUND_IMAGES = True  # Allow background image import
ENABLE_STICKERS = True  # Enable sticker library
ENABLE_TEXT_TOOL = True  # Enable text input tool
ENABLE_UNDO_HISTORY = True  # Enable extended undo history
MAX_UNDO_STEPS = 50  # Maximum number of undo steps
ENABLE_RULER_TOOLS = True  # Enable ruler and measurement tools
