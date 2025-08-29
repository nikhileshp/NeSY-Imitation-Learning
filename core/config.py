"""
Core configuration file for game object detection framework.
Contains base constants and visualization settings.
"""

# Base visualization colors (BGR format for OpenCV)
BASE_VISUALIZATION_COLORS = {
    "gaze_position": (0, 0, 255),       # Red
    "relationship_line": (0, 0, 0),     # Black
    "relationship_text": (0, 0, 0)     # Black
}

# General constants
MIN_COVERAGE_RATIO = 0.51

# Default object visualization colors
DEFAULT_OBJECT_COLORS = {
    "player": (0, 255, 0),      # Green
    "enemy": (0, 0, 255),       # Red
    "neutral": (255, 255, 255), # White
    "collectible": (255, 0, 255), # Magenta
    "obstacle": (100, 100, 100), # Gray
    "projectile": (0, 255, 255)  # Cyan
}
