"""
Seaquest-specific configuration file.
Contains colors, detection parameters, and game constants for Seaquest.
"""

# Water surface level for Seaquest
WATER_SURFACE_Y = 47

# Enemy color definitions specific to Seaquest
ENEMY_COLORS = {
    "green": [92, 186, 92],
    "orange": [[58, 108, 198], [111, 111, 111]],
    "yellow": [[111, 111, 111], [79, 171, 160]],
    "lightgreen": [72, 160, 72],
    "pink": [[179, 89, 198], [192, 70, 146]]
}

# Game object color definitions for Seaquest
OBJECT_COLORS = {
    "player": [[53, 187, 187]],
    "diver": [[200, 72, 66], [184, 50, 45]],
    "background_water": [[0, 28, 136]],
    "player_score": [[210, 210, 64]],
    "oxygen_bar": [[214, 214, 214]],
    "lives": [[64, 210, 210]],
    "logo": [[66, 72, 200]],
    "player_missile": [[142, 142, 142], [53, 187, 187]],
    "oxygen_bar_depleted": [[21, 57, 163]],
    "oxygen_logo": [[0, 0, 0]],
    "collected_diver": [[167, 26, 24]],
    "enemy_missile": [[200, 72, 66]],
    "submarine": [[170, 170, 170]]
}

# Object detection parameters for Seaquest
DETECTION_PARAMS = {
    "player": {"size": (15, 10), "tol_s": (14, 8)},
    "diver": {"size": (12, 8), "tol_s": 6, "miny": 40, "maxy": 150},
    "player_missile": {"miny": 40, "maxy": 150, "size": (12, 1), "tol_s": (5, 1)},
    "enemy_missile": {"size": (8, 4), "tol_s": 1, "minx": 20, "miny": 90, "maxy": 310},
    "lives": {"closing_active": False, "miny": 20},
    "submarine": {"size": (8, 10), "tol_s": 4, "miny": 56, "maxy": 150, "min_distance": 1},
    "submarine_on_water": {"miny": 40, "maxy": 55, "min_distance": 1},
    "oxygen_bar": {"min_distance": 1},
    "oxygen_bar_depleted": {"min_distance": 1},
    "oxygen_logo": {"min_distance": 1},
    "collected_diver": {"closing_active": False},
    "enemy": {"maxy": 150, "miny": 40}
}

# Visualization colors for Seaquest objects (BGR format for OpenCV)
VISUALIZATION_COLORS = {
    "player": (0, 255, 0),      # Green
    "diver": (255, 0, 0),       # Blue
    "collected_diver": (150, 0, 150),   # Purple
    "player_missile": (0, 255, 0),      # Green
    "enemy_missile": (0, 0, 255),       # Red
    "lives": (255, 255, 0),     # Cyan
    "enemy_submarine": (0, 0, 255),     # Red
    "oxygen_bar": (255, 0, 255),        # Magenta
    "oxygen_depleted": (100, 100, 100), # Gray
    "enemy": (0, 0, 255)        # Red
}
