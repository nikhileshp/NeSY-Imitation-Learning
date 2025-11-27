"""
Freeway-specific configuration file.
Contains colors, detection parameters, and game constants for Freeway.
"""

# Road boundaries for Freeway
ROAD_TOP_Y = 24
ROAD_BOTTOM_Y = 180

# Object color definitions for Freeway
OBJECT_COLORS = {
    "chicken": [[252, 252, 84]],
    "player_score": [[228, 111, 111]],
    "enemy_score": [[228, 111, 111]],
    # Car colors - 10 different lanes
    "car1": [[167, 26, 26]],
    "car2": [[180, 231, 117]],
    "car3": [[105, 105, 15]],
    "car4": [[228, 111, 111]],
    "car5": [[24, 26, 167]],
    "car6": [[162, 98, 33]],
    "car7": [[84, 92, 214]],
    "car8": [[184, 50, 50]],
    "car9": [[135, 183, 84]],
    "car10": [[210, 210, 64]]
}

# Combined car colors for easier detection
CAR_COLORS = {
    "car1": [167, 26, 26],
    "car2": [180, 231, 117],
    "car3": [105, 105, 15],
    "car4": [228, 111, 111],
    "car5": [24, 26, 167],
    "car6": [162, 98, 33],
    "car7": [84, 92, 214],
    "car8": [184, 50, 50],
    "car9": [135, 183, 84],
    "car10": [210, 210, 64]
}

# Lane positions based on the actual game frame (10 lanes, ~16 pixels each)
LANE_POSITIONS = {
    1: (24, 40),    # Top lane
    2: (40, 56),
    3: (56, 72),
    4: (72, 88),
    5: (88, 104),
    6: (104, 120),
    7: (120, 136),
    8: (136, 152),
    9: (152, 168),
    10: (168, 184)  # Bottom lane
}

# Object detection parameters for Freeway
DETECTION_PARAMS = {
    "chicken": {
        "size": (7, 8),
        "tol_s": 3,
        "miny": 20,
        "maxy": 185
    },
    "player_score": {
        "min_distance": 1,
        "maxy": 20,
        "maxx": 80
    },
    "enemy_score": {
        "min_distance": 1,
        "maxy": 20,
        "minx": 80
    },
    # Car detection parameters - one for each lane (exactly 1 car per lane max)
    "car1": {
        "min_distance": 1,
        "miny": 24,
        "maxy": 40
    },
    "car2": {
        "min_distance": 1,
        "miny": 40,
        "maxy": 56
    },
    "car3": {
        "min_distance": 1,
        "miny": 56,
        "maxy": 72
    },
    "car4": {
        "min_distance": 1,
        "miny": 72,
        "maxy": 88
    },
    "car5": {
        "min_distance": 1,
        "miny": 88,
        "maxy": 104
    },
    "car6": {
        "min_distance": 1,
        "miny": 104,
        "maxy": 120
    },
    "car7": {
        "min_distance": 1,
        "miny": 120,
        "maxy": 136
    },
    "car8": {
        "min_distance": 1,
        "miny": 136,
        "maxy": 152
    },
    "car9": {
        "min_distance": 1,
        "miny": 152,
        "maxy": 168
    },
    "car10": {
        "min_distance": 1,
        "miny": 168,
        "maxy": 184
    }
}

# Visualization colors for Freeway objects (BGR format for OpenCV)
VISUALIZATION_COLORS = {
    "chicken": (84, 252, 252),  # Yellow (BGR)
    "player_score": (111, 111, 228),  # Red
    "enemy_score": (111, 111, 228),  # Red
    "car": (0, 0, 255),  # Red for generic cars
    "car1": (26, 26, 167),  # Dark red
    "car2": (117, 231, 180),  # Light green
    "car3": (15, 105, 105),  # Dark olive
    "car4": (111, 111, 228),  # Red
    "car5": (167, 26, 24),  # Blue
    "car6": (33, 98, 162),  # Brown
    "car7": (214, 92, 84),  # Light blue
    "car8": (50, 50, 184),  # Dark red
    "car9": (84, 183, 135),  # Light green
    "car10": (64, 210, 210)  # Yellow
}