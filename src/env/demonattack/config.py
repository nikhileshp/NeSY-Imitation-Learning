"""
DemonAttack-specific configuration file.
Contains colors, detection parameters, and game constants for DemonAttack.
"""

# Screen boundaries for DemonAttack
SCREEN_TOP_Y = 10
SCREEN_BOTTOM_Y = 180
PLAYER_Y_POSITION = 180  # Player is fixed at bottom

# Object color definitions for DemonAttack
OBJECT_COLORS = {
    "player": [[184, 70, 162]],
    "projectile_friendly": [[212, 140, 252]],
    "projectile_hostile": [[252, 144, 144]],
    "live": [[240, 128, 128]],
    "score": [[223, 183, 85]],
    # Enemy has multiple color variations
    "enemy": [
        [72, 160, 72],
        [84, 92, 214], [84, 138, 210], [84, 160, 197], [84, 184, 153],
        [92, 186, 92], [101, 111, 228], [104, 72, 198], [127, 92, 213],
        [149, 111, 227], [181, 108, 224], [195, 144, 61], [197, 124, 238],
        [212, 108, 195], [213, 130, 74], [214, 92, 92], [214, 214, 214],
        [224, 236, 124], [227, 151, 89], [228, 111, 111]
    ]
}

# Object detection parameters for DemonAttack
DETECTION_PARAMS = {
    "player": {
        "min_distance": 1,
        "miny": 170,
        "maxy": 195
    },
    "enemy_large": {
        "closing_dist": 4,
        "all_colors": False,
        "size": (14, 7),
        "tol_s": (3, 3),
        "miny": 10,
        "maxy": 180
    },
    "enemy_small": {
        "closing_dist": 1,
        "all_colors": False,
        "size": (7, 4),
        "tol_s": (2, 2),
        "miny": 10,
        "maxy": 180
    },
    "projectile_friendly": {
        "min_distance": 1,
        "miny": 10,
        "maxy": 180
    },
    "projectile_hostile": {
        "min_distance": 1,
        "miny": 10,
        "maxy": 180
    },
    "score": {
        "min_distance": 1,
        "closing_dist": 5,
        "maxy": 20
    },
    "live": {
        "min_distance": 1,
        "maxy": 20
    }
}

# Visualization colors for DemonAttack objects (BGR format for OpenCV)
VISUALIZATION_COLORS = {
    "player": (162, 70, 184),  # Purple (BGR)
    "enemy": (74, 130, 213),  # Orange-brown
    "projectile_friendly": (252, 140, 212),  # Light purple
    "projectile_hostile": (144, 144, 252),  # Red-pink
    "score": (85, 183, 223),  # Yellow
    "live": (128, 128, 240)  # Light red
}
