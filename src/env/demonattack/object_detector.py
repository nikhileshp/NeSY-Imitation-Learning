"""
DemonAttack-specific object detector implementation with integrated visualization and relationships.
"""
import numpy as np
from typing import List, Dict
import sys
import os

# Add models.ocatari to path for imports
sys.path.append('/Users/varun/Desktop/NeSy-Imitation-Learning/')
sys.path.append('/Users/varun/Desktop/NeSy-Imitation-Learning/src')

from core.object_detector import BaseObjectDetector, GameConfig
from core.game_object import GameObject
from core.object_tracker import ObjectTracker

# Handle both relative and absolute imports
try:
    from .config import OBJECT_COLORS, DETECTION_PARAMS
except ImportError:
    try:
        from config import OBJECT_COLORS, DETECTION_PARAMS
    except ImportError:
        from env.demonattack.config import OBJECT_COLORS, DETECTION_PARAMS

from models.ocatari.ocatari.vision.utils import find_objects, find_mc_objects

# Import relationship analyzer
try:
    from .relationship_analyzer import DemonAttackRelationshipAnalyzer
except ImportError:
    try:
        from relationship_analyzer import DemonAttackRelationshipAnalyzer
    except ImportError:
        from env.demonattack.relationship_analyzer import DemonAttackRelationshipAnalyzer


class DemonAttackGameConfig:
    """Game configuration for DemonAttack."""
    
    def __init__(self):
        self.object_colors = OBJECT_COLORS
        self.detection_params = DETECTION_PARAMS
    
    def get_object_types(self) -> List[str]:
        """Return list of object types for DemonAttack."""
        return [
            'player', 'enemy', 
            'projectile_friendly', 'projectile_hostile',
            'score', 'live'
        ]


class DemonAttackObjectDetector(BaseObjectDetector):
    """DemonAttack-specific object detector with custom detection logic and object tracking."""
    
    def __init__(self):
        """Initialize with DemonAttack configuration."""
        super().__init__(DemonAttackGameConfig())
        
        # Define max objects for tracking
        max_objects = {
            'player': 1,
            'enemy': 20,
            'projectile_friendly': 10,
            'projectile_hostile': 10,
            'score': 1,
            'live': 10
        }
        
        self.object_tracker = ObjectTracker(max_objects)
        self.use_tracking = True
    
    def detect_objects_by_type(self, image, object_type):
        """Generic detection for simple object types."""
        if object_type not in self.game_config.object_colors:
            return []
        
        colors = self.game_config.object_colors[object_type]
        params = self.game_config.detection_params.get(object_type, {})
        
        coords_list = find_objects(image, colors, **params)
        
        objects = []
        for i, coords in enumerate(coords_list):
            obj = GameObject(
                object_type=object_type,
                bounding_box=coords,
                object_id=f'{object_type}_{i}'
            )
            objects.append(obj)
        
        return objects
    
    def detect_all_objects(self, image: np.ndarray) -> Dict[str, List[GameObject]]:
        """Detect all objects in the DemonAttack game frame."""
        detected_objects = {}
        
        # Detect player
        detected_objects['player'] = self._detect_player(image)
        
        # Detect enemies (multiple detection passes for different sizes)
        detected_objects['enemy'] = self._detect_enemies(image)
        
        # Detect projectiles
        detected_objects['projectile_friendly'] = self._detect_projectiles(image, 'projectile_friendly')
        detected_objects['projectile_hostile'] = self._detect_projectiles(image, 'projectile_hostile')
        
        # Detect HUD elements
        detected_objects['score'] = self._detect_score(image)
        detected_objects['live'] = self._detect_lives(image)
        
        # Cleanup detections
        self._cleanup_detections(detected_objects)
        
        # Apply tracking
        if self.use_tracking:
            detected_objects = self.object_tracker.track_all_objects(detected_objects)
        
        return detected_objects
    
    def _detect_player(self, image: np.ndarray) -> List[GameObject]:
        """Detect player at bottom of screen."""
        colors = self.game_config.object_colors['player']
        params = self.game_config.detection_params['player']
        
        player_coords = find_objects(image, colors, **params)
        
        if player_coords:
            coords = player_coords[0]  # Only one player
            player = GameObject(
                'player', 
                coords, 
                object_id='player',
                characteristics={'color_rgb': [184, 70, 162]}
            )
            return [player]
        
        return []
    
    def _detect_enemies(self, image: np.ndarray) -> List[GameObject]:
        """Detect enemies using multi-color and multi-size detection."""
        enemies = []
        colors = self.game_config.object_colors['enemy']
        
        # Detect large enemies
        params_large = self.game_config.detection_params['enemy_large']
        enemy_coords_large = find_mc_objects(
            image, 
            colors,
            closing_dist=params_large['closing_dist'],
            all_colors=params_large['all_colors'],
            size=params_large['size'],
            tol_s=params_large['tol_s'],
            miny=params_large['miny'],
            maxy=params_large['maxy']
        )
        
        # Detect small enemies
        params_small = self.game_config.detection_params['enemy_small']
        enemy_coords_small = find_mc_objects(
            image, 
            colors,
            closing_dist=params_small['closing_dist'],
            all_colors=params_small['all_colors'],
            size=params_small['size'],
            tol_s=params_small['tol_s'],
            miny=params_small['miny'],
            maxy=params_small['maxy']
        )
        
        # Combine all enemy detections
        all_enemy_coords = enemy_coords_large + enemy_coords_small
        
        for i, coords in enumerate(all_enemy_coords):
            # Determine enemy size category
            width = coords[2]
            height = coords[3]
            size_category = 'large' if (width >= 10 or height >= 6) else 'small'
            
            enemy = GameObject(
                'enemy', 
                coords, 
                object_id=f'enemy_{i}',
                characteristics={
                    'size': size_category,
                    'color_rgb': [213, 130, 74]
                }
            )
            enemies.append(enemy)
        
        return enemies
    
    def _group_close_projectiles(self, proj_coords: List, max_horizontal_distance: int = 30, max_vertical_distance: int = 50) -> List:
        """
        Group projectiles that are close together vertically and reasonably close horizontally.
        This groups missiles from the same enemy into one bounding box.
        """
        if not proj_coords:
            return []
        
        # Convert to center points for distance calculation
        projectiles = []
        for coords in proj_coords:
            x, y, w, h = coords
            cx = x + w / 2
            cy = y + h / 2
            projectiles.append({'cx': cx, 'cy': cy, 'coords': coords})
        
        # Sort by vertical position (y coordinate) to group vertically aligned projectiles
        projectiles.sort(key=lambda p: p['cy'])
        
        grouped = []
        used = set()
        
        for i, proj1 in enumerate(projectiles):
            if i in used:
                continue
            
            # Start a new group with this projectile
            group = [proj1['coords']]
            used.add(i)
            
            # Find all nearby projectiles (vertically close and horizontally aligned)
            for j, proj2 in enumerate(projectiles):
                if j in used:
                    continue
                
                # Calculate vertical and horizontal distances
                vertical_dist = abs(proj2['cy'] - proj1['cy'])
                horizontal_dist = abs(proj2['cx'] - proj1['cx'])
                
                # Group if vertically close AND horizontally aligned (from same enemy)
                if vertical_dist <= max_vertical_distance and horizontal_dist <= max_horizontal_distance:
                    group.append(proj2['coords'])
                    used.add(j)
            
            # Create a single bounding box encompassing all projectiles in group
            if group:
                min_x = min(c[0] for c in group)
                min_y = min(c[1] for c in group)
                max_x = max(c[0] + c[2] for c in group)
                max_y = max(c[1] + c[3] for c in group)
                
                # Create combined bounding box
                combined_bbox = (min_x, min_y, max_x - min_x, max_y - min_y)
                grouped.append(combined_bbox)
        
        return grouped
    
    def _detect_projectiles(self, image: np.ndarray, projectile_type: str) -> List[GameObject]:
        """Detect projectiles (friendly or hostile)."""
        colors = self.game_config.object_colors[projectile_type]
        
        # Simplified detection - projectiles are very small
        proj_coords = find_objects(image, colors, min_distance=1, miny=10, maxy=180)
        
        projectiles = []
        
        # For hostile projectiles, group them if they're close together (from same enemy)
        if projectile_type == 'projectile_hostile' and proj_coords:
            grouped_coords = self._group_close_projectiles(proj_coords, max_horizontal_distance=30, max_vertical_distance=50)
            for i, coords in enumerate(grouped_coords):
                proj = GameObject(
                    projectile_type,
                    coords,
                    object_id=f'{projectile_type}_group_{i}',
                    characteristics={'color_rgb': colors[0], 'grouped': True}
                )
                projectiles.append(proj)
        else:
            # For friendly projectiles, keep them separate
            for i, coords in enumerate(proj_coords):
                proj = GameObject(
                    projectile_type,
                    coords,
                    object_id=f'{projectile_type}_{i}',
                    characteristics={'color_rgb': colors[0]}
                )
                projectiles.append(proj)
        
        return projectiles
    
    def _detect_score(self, image: np.ndarray) -> List[GameObject]:
        """Detect score display."""
        colors = self.game_config.object_colors['score']
        params = self.game_config.detection_params['score']
        
        score_coords = find_objects(image, colors, **params)
        
        scores = []
        for i, coords in enumerate(score_coords):
            score = GameObject(
                'score',
                coords,
                object_id='score',
                characteristics={'color_rgb': [223, 183, 85]}
            )
            scores.append(score)
        
        return scores
    
    def _detect_lives(self, image: np.ndarray) -> List[GameObject]:
        """Detect lives indicators."""
        colors = self.game_config.object_colors['live']
        params = self.game_config.detection_params['live']
        
        live_coords = find_objects(image, colors, **params)
        
        lives = []
        for i, coords in enumerate(live_coords):
            live = GameObject(
                'live',
                coords,
                object_id=f'live_{i}',
                characteristics={'color_rgb': [240, 128, 128]}
            )
            lives.append(live)
        
        return lives
    
    def _cleanup_detections(self, detected_objects: Dict[str, List[GameObject]]):
        """Clean up detections to remove invalid objects."""
        # For friendly projectiles, remove tiny ones
        if detected_objects.get('projectile_friendly'):
            detected_objects['projectile_friendly'] = [
                p for p in detected_objects['projectile_friendly'] 
                if p.width >= 1 and p.height >= 1
            ]
        
        # For hostile projectiles (already grouped), just ensure they exist
        if detected_objects.get('projectile_hostile'):
            detected_objects['projectile_hostile'] = [
                p for p in detected_objects['projectile_hostile'] 
                if p.width >= 1 and p.height >= 1
            ]
        
        # Remove tiny enemies
        if detected_objects.get('enemy'):
            detected_objects['enemy'] = [
                e for e in detected_objects['enemy']
                if e.width >= 3 and e.height >= 3
            ]
        
        # Ensure only one player
        if detected_objects.get('player') and len(detected_objects['player']) > 1:
            detected_objects['player'] = [detected_objects['player'][0]]


# ============================================================================
# DemonAttackVisualizer Class - With Relationship Support
# ============================================================================

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image


class DemonAttackVisualizer:
    """Class to handle DemonAttack game visualization with gaze data, object detection, and relationships."""
    
    def __init__(self, data_folder, txt_file):
        self.data_folder = data_folder
        self.txt_file = txt_file
        self.df = None
        self.previous_gaze_points = None
        self.should_exit = False
        self._load_data()
    
    def _load_data(self):
        """Load data from CSV file."""
        data = []
        with open(self.txt_file, 'r') as f:
            header = f.readline().strip().split(',')
            for line in f:
                parts = line.strip().split(',')
                if len(parts) >= 6:
                    data.append({
                        'qframe_id': parts[0],
                        'episode_id': parts[1],
                        'score': parts[2],
                        'duration(ms)': parts[3],
                        'unclipped_reward': parts[4],
                        'action': parts[5],
                        'gaze_positions': ','.join(parts[6:])
                    })
        
        self.df = pd.DataFrame(data)
        self.df['duration(ms)'] = pd.to_numeric(self.df['duration(ms)'], errors='coerce')
        self.df['action'] = pd.to_numeric(self.df['action'], errors='coerce')
        self.df['score'] = pd.to_numeric(self.df['score'], errors='coerce')
        
        print(f"Loaded {len(self.df)} frames")
    
    def parse_gaze_positions(self, gaze_str):
        """Parse gaze positions from string."""
        try:
            if pd.isna(gaze_str) or str(gaze_str).strip().lower() in ['null', 'nan', '']:
                return None
            
            values = []
            for x in str(gaze_str).split(','):
                x = x.strip()
                if x and x.lower() not in ['null', 'nan']:
                    try:
                        values.append(float(x))
                    except ValueError:
                        continue
            
            gaze_points = []
            for i in range(0, len(values), 2):
                if i + 1 < len(values):
                    gaze_points.append((values[i], values[i+1]))
            
            return gaze_points if gaze_points else None
        
        except Exception as e:
            print(f"Error parsing gaze positions: {e}")
            return None
    
    def on_key(self, event):
        """Handle keyboard events."""
        if event.key == ' ':
            plt.close()
        elif event.key == 'escape':
            self.should_exit = True
            plt.close()
    
    def visualize(self, detector=None, show_objects=True, show_gaze=True, show_relationships=False):
        """
        Visualize frames with optional object detection, gaze data, and print relationships to terminal.
        """
        print(f"\nControls: Press SPACE for next frame, ESC to exit")
        print(f"Object detection: {'ON' if show_objects and detector else 'OFF'}")
        print(f"Gaze visualization: {'ON' if show_gaze else 'OFF'}")
        print(f"Relationships: {'ON (Terminal)' if show_relationships else 'OFF'}\n")
        
        analyzer = DemonAttackRelationshipAnalyzer() if show_relationships else None
        
        # Action names for DemonAttack - CORRECTED MAPPING
        ACTION_NAMES = {
            0: "NOOP",
            1: "FIRE",
            2: "RIGHT",  # This might be wrong in your data
            3: "RIGHT",  # Corrected
            4: "LEFT",   # Corrected
            5: "LEFTFIRE"
        }
        
        for idx, row in self.df.iterrows():
            if self.should_exit:
                print("\nExiting visualization...")
                break
            
            frame_id = row['qframe_id']
            gaze_positions = row['gaze_positions']
            numeric_id = frame_id.split('_')[-1]
            
            # Adjust frame filename pattern as needed
            frame_filename = f"{frame_id}.png"
            frame_path = os.path.join(self.data_folder, frame_filename)
            
            if not os.path.exists(frame_path):
                print(f"Frame {frame_filename} not found, skipping...")
                continue
            
            img = Image.open(frame_path)
            
            # Parse gaze data
            gaze_points = self.parse_gaze_positions(gaze_positions)
            use_previous = False
            
            if gaze_points is None or len(gaze_points) < 45:
                if self.previous_gaze_points is not None and len(self.previous_gaze_points) >= 45:
                    gaze_points = self.previous_gaze_points
                    use_previous = True
                    print(f"Frame {numeric_id}: Using previous gaze data ({len(gaze_points)} points)")
                else:
                    print(f"Frame {numeric_id}: Insufficient gaze data")
            else:
                self.previous_gaze_points = gaze_points
                print(f"Frame {numeric_id}: {len(gaze_points)} gaze points")
            
            # Create visualization
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111)
            ax.imshow(img)
            
            obj_count = 0
            relationships = []
            if show_objects and detector is not None:
                obj_count, detected_objects = self._add_object_bboxes(ax, frame_path, detector, return_objects=True)
                
                if show_relationships and analyzer:
                    relationships = analyzer.analyze_all_relationships(detected_objects)
                    self._print_relationships_to_terminal(frame_id, detected_objects, relationships, analyzer)
            
            if show_gaze and gaze_points is not None and len(gaze_points) > 0:
                self._add_gaze_points(ax, gaze_points, use_previous)
            
            # Build title
            action = int(row['action']) if pd.notna(row['action']) else 0
            action_name = ACTION_NAMES.get(action, f"Action {action}")
            duration = row['duration(ms)']
            score = row['score']
            gaze_count = len(gaze_points) if gaze_points else 0
            status = " (prev)" if use_previous else ""
            
            title_parts = [
                f"DemonAttack - Frame {frame_id}",
                f"Score: {score}",
                f"Duration: {duration}ms",
                f"Action: {action_name}"
            ]
            
            if show_gaze:
                title_parts.append(f"Gaze: {gaze_count}{status}")
            if show_objects and detector:
                title_parts.append(f"Objects: {obj_count}")
            if show_relationships and relationships:
                title_parts.append(f"Relations: {len(relationships)}")
            
            ax.set_title(" | ".join(title_parts), fontsize=9)
            ax.axis('off')
            
            controls = 'Press SPACE for next frame | Press ESC to exit'
            if use_previous:
                controls += ' | Orange = Previous Frame Data'
            fig.text(0.5, 0.02, controls, ha='center', fontsize=10, style='italic')
            
            plt.tight_layout(rect=[0, 0.03, 1, 0.97])
            fig.canvas.mpl_connect('key_press_event', self.on_key)
            plt.show()
            
            if idx % 5 == 0:
                print(f"Processed {idx+1}/{len(self.df)} frames")
        
        print("\nVisualization complete!")
    
    def _print_relationships_to_terminal(self, frame_id, detected_objects, relationships, analyzer):
        """Print relationships to terminal in a readable format."""
        print(f"\n{'='*80}")
        print(f"FRAME: {frame_id}")
        print(f"{'='*80}")
        
        # Print player info
        players = detected_objects.get('player', [])
        if players:
            player = players[0]
            player_x, player_y = player.center
            print(f"\n🎮 PLAYER: Position=({player_x:.1f}, {player_y:.1f})")
        
        # Print enemy count
        enemy_count = len(detected_objects.get('enemy', []))
        print(f"👾 ENEMIES: {enemy_count} total")
        
        # Print projectile counts
        friendly_count = len(detected_objects.get('projectile_friendly', []))
        hostile_count = len(detected_objects.get('projectile_hostile', []))
        print(f"🔫 FRIENDLY MISSILES: {friendly_count}")
        print(f"💥 HOSTILE MISSILES: {hostile_count} groups")
        
        # Print relationships
        if relationships:
            print(f"\n📊 RELATIONSHIPS ({len(relationships)} total):")
            print(f"{'-'*80}")
            
            # Group by type
            rel_by_type = {}
            for rel in relationships:
                rel_type = rel.relationship_type
                if rel_type not in rel_by_type:
                    rel_by_type[rel_type] = []
                rel_by_type[rel_type].append(rel)
            
            for rel_type in sorted(rel_by_type.keys()):
                rels = rel_by_type[rel_type]
                print(f"\n  {rel_type} ({len(rels)}):")
                for rel in rels:
                    formatted = analyzer.game_config.format_relationship_description(rel)
                    
                    # Add position info for better understanding
                    if rel.obj2:  # Two-object relationships
                        obj = rel.obj2
                        obj_x, obj_y = obj.center
                        print(f"    • {formatted} [pos=({obj_x:.1f}, {obj_y:.1f})]")
                    else:  # First occurrence relationships
                        obj = rel.obj1
                        obj_x, obj_y = obj.center
                        print(f"    • {formatted} [pos=({obj_x:.1f}, {obj_y:.1f})]")
            
            # Print formatted string for dataframe
            print(f"\n📝 FORMATTED OUTPUT:")
            formatted_str = analyzer.format_relationships_for_dataframe(relationships)
            print(f"  {formatted_str}")
        else:
            print(f"\n📊 RELATIONSHIPS: None detected")
        
        print(f"{'='*80}\n")
    
    def _add_object_bboxes(self, ax, frame_path, detector, return_objects=False):
        """Add object bounding boxes with labels to axis."""
        import cv2
        
        try:
            from .config import VISUALIZATION_COLORS
        except ImportError:
            try:
                from config import VISUALIZATION_COLORS
            except ImportError:
                from env.demonattack.config import VISUALIZATION_COLORS
        
        image = cv2.imread(str(frame_path))
        if image is None:
            return (0, {}) if return_objects else 0
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        detected_objects = detector.detect_all_objects(image_rgb)
        
        obj_count = 0
        
        for object_type, objects in detected_objects.items():
            if not objects:
                continue
            
            color_bgr = VISUALIZATION_COLORS.get(object_type, (255, 255, 255))
            color_rgb = (color_bgr[2]/255, color_bgr[1]/255, color_bgr[0]/255)
            
            for obj in objects:
                obj_count += 1
                x, y, w, h = obj.xywh
                
                rect = Rectangle((x, y), w, h, linewidth=2, 
                               edgecolor=color_rgb, facecolor='none')
                ax.add_patch(rect)
                
                label = object_type
                if hasattr(obj, 'characteristics') and obj.characteristics:
                    if 'size' in obj.characteristics:
                        label += f" ({obj.characteristics['size']})"
                    if 'grouped' in obj.characteristics:
                        label = "hostile_missiles"
                
                ax.text(x, y - 2, label, fontsize=8, color='white',
                       bbox=dict(facecolor=color_rgb, alpha=0.7, 
                                edgecolor='none', pad=1),
                       verticalalignment='bottom')
        
        if return_objects:
            return obj_count, detected_objects
        return obj_count
    
    def _add_gaze_points(self, ax, gaze_points, use_previous):
        """Add gaze points to axis."""
        x_coords = [point[0] for point in gaze_points]
        y_coords = [point[1] for point in gaze_points]
        
        color = 'orange' if use_previous else 'red'
        
        ax.scatter(x_coords, y_coords, c=color, s=30, alpha=0.6,
                  edgecolors='yellow', linewidths=1)
        ax.plot(x_coords, y_coords, color=color, alpha=0.3, linewidth=1)


# ============================================================================
# Testing and Single Frame Detection
# ============================================================================

def test_detection(image_path, save_path=None, bbox_output_path=None):
    """Quick test function for object detection visualization."""
    import cv2
    import json
    
    try:
        from .config import VISUALIZATION_COLORS
    except ImportError:
        try:
            from config import VISUALIZATION_COLORS
        except ImportError:
            from env.demonattack.config import VISUALIZATION_COLORS
    
    detector = DemonAttackObjectDetector()
    
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Error: Could not read image from {image_path}")
        return
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    detected_objects = detector.detect_all_objects(image_rgb)
    
    annotated = image.copy()
    bbox_data = []
    total_objects = 0
    
    for object_type, objects in detected_objects.items():
        if not objects:
            continue
        
        color = VISUALIZATION_COLORS.get(object_type, (255, 255, 255))
        
        for obj in objects:
            total_objects += 1
            x, y, w, h = obj.xywh
            x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
            
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            
            bbox_info = {
                'object_type': object_type,
                'object_id': obj.object_id,
                'x': int(x), 'y': int(y),
                'width': int(w), 'height': int(h),
                'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2
            }
            
            if hasattr(obj, 'characteristics') and obj.characteristics:
                for key, value in obj.characteristics.items():
                    bbox_info[key] = value
            
            bbox_data.append(bbox_info)
    
    print(f"\n=== Detection Summary ===")
    print(f"Total objects: {total_objects}")
    print(f"Player: {len(detected_objects.get('player', []))}")
    print(f"Enemies: {len(detected_objects.get('enemy', []))}")
    print(f"Friendly projectiles: {len(detected_objects.get('projectile_friendly', []))}")
    print(f"Hostile projectile groups: {len(detected_objects.get('projectile_hostile', []))}")
    
    if bbox_output_path:
        if bbox_output_path.endswith('.json'):
            with open(bbox_output_path, 'w') as f:
                json.dump(bbox_data, f, indent=2)
            print(f"\nBounding boxes saved to: {bbox_output_path} (JSON)")
        else:
            df = pd.DataFrame(bbox_data)
            df.to_csv(bbox_output_path, index=False)
            print(f"\nBounding boxes saved to: {bbox_output_path} (CSV)")
    
    if save_path:
        cv2.imwrite(str(save_path), annotated)
        print(f"Visualization saved to: {save_path}")
    
    cv2.imshow('DemonAttack Detection Test', annotated)
    print("\nPress any key to close...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return annotated, detected_objects, bbox_data


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--visualize':
        if len(sys.argv) < 4:
            print("Usage: python object_detector.py --visualize <data_folder> <txt_file> [--relationships]")
            sys.exit(1)
        
        data_folder = sys.argv[2]
        txt_file = sys.argv[3]
        show_relationships = '--relationships' in sys.argv
        
        print(f"\n{'='*60}")
        print(f"DemonAttack Visualization with Object Detection")
        print(f"{'='*60}")
        print(f"Data folder: {data_folder}")
        print(f"Data file: {txt_file}")
        print(f"Relationships: {'ON' if show_relationships else 'OFF'}")
        print(f"{'='*60}")
        
        detector = DemonAttackObjectDetector()
        visualizer = DemonAttackVisualizer(data_folder, txt_file)
        visualizer.visualize(detector=detector, show_objects=True, show_gaze=True, 
                           show_relationships=show_relationships)
    
    else:
        if len(sys.argv) > 1:
            image_path = sys.argv[1]
        else:
            image_path = "demonattack_frame.png"
        
        save_path = sys.argv[2] if len(sys.argv) > 2 else "detection_test.png"
        bbox_path = sys.argv[3] if len(sys.argv) > 3 else "bounding_boxes.csv"
        
        print(f"\n{'='*60}")
        print(f"DemonAttack Object Detection Test")
        print(f"{'='*60}")
        print(f"Image: {image_path}")
        print(f"Output: {save_path}")
        print(f"BBox Data: {bbox_path}")
        print(f"{'='*60}")
        
        try:
            test_detection(image_path, save_path=save_path, bbox_output_path=bbox_path)
        except Exception as e:
            print(f"\nError during testing: {e}")
            import traceback
            traceback.print_exc()
