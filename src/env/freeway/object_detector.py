"""
Freeway-specific object detector implementation with integrated visualization and relationships.
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
    from .config import OBJECT_COLORS, CAR_COLORS, DETECTION_PARAMS, LANE_POSITIONS
except ImportError:
    try:
        from config import OBJECT_COLORS, CAR_COLORS, DETECTION_PARAMS, LANE_POSITIONS
    except ImportError:
        from env.freeway.config import OBJECT_COLORS, CAR_COLORS, DETECTION_PARAMS, LANE_POSITIONS

from models.ocatari.ocatari.vision.utils import find_objects

# Import relationship analyzer
try:
    from .relationship_analyzer import FreewayRelationshipAnalyzer
except ImportError:
    try:
        from relationship_analyzer import FreewayRelationshipAnalyzer
    except ImportError:
        from env.freeway.relationship_analyzer import FreewayRelationshipAnalyzer


class FreewayGameConfig:
    """Game configuration for Freeway."""
    
    def __init__(self):
        self.object_colors = OBJECT_COLORS
        self.car_colors = CAR_COLORS
        self.detection_params = DETECTION_PARAMS
        self.lane_positions = LANE_POSITIONS
    
    def get_object_types(self) -> List[str]:
        """Return list of object types for Freeway."""
        return [
            'chicken', 'player_score', 'enemy_score',
            'car1', 'car2', 'car3', 'car4', 'car5',
            'car6', 'car7', 'car8', 'car9', 'car10'
        ]


class FreewayObjectDetector(BaseObjectDetector):
    """Freeway-specific object detector with custom detection logic and object tracking."""
    
    def __init__(self):
        """Initialize with Freeway configuration."""
        super().__init__(FreewayGameConfig())
        
        max_objects = {
            'chicken': 2,
            'player_score': 1,
            'enemy_score': 1,
            'car1': 1, 'car2': 1, 'car3': 1, 'car4': 1, 'car5': 1,
            'car6': 1, 'car7': 1, 'car8': 1, 'car9': 1, 'car10': 1
        }
        
        self.object_tracker = ObjectTracker(max_objects)
        self.use_tracking = True
    
    def detect_objects_by_type(self, image, object_type):
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
                object_id=f'{object_type}'
            )
            objects.append(obj)
        
        return objects
    
    def detect_all_objects(self, image: np.ndarray) -> Dict[str, List[GameObject]]:
        detected_objects = {}
        detected_objects['chicken'] = self._detect_chickens(image)
        detected_objects['player_score'] = self.detect_objects_by_type(image, 'player_score')
        detected_objects['enemy_score'] = self.detect_objects_by_type(image, 'enemy_score')
        
        for lane_num in range(1, 11):
            car_type = f'car{lane_num}'
            detected_objects[car_type] = self._detect_cars_in_lane(image, lane_num)
        
        self._cleanup_detections(detected_objects)
        
        if self.use_tracking:
            detected_objects = self.object_tracker.track_all_objects(detected_objects)
        for lane_num in range(1, 11):
            car_type = f'car{lane_num}'
            if car_type in detected_objects and detected_objects[car_type]:
                for car in detected_objects[car_type]:
                    car.object_id = car_type  # Force ID to be just car_type
        
        return detected_objects
        
        return detected_objects
    
    def _detect_chickens(self, image: np.ndarray) -> List[GameObject]:
        chickens = []
        colors = self.game_config.object_colors['chicken']
        
        chicken1_coords = find_objects(image, colors, size=(7, 8), tol_s=3, maxx=80)
        chicken2_coords = find_objects(image, colors, size=(7, 8), tol_s=3, minx=80)
        
        for i, coords in enumerate(chicken1_coords):
            chicken = GameObject('chicken', coords, object_id=f'chicken_p1_{i}',
                                characteristics={'player': 1, 'color_rgb': [252, 252, 84]})
            chickens.append(chicken)
        
        for i, coords in enumerate(chicken2_coords):
            chicken = GameObject('chicken', coords, object_id=f'chicken_p2_{i}',
                                characteristics={'player': 2, 'color_rgb': [252, 252, 84]})
            chickens.append(chicken)
        
        return chickens
    
    def _detect_cars_in_lane(self, image: np.ndarray, lane_num: int) -> List[GameObject]:
        car_type = f'car{lane_num}'
        
        if car_type not in self.game_config.car_colors:
            return []
        
        color = self.game_config.car_colors[car_type]
        params = self.game_config.detection_params.get(car_type, {})
        car_coords = find_objects(image, color, **params)
        
        if car_coords:
            coords = car_coords[0]
            direction = 'left' if lane_num <= 5 else 'right'
            
            car = GameObject(car_type, coords, object_id=f'{car_type}',
                           characteristics={'lane': lane_num, 'direction': direction, 'color_rgb': color})
            return [car]
        
        return []
    
    def _cleanup_detections(self, detected_objects: Dict[str, List[GameObject]]):
        if detected_objects.get('chicken'):
            detected_objects['chicken'] = [c for c in detected_objects['chicken'] if 18 <= c.top <= 188]
        
        for lane_num in range(1, 11):
            car_type = f'car{lane_num}'
            if detected_objects.get(car_type):
                cars = detected_objects[car_type]
                valid_cars = [car for car in cars if car.width >= 4 and car.height >= 4]
                if len(valid_cars) > 1:
                    valid_cars = [max(valid_cars, key=lambda c: c.width * c.height)]
                detected_objects[car_type] = valid_cars
    
    def get_all_cars(self, detected_objects: Dict[str, List[GameObject]]) -> List[GameObject]:
        all_cars = []
        for lane_num in range(1, 11):
            car_type = f'car{lane_num}'
            if car_type in detected_objects:
                all_cars.extend(detected_objects[car_type])
        return all_cars


# ============================================================================
# FreewayVisualizer Class - With Relationship Support
# ============================================================================

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch, Circle
from PIL import Image


class FreewayVisualizer:
    """Class to handle Freeway game visualization with gaze data and object detection."""
    
    def __init__(self, data_folder, txt_file):
        self.data_folder = data_folder
        self.txt_file = txt_file
        self.df = None
        self.previous_gaze_points = None
        self.should_exit = False
        self._load_data()
    
    def _load_data(self):
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
        print(f"Loaded {len(self.df)} frames")
    
    def parse_gaze_positions(self, gaze_str):
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
        
        analyzer = FreewayRelationshipAnalyzer() if show_relationships else None
        
        for idx, row in self.df.iterrows():
            if self.should_exit:
                print("\nExiting visualization...")
                break
            
            frame_id = row['qframe_id']
            gaze_positions = row['gaze_positions']
            
            numeric_id = frame_id.split('_')[-1]
            frame_filename = f"RZ_2464601_{numeric_id}.png"
            frame_path = os.path.join(self.data_folder, frame_filename)
            
            if not os.path.exists(frame_path):
                print(f"Frame {frame_filename} not found, skipping...")
                continue
            
            img = Image.open(frame_path)
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
            
            action = row['action']
            duration = row['duration(ms)']
            gaze_count = len(gaze_points) if gaze_points else 0
            status = " (prev)" if use_previous else ""
            
            title_parts = [f"Frame {frame_id}", f"Duration: {duration}ms", f"Action: {action}"]
            if show_gaze:
                title_parts.append(f"Gaze: {gaze_count}{status}")
            if show_objects and detector:
                title_parts.append(f"Objects: {obj_count}")
            if show_relationships and relationships:
                title_parts.append(f"Relations: {len(relationships)}")
            
            ax.set_title(" | ".join(title_parts), fontsize=10)
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
        
        chickens = detected_objects.get('chicken', [])
        chicken = None
        for c in chickens:
            if hasattr(c, 'characteristics') and c.characteristics.get('player') == 1:
                chicken = c
                break
        
        if chicken:
            chicken_x, chicken_y = chicken.center
            print(f"\n🐔 CHICKEN (Player 1): Position=({chicken_x:.1f}, {chicken_y:.1f})")
        
        car_count = 0
        for lane_num in range(1, 11):
            car_type = f'car{lane_num}'
            cars = detected_objects.get(car_type, [])
            car_count += len(cars)
        
        print(f"🚗 CARS: {car_count} total")
        
        if relationships:
            print(f"\n📊 RELATIONSHIPS ({len(relationships)} total):")
            print(f"{'-'*80}")
            
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
                    if rel.obj2.object_type.startswith('car'):
                        car = rel.obj2
                        car_x, car_y = car.center
                        lane = car.characteristics.get('lane', '?')
                        direction = car.characteristics.get('direction', '?')
                        print(f"    • {formatted} [Lane {lane}, {direction}, pos=({car_x:.1f}, {car_y:.1f})]")
                    else:
                        print(f"    • {formatted}")
            
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
                from env.freeway.config import VISUALIZATION_COLORS
        
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
                if object_type == 'chicken' and hasattr(obj, 'characteristics'):
                    if obj.characteristics.get('player') == 2:
                        continue
                
                obj_count += 1
                x, y, w, h = obj.xywh
                
                rect = Rectangle((x, y), w, h, linewidth=2, edgecolor=color_rgb, facecolor='none')
                ax.add_patch(rect)
                
                label = object_type
                if hasattr(obj, 'characteristics') and obj.characteristics:
                    if 'player' in obj.characteristics:
                        label += f" P{obj.characteristics['player']}"
                    if 'lane' in obj.characteristics:
                        label += f" L{obj.characteristics['lane']}"
                    if 'direction' in obj.characteristics:
                        label += f" ({obj.characteristics['direction'][0].upper()})"
                
                ax.text(x, y - 2, label, fontsize=8, color='white',
                       bbox=dict(facecolor=color_rgb, alpha=0.7, edgecolor='none', pad=1),
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
                  edgecolors='white', linewidths=1)
        ax.plot(x_coords, y_coords, color=color, alpha=0.3, linewidth=1)



# ============================================================================
# Testing and Single Frame Detection
# ============================================================================

def test_detection(image_path, show_lanes=False, save_path=None, bbox_output_path=None):
    """Quick test function for object detection visualization."""
    import cv2
    import json
    
    try:
        from .config import VISUALIZATION_COLORS, LANE_POSITIONS
    except ImportError:
        try:
            from config import VISUALIZATION_COLORS, LANE_POSITIONS
        except ImportError:
            from env.freeway.config import VISUALIZATION_COLORS, LANE_POSITIONS
    
    detector = FreewayObjectDetector()
    
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Error: Could not read image from {image_path}")
        return
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    detected_objects = detector.detect_all_objects(image_rgb)
    
    annotated = image.copy()
    
    if show_lanes:
        for lane_num, (miny, maxy) in LANE_POSITIONS.items():
            cv2.line(annotated, (0, miny), (annotated.shape[1], miny), (100, 100, 100), 1)
    
    bbox_data = []
    total_objects = 0
    car_count = 0
    
    for object_type, objects in detected_objects.items():
        if not objects:
            continue
        
        color = VISUALIZATION_COLORS.get(object_type, (255, 255, 255))
        
        for obj in objects:
            if object_type == 'chicken' and hasattr(obj, 'characteristics'):
                if obj.characteristics.get('player') == 2:
                    continue
            
            total_objects += 1
            if object_type.startswith('car'):
                car_count += 1
            
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
    print(f"Cars detected: {car_count}")
    
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
    
    cv2.imshow('Freeway Detection Test', annotated)
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
        print(f"Freeway Visualization with Object Detection")
        print(f"{'='*60}")
        print(f"Data folder: {data_folder}")
        print(f"Data file: {txt_file}")
        print(f"Relationships: {'ON' if show_relationships else 'OFF'}")
        print(f"{'='*60}")
        
        detector = FreewayObjectDetector()
        visualizer = FreewayVisualizer(data_folder, txt_file)
        visualizer.visualize(detector=detector, show_objects=True, show_gaze=True, 
                           show_relationships=show_relationships)
    
    else:
        if len(sys.argv) > 1:
            image_path = sys.argv[1]
        else:
            image_path = "RZ_2494228_2754.png"
        
        save_path = sys.argv[2] if len(sys.argv) > 2 else "detection_test.png"
        bbox_path = sys.argv[3] if len(sys.argv) > 3 else "bounding_boxes.csv"
        
        print(f"\n{'='*60}")
        print(f"Freeway Object Detection Test")
        print(f"{'='*60}")
        print(f"Image: {image_path}")
        print(f"Output: {save_path}")
        print(f"BBox Data: {bbox_path}")
        print(f"{'='*60}")
        
        try:
            test_detection(image_path, show_lanes=False, save_path=save_path, bbox_output_path=bbox_path)
        except Exception as e:
            print(f"\nError during testing: {e}")
            import traceback
            traceback.print_exc()
