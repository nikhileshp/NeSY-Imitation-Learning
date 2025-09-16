"""
Visualization manager module for rendering game objects, relationships, and gaze data.
"""
import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
from models.OC_Atari.ocatari.vision.utils import mark_bb
from core.game_object import GameObject, SpatialRelationship
from .config import BASE_VISUALIZATION_COLORS, DEFAULT_OBJECT_COLORS


class VisualizationManager:
    """Handles visualization of game objects, relationships, and gaze data."""
    
    def __init__(self, object_color_mapping: Optional[Dict[str, Tuple[int, int, int]]] = None):
        """
        Initialize the visualization manager.
        
        Args:
            object_color_mapping: Optional dictionary mapping object types to BGR colors.
                                If None, uses default Seaquest colors from config.
        """
        self.base_colors = BASE_VISUALIZATION_COLORS
        self.object_color_mapping = object_color_mapping or DEFAULT_OBJECT_COLORS.copy()
        self.main_window_name = "Game Analysis"
        self.window_initialized = False
    
    
    def update_color_mapping(self, new_mapping: Dict[str, Tuple[int, int, int]]):
        """
        Update the object color mapping.
        
        Args:
            new_mapping: New dictionary mapping object types to BGR colors
        """
        self.object_color_mapping.update(new_mapping)
    
    def draw_all_objects(self, image: np.ndarray, 
                        detected_objects: Dict[str, List[GameObject]], 
                        relationships: Optional[List[SpatialRelationship]] = None) -> np.ndarray:
        """
        Draw bounding boxes around all detected objects with enhanced features.
        
        Args:
            image: Input image as numpy array
            detected_objects: Dictionary mapping object types to GameObjects
            relationships: Optional list of relationships for directional arrows
            
        Returns:
            Image with bounding boxes, indexes, and directional arrows drawn
        """
        # Create a copy to avoid modifying the original image
        annotated_image = image.copy()
        
        # Draw bounding boxes for each object type
        for object_type, objects in detected_objects.items():
            color = self.object_color_mapping.get(object_type, (255, 255, 255))
            
            if not objects:
                continue
                
            for game_object in objects:
                # Draw bounding box
                self._draw_bounding_box_with_index(annotated_image, game_object, color)
                
                # Draw directional arrow if object has facing side relationship OR facing_side characteristic
                if relationships:
                    self._draw_directional_arrow(annotated_image, game_object, relationships)
                elif hasattr(game_object, 'characteristics') and 'facing_side' in game_object.characteristics:
                    # Draw arrow directly from object characteristics
                    self._draw_arrow_for_direction(annotated_image, game_object, game_object.characteristics['facing_side'])
        
        return annotated_image
    
    def draw_relationships(self, image: np.ndarray, 
                          connection_list: List[Dict]) -> np.ndarray:
        """
        Draw lines and labels representing relationships between objects.
        
        Args:
            image: Input image as numpy array
            connection_list: List of connection dictionaries
            
        Returns:
            Image with relationship lines and labels drawn
        """
        annotated_image = image.copy()
        
        for connection in connection_list:
            obj1 = connection['obj1']
            obj2 = connection['obj2']
            relationships = connection['relationships']
            
            # Calculate centers of both objects
            center1 = obj1.center
            center2 = obj2.center
            
            # Determine line color based on relationship type
            line_color = self.base_colors.get('relationship_line', (0, 0, 0))
            
            # Use green line for nearby relationships
            if 'nearby' in relationships:
                line_color = self.base_colors.get('nearby_line', (0, 255, 0))  # Green
            
            # Draw line between centers
            cv2.line(annotated_image, center1, center2, line_color, 1)
            
            # Create relationship text
            relationships_text = '-'.join([rel[0].upper() for rel in relationships])
            
            # Calculate midpoint for text placement
            mid_point = ((center1[0] + center2[0]) // 2, (center1[1] + center2[1]) // 2)
            
            # Draw relationship text
            text_color = self.base_colors.get('relationship_text', (255,255,255))
            cv2.putText(annotated_image, relationships_text, mid_point, 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, text_color, 1)
        
        return annotated_image
    
    def draw_gaze_positions(self, image: np.ndarray, 
                           gaze_positions: List[Tuple[int, int]], 
                           image_width: int, image_height: int) -> np.ndarray:
        """
        Draw gaze positions as red dots on the image.
        
        Args:
            image: Input image as numpy array
            gaze_positions: List of (x, y) gaze position tuples
            image_width: Width of the original image
            image_height: Height of the original image
            
        Returns:
            Image with gaze positions drawn
        """
        annotated_image = image.copy()
        gaze_color = self.base_colors.get('gaze_position', (0, 0, 255))
        
        for x, y in gaze_positions:
            # Check if gaze position is within image bounds
            if 0 <= x < image_width and 0 <= y < image_height:
                cv2.circle(annotated_image, (x, y), 1, gaze_color, -1)
        
        return annotated_image
    
    def create_comprehensive_visualization(self, image: np.ndarray,
                                         detected_objects: Dict[str, List[GameObject]],
                                         connection_list: List[Dict],
                                         gaze_positions: List[Tuple[int, int]],
                                         relationships: Optional[List[SpatialRelationship]] = None,
                                         scale_factor: int = 2,
                                         detected_goal: str = "") -> np.ndarray:
        """
        Create a comprehensive visualization with all elements.
        
        Args:
            image: Input image as numpy array
            detected_objects: Dictionary mapping object types to GameObjects
            connection_list: List of connection dictionaries
            gaze_positions: List of (x, y) gaze position tuples
            relationships: Optional list of relationships for enhanced features
            scale_factor: Factor by which to scale the output image
            detected_goal: Detected goal text to display on frame
            
        Returns:
            Comprehensive annotated image
        """
        # Start with the original image
        annotated_image = image.copy()
        
        # Get image dimensions
        height, width = image.shape[:2]
        
        # Draw all objects with bounding boxes, indexes, and directional arrows
        annotated_image = self.draw_all_objects(annotated_image, detected_objects, relationships)
        
        # Draw relationships
        annotated_image = self.draw_relationships(annotated_image, connection_list)
        
        # Draw gaze positions
        annotated_image = self.draw_gaze_positions(annotated_image, gaze_positions, 
                                                  width, height)
        
        # Add goal text to top left corner
        if detected_goal:
            self._draw_goal_text(annotated_image, detected_goal)
        
        # Scale up the image for better visibility
        if scale_factor > 1:
            new_width = width * scale_factor
            new_height = height * scale_factor
            annotated_image = cv2.resize(annotated_image, (new_width, new_height), 
                                       interpolation=cv2.INTER_NEAREST)
        
        return annotated_image
    
    def display_image(self, image: np.ndarray, window_name: str = 'Frame', 
                     wait_for_key: bool = True) -> int:
        """
        Display an image in the same OpenCV window, reusing the window if it exists.
        
        Args:
            image: Image to display
            window_name: Name of the display window (updated to show frame info)
            wait_for_key: Whether to wait for a key press
            
        Returns:
            Key code if wait_for_key is True, otherwise -1
        """
        # Initialize window with specific properties if not already done
        if not self.window_initialized:
            cv2.namedWindow(self.main_window_name, cv2.WINDOW_AUTOSIZE)
            try:
                cv2.setWindowProperty(self.main_window_name, cv2.WND_PROP_TOPMOST, 1)
            except cv2.error:
                # Ignore if setting window property fails (some systems don't support it)
                pass
            self.window_initialized = True
        
        # Display the image in the same window first
        cv2.imshow(self.main_window_name, image)
        
        # Update window title to show current frame info (after image is shown)
        try:
            cv2.setWindowTitle(self.main_window_name, window_name)
        except cv2.error:
            # If setting title fails, continue without it
            pass
        
        if wait_for_key:
            key = cv2.waitKey(0)
            return key
        else:
            cv2.waitKey(1)
            return -1
    
    def close_all_windows(self):
        """Close all OpenCV windows and reset window state."""
        cv2.destroyAllWindows()
        self.window_initialized = False
    
    def close_main_window(self):
        """Close the main visualization window and reset its state."""
        if self.window_initialized:
            cv2.destroyWindow(self.main_window_name)
            self.window_initialized = False
    
    def add_object_labels(self, image: np.ndarray, 
                         detected_objects: Dict[str, List[GameObject]]) -> np.ndarray:
        """
        Add text labels to detected objects.
        
        Args:
            image: Input image as numpy array
            detected_objects: Dictionary mapping object types to GameObjects
            
        Returns:
            Image with object labels
        """
        annotated_image = image.copy()
        
        for object_type, objects in detected_objects.items():
            for game_object in objects:
                # Position label at the top-left of the bounding box
                label_pos = (game_object.x, max(game_object.y - 5, 10))
                
                # Use object ID as label
                label_text = game_object.object_id
                
                # Draw text with background for better visibility
                cv2.putText(annotated_image, label_text, label_pos,
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        
        return annotated_image
    
    def create_debug_visualization(self, image: np.ndarray,
                                 detected_objects: Dict[str, List[GameObject]],
                                 relationships: List[SpatialRelationship]) -> np.ndarray:
        """
        Create a debug visualization with detailed information.
        
        Args:
            image: Input image as numpy array
            detected_objects: Dictionary mapping object types to GameObjects
            relationships: List of SpatialRelationship objects
            
        Returns:
            Debug annotated image
        """
        annotated_image = image.copy()
        
        # Draw objects with labels
        annotated_image = self.draw_all_objects(annotated_image, detected_objects)
        annotated_image = self.add_object_labels(annotated_image, detected_objects)
        
        # Add debug information text
        y_offset = 20
        for object_type, objects in detected_objects.items():
            if objects:
                debug_text = f"{object_type}: {len(objects)}"
                cv2.putText(annotated_image, debug_text, (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                y_offset += 15
        
        # Add relationship count
        if relationships:
            rel_text = f"Relationships: {len(relationships)}"
            cv2.putText(annotated_image, rel_text, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        return annotated_image
    
    def _draw_bounding_box_with_index(self, image: np.ndarray, game_object: GameObject, color: Tuple[int, int, int]):
        """
        Draw bounding box with object index on the top-right corner.
        
        Args:
            image: Image to draw on
            game_object: GameObject to draw
            color: Bounding box color
        """
        # Draw the bounding box using the existing mark_bb function
        mark_bb(image, game_object.bounding_box, color=color)
        
        # Extract index from object_id (e.g., "enemy_0" -> "0")
        object_index = self._extract_object_index(game_object.object_id)
        
        if object_index is not None:
            # Position index at top-right corner of bounding box
            index_x = game_object.x + game_object.width - 8  # 8 pixels from right edge
            index_y = game_object.y + 12  # 12 pixels down from top
            
            # Ensure index is within image bounds
            index_x = max(0, min(index_x, image.shape[1] - 10))
            index_y = max(12, min(index_y, image.shape[0] - 5))
           
            # Draw index number
            cv2.putText(image, str(object_index), (index_x, index_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.2, (255, 255, 255), 1)
    
    def _extract_object_index(self, object_id: str) -> Optional[int]:
        """
        Extract the numeric index from an object ID.
        
        Args:
            object_id: Object ID string (e.g., "enemy_0", "player_1")
            
        Returns:
            Integer index or None if not found
        """
        try:
            # Split by underscore and get the last part
            parts = object_id.split('_')
            if len(parts) >= 2:
                return int(parts[-1])
        except (ValueError, IndexError):
            pass
        return None
    
    def _draw_directional_arrow(self, image: np.ndarray, game_object: GameObject, 
                               relationships: List[SpatialRelationship]):
        """
        Draw directional arrow for objects with facing side relationships.
        
        Args:
            image: Image to draw on
            game_object: GameObject to check for facing relationships
            relationships: List of spatial relationships
        """
        # Find facing side relationship for this object
        facing_direction = None
        for relationship in relationships:
            if (relationship.obj1 == game_object and 
                (relationship.relationship_type.startswith('facing') or 
                 relationship.relationship_type.startswith('enemyFacing'))):
                # Extract direction from relationship type
                if relationship.relationship_type.startswith('enemyFacing'):
                    # e.g., 'enemyFacingLeft' -> 'left'
                    direction_part = relationship.relationship_type.replace('enemyFacing', '').lower()
                else:
                    # e.g., 'facingLeft' -> 'left'
                    direction_part = relationship.relationship_type.replace('facing', '').lower()
                facing_direction = direction_part
                break
        
        if facing_direction:
            self._draw_arrow_for_direction(image, game_object, facing_direction)
    
    def _draw_arrow_for_direction(self, image: np.ndarray, game_object: GameObject, direction: str):
        """
        Draw an arrow indicating the facing direction.
        
        Args:
            image: Image to draw on
            game_object: GameObject to draw arrow for
            direction: Direction string ('left', 'right', 'up', 'down')
        """
        # Calculate arrow position (center of object)
        center_x, center_y = game_object.center
        
        # Arrow properties
        arrow_length = 15  # Length of the arrow
        arrow_color = (0, 0, 0)  # Black arrow
        arrow_thickness = 2
        
        # Calculate arrow end point based on direction
        if direction == 'left':
            end_x = center_x - arrow_length
            end_y = center_y
        elif direction == 'right':
            end_x = center_x + arrow_length
            end_y = center_y
        elif direction == 'up':
            end_x = center_x
            end_y = center_y - arrow_length
        elif direction == 'down':
            end_x = center_x
            end_y = center_y + arrow_length
        else:
            return  # Unknown direction
        
        # Draw arrow line
        cv2.arrowedLine(image, (center_x, center_y), (end_x, end_y), 
                       arrow_color, arrow_thickness, tipLength=0.3)
        
        # Add direction label near the arrow
        label_x = end_x + (5 if direction == 'right' else -15 if direction == 'left' else -8)
        label_y = end_y + (5 if direction == 'down' else -5 if direction == 'up' else 5)
        
        cv2.putText(image, direction[0].upper(), (label_x, label_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.2, (255,255,255), 1)
    
    def _draw_goal_text(self, image: np.ndarray, goal: str):
        """
        Draw the detected goal text on the top left of the image.
        
        Args:
            image: Image to draw on
            goal: Goal text to display
        """
        # Goal text properties
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.2
        thickness = 1
        
        # Format goal text
        goal_text = f"Goal: {goal}"
        
        # Get text size for background rectangle
        (text_width, text_height), baseline = cv2.getTextSize(goal_text, font, font_scale, thickness)
        
        # Position for top left corner with some padding
        x_pos = 10
        y_pos = text_height + 15  # 15 pixels from top
        
        # Draw background rectangle for better text visibility
        padding = 5
        cv2.rectangle(image, 
                     (x_pos - padding, y_pos - text_height - padding), 
                     (x_pos + text_width + padding, y_pos + baseline + padding),
                     (0, 0, 0, 128),  # Semi-transparent black background
                     -1)
        
        # Choose text color based on goal type
        text_colors = {
            'retrieve_diver': (0, 255, 255),    # Yellow
            'kill_enemy': (0, 0, 255),          # Red  
            'avoid_enemy': (0, 165, 255),       # Orange
            'surface': (0, 255, 0),             # Green
            'waitForOxygen': (255, 0, 255),     # Magenta
            'unknown': (128, 128, 128)          # Gray
        }
        
        text_color = text_colors.get(goal, (255, 255, 255))  # Default to white
        
        # Draw the goal text
        cv2.putText(image, goal_text, (x_pos, y_pos), font, font_scale, text_color, thickness)


def create_seaquest_visualization_manager() -> VisualizationManager:
    """
    Create a VisualizationManager configured for Seaquest game.
    
    Returns:
        VisualizationManager with Seaquest-specific color mapping
    """
    seaquest_colors = {
        'player': (0, 255, 0),      # Green
        'diver': (255, 0, 0),       # Blue  
        'collected_diver': (150, 0, 150),   # Purple
        'player_missile': (0, 255, 0),      # Green
        'enemy_missile': (0, 0, 255),       # Red
        'lives': (255, 255, 0),     # Cyan
        'enemy_submarine': (0, 0, 255),     # Red
        'oxygen_bar': (255, 0, 255),        # Magenta
        'oxygen_depleted': (100, 100, 100), # Gray
        'enemy': (0, 0, 255)        # Red
    }
    
    return VisualizationManager(seaquest_colors)


def create_custom_visualization_manager(color_mapping: Dict[str, Tuple[int, int, int]]) -> VisualizationManager:
    """
    Create a VisualizationManager with custom color mapping.
    
    Args:
        color_mapping: Dictionary mapping object types to BGR colors
        
    Returns:
        VisualizationManager with custom color mapping
    """
    return VisualizationManager(color_mapping)
