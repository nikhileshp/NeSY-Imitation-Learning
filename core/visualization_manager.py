"""
Visualization manager module for rendering game objects, relationships, and gaze data.
"""
import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
from models.OC_Atari.ocatari.vision.utils import mark_bb
from .game_object import GameObject, SpatialRelationship
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
    
    
    def update_color_mapping(self, new_mapping: Dict[str, Tuple[int, int, int]]):
        """
        Update the object color mapping.
        
        Args:
            new_mapping: New dictionary mapping object types to BGR colors
        """
        self.object_color_mapping.update(new_mapping)
    
    def draw_all_objects(self, image: np.ndarray, 
                        detected_objects: Dict[str, List[GameObject]]) -> np.ndarray:
        """
        Draw bounding boxes around all detected objects.
        
        Args:
            image: Input image as numpy array
            detected_objects: Dictionary mapping object types to GameObjects
            
        Returns:
            Image with bounding boxes drawn
        """
        # Create a copy to avoid modifying the original image
        annotated_image = image.copy()
        
        # Draw bounding boxes for each object type
        for object_type, objects in detected_objects.items():
            color = self.object_color_mapping.get(object_type, (255, 255, 255))
            
            for game_object in objects:
                mark_bb(annotated_image, game_object.bounding_box, color=color)
        
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
            
            # Draw line between centers
            line_color = self.base_colors.get('relationship_line', (0, 0, 0))
            cv2.line(annotated_image, center1, center2, line_color, 1)
            
            # Create relationship text
            relationships_text = '-'.join([rel[0].upper() for rel in relationships])
            
            # Calculate midpoint for text placement
            mid_point = ((center1[0] + center2[0]) // 2, (center1[1] + center2[1]) // 2)
            
            # Draw relationship text
            text_color = self.base_colors.get('relationship_text', (0, 0, 0))
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
                                         scale_factor: int = 2) -> np.ndarray:
        """
        Create a comprehensive visualization with all elements.
        
        Args:
            image: Input image as numpy array
            detected_objects: Dictionary mapping object types to GameObjects
            connection_list: List of connection dictionaries
            gaze_positions: List of (x, y) gaze position tuples
            scale_factor: Factor by which to scale the output image
            
        Returns:
            Comprehensive annotated image
        """
        # Start with the original image
        annotated_image = image.copy()
        
        # Get image dimensions
        height, width = image.shape[:2]
        
        # Draw all objects with bounding boxes
        annotated_image = self.draw_all_objects(annotated_image, detected_objects)
        
        # Draw relationships
        annotated_image = self.draw_relationships(annotated_image, connection_list)
        
        # Draw gaze positions
        annotated_image = self.draw_gaze_positions(annotated_image, gaze_positions, 
                                                  width, height)
        
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
        Display an image in an OpenCV window.
        
        Args:
            image: Image to display
            window_name: Name of the display window
            wait_for_key: Whether to wait for a key press
            
        Returns:
            Key code if wait_for_key is True, otherwise -1
        """
        cv2.imshow(window_name, image)
        
        if wait_for_key:
            key = cv2.waitKey(0)
            return key
        else:
            cv2.waitKey(1)
            return -1
    
    def close_all_windows(self):
        """Close all OpenCV windows."""
        cv2.destroyAllWindows()
    
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
