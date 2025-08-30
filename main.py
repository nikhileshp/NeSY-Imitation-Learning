"""
Main application module for game object detection and relationship analysis.
Updated to use the new modular structure with environment-specific configurations.
"""
import sys
import os
import cv2
import pandas as pd
from typing import List, Dict, Optional, Tuple

# Import core modules
from core.gaze_data_processor import GazeDataProcessor
from core.visualization_manager import VisualizationManager
from core.game_object import GameObject

# Import Seaquest-specific modules
from env.seaquest.object_detector import SeaquestObjectDetector
from env.seaquest.relationship_analyzer import SeaquestRelationshipAnalyzer
from env.seaquest.config import VISUALIZATION_COLORS


class GameAnalysisApp:
    """Main application class for game object detection and relationship analysis."""
    
    def __init__(self, game_type: str = "seaquest"):
        """
        Initialize the game analysis application.
        
        Args:
            game_type: Type of game ("seaquest" for now, extensible for other games)
        """
        if game_type == "seaquest":
            self.object_detector = SeaquestObjectDetector()
            self.relationship_analyzer = SeaquestRelationshipAnalyzer()
            self.visualizer = VisualizationManager(VISUALIZATION_COLORS)
        else:
            raise ValueError(f"Game type '{game_type}' not supported yet")
        
        self.gaze_processor = GazeDataProcessor()
        self.gaze_df = pd.DataFrame()
    
    def run(self, image_folder: str, output_video: str = "test_output.mp4", fps: int = 1):
        """
        Run the main analysis pipeline.
        
        Args:
            image_folder: Path to folder containing game images
            output_video: Output video filename (not currently used)
            fps: Processing frequency (process every fps-th image)
        """
        # Validate input folder
        if not os.path.exists(image_folder):
            raise FileNotFoundError(f"Image folder {image_folder} does not exist.")
        
        # Load gaze data
        text_file_path = image_folder + ".txt"
        try:
            self.gaze_df = self.gaze_processor.load_gaze_data(text_file_path)
            print(f"Loaded gaze data from {text_file_path}")
        except FileNotFoundError:
            print(f"Warning: Gaze data file {text_file_path} not found. Continuing without gaze data.")
            self.gaze_df = pd.DataFrame()
        
        # Get and sort image files
        images = self._get_sorted_images(image_folder)
        if not images:
            raise ValueError(f"No valid image files found in {image_folder}")
        
        print(f"Found {len(images)} images to process")
        
        # Get image dimensions from first frame
        first_frame = cv2.imread(os.path.join(image_folder, images[0]))
        height, width, _ = first_frame.shape
        
        # Process each image
        for i, img_name in enumerate(images):
            if i % fps != 0:
                continue
            self._process_single_image(image_folder, img_name, width, height)

            
            # try:
            #     self._process_single_image(image_folder, img_name, width, height)
            # except KeyboardInterrupt:
            #     print("\\nProcessing interrupted by user")
            #     break
            # except Exception as e:
            #     print(f"Error processing {img_name}: {e}")
            #     continue
        
        # Save updated gaze data if available
        if not self.gaze_df.empty:
            new_path = self.gaze_processor.save_updated_gaze_data(self.gaze_df, text_file_path)
            print(f"Saved updated gaze data with relationships to {new_path}")
        
        self.visualizer.close_all_windows()
    
    def _get_sorted_images(self, image_folder: str) -> List[str]:
        """Get sorted list of image files from folder."""
        valid_extensions = (".png", ".jpg", ".jpeg")
        images = [img for img in os.listdir(image_folder) 
                 if img.lower().endswith(valid_extensions)]
        
        # Sort by the index after the last underscore
        try:
            return sorted(images, key=lambda x: int(x.split('_')[-1].split('.')[0]))
        except (ValueError, IndexError):
            # Fall back to alphabetical sorting if parsing fails
            return sorted(images)
    
    def _process_single_image(self, image_folder: str, img_name: str, 
                            width: int, height: int):
        """Process a single image through the complete pipeline."""
        print(f"Processing {img_name}:")
        
        # Load image
        img_path = os.path.join(image_folder, img_name)
        image = cv2.imread(img_path)
        if image is None:
            print(f"Failed to load image: {img_path}")
            return
        
        # Detect objects
        detected_objects = self.object_detector.detect_all_objects(image)
        self._print_detected_objects(detected_objects)
        
        # Analyze relationships
        try:
            relationships = self.relationship_analyzer.analyze_all_relationships(detected_objects)
            self._print_relationships(relationships)
            
            # Create connection list for visualization
            connection_list = self.relationship_analyzer.create_connection_list(relationships)
            print(f"Connection list: {len(connection_list)} connections")
            
        except Exception as e:
            print(f"Error in relationship analysis: {e}")
            relationships = []
            connection_list = []
        
        # Process gaze data for this frame
        frame_id = img_name.replace('.png', '').replace('.jpg', '').replace('.jpeg', '')
        gaze_positions = []
        
        if not self.gaze_df.empty:
            gaze_positions = self.gaze_processor.get_gaze_positions_for_frame(self.gaze_df, frame_id)
            
            # Update gaze DataFrame with object and relationship information
            objects_list = self.object_detector.get_all_objects_as_list(detected_objects)
            relationships_text = self.relationship_analyzer.format_relationships_for_dataframe(relationships)
            self.gaze_processor.update_frame_data(self.gaze_df, frame_id, objects_list, relationships_text)
        
        # Create comprehensive visualization
        annotated_image = self.visualizer.create_comprehensive_visualization(
            image, detected_objects, connection_list, gaze_positions, scale_factor=2
        )
        
        # Display the image
        key = self.visualizer.display_image(annotated_image, 'Frame', wait_for_key=True)
        
        # Check for ESC key to exit
        if key == 27:  # ESC key
            raise KeyboardInterrupt
    
    def _print_detected_objects(self, detected_objects: Dict[str, List[GameObject]]):
        """Print information about detected objects."""
        print("Found objects:")
        for object_type, objects in detected_objects.items():
            if objects:
                object_dict = {obj.object_id: obj.bounding_box for obj in objects}
                print(f"{object_type.capitalize()} objects: {object_dict}")
    
    def _print_relationships(self, relationships):
        """Print relationship information."""
        print("\\nRelationship between objects:")
        descriptions = self.relationship_analyzer.get_relationship_descriptions(relationships)
        for description in descriptions:
            print(description)
        print()
    
    def process_single_image_file(self, image_path: str, 
                                gaze_data_path: Optional[str] = None) -> Dict:
        """
        Process a single image file and return analysis results.
        
        Args:
            image_path: Path to the image file
            gaze_data_path: Optional path to gaze data file
            
        Returns:
            Dictionary containing analysis results
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        height, width = image.shape[:2]
        
        # Load gaze data if provided
        gaze_positions = []
        if gaze_data_path and os.path.exists(gaze_data_path):
            gaze_df = self.gaze_processor.load_gaze_data(gaze_data_path)
            frame_id = os.path.basename(image_path).split('.')[0]
            gaze_positions = self.gaze_processor.get_gaze_positions_for_frame(gaze_df, frame_id)
        
        # Detect objects
        detected_objects = self.object_detector.detect_all_objects(image)
        
        # Analyze relationships
        relationships = self.relationship_analyzer.analyze_all_relationships(detected_objects)
        connection_list = self.relationship_analyzer.create_connection_list(relationships)
        
        # Create visualization
        annotated_image = self.visualizer.create_comprehensive_visualization(
            image, detected_objects, connection_list, gaze_positions
        )
        
        return {
            'original_image': image,
            'annotated_image': annotated_image,
            'detected_objects': detected_objects,
            'relationships': relationships,
            'connection_list': connection_list,
            'gaze_positions': gaze_positions
        }


def main():
    """Main entry point for the application."""
    if len(sys.argv) < 2:
        print('Usage: python main.py image_folder [output_video] [fps] [game_type]')
        print('  image_folder: Path to folder containing game images')
        print('  output_video: Output video filename (optional, default: test_output.mp4)')
        print('  fps: Processing frequency (optional, default: 1)')
        print('  game_type: Game type (optional, default: seaquest)')
        sys.exit(1)
    
    image_folder = sys.argv[1]
    output_video = sys.argv[2] if len(sys.argv) > 2 else "test_output.mp4"
    fps = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    game_type = sys.argv[4] if len(sys.argv) > 4 else "seaquest"
    app = GameAnalysisApp(game_type)
    app.run(image_folder, output_video, fps)
    
    # try:
    #     # Create and run the application
    #     app = GameAnalysisApp(game_type)
    #     app.run(image_folder, output_video, fps)
        
    # except Exception as e:
    #     print(f"Application error: {e}")
    #     sys.exit(1)


if __name__ == '__main__':
    main()
