"""
Main application module for game object detection and relationship analysis.
Updated to use the new modular structure with environment-specific configurations.
"""
import sys
import os
import cv2
import pandas as pd
import argparse
from typing import List, Dict, Optional, Tuple

# Import core modules
from core.gaze_data_processor import GazeDataProcessor
from core.visualization_manager import VisualizationManager
from core.game_object import GameObject
from core.goal_detector import GoalDetector

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
        self.goal_detector = GoalDetector()
        self.gaze_df = pd.DataFrame()
    
    def run(self, image_folder: str, output_video: str = "test_output.mp4", fps: int = 1, 
            start_frame: int = 0, no_visual: bool = False, process_all: bool = False, save_rel: bool = False):
        """
        Run the main analysis pipeline.
        
        Args:
            image_folder: Path to folder containing game images
            output_video: Output video filename (not currently used)
            fps: Processing frequency (process every fps-th image)
            start_frame: Frame index to start processing from (default: 0)
            no_visual: Skip visual display of frames
            process_all: Process all frames instead of stepping through
            save_rel: Save relationship data to files
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
        
        print(f"Found {len(images)} images to process (starting from frame {start_frame})")
        
        # Get image dimensions from first frame
        first_frame = cv2.imread(os.path.join(image_folder, images[0]))
        height, width, _ = first_frame.shape
        
        # Store processing options in instance variables for access in _process_single_image
        self.no_visual = no_visual
        self.process_all = process_all
        self.save_rel = save_rel
        
        # Process each image starting from the specified frame
        try:
            processed_count = 0
            total_to_process = len([i for i in range(len(images)) if i >= start_frame and i % fps == 0])
            
            for i, img_name in enumerate(images):
                if i < start_frame:
                    continue
                if i % fps != 0:
                    continue
                
                processed_count += 1
                if no_visual:
                    print(f"Progress: {processed_count}/{total_to_process} frames processed")
                
                self._process_single_image(image_folder, img_name, width, height)
                
        except KeyboardInterrupt:
            print("\nProcessing interrupted by user")
        except Exception as e:
            print(f"Error during processing: {e}")
            raise
        
        # Save updated gaze data if available
        if not self.gaze_df.empty:
            new_path = self.gaze_processor.save_updated_gaze_data(self.gaze_df, text_file_path)
            print(f"Saved updated gaze data with relationships to {new_path}")
        
        # Save relationship data if requested
        if save_rel:
            rel_output_path = os.path.join(image_folder, "relationships_output.txt")
            print(f"Saving relationship data to {rel_output_path}")
            # This would need to be implemented based on your specific requirements
        
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
        # print("Printing detected objs")
        self._print_detected_objects(detected_objects)
        
        # Analyze relationships
        try:
            relationships = self.relationship_analyzer.analyze_all_relationships(detected_objects)
            self._print_relationships(relationships)
            
            # Create connection list for visualization
            # print(relationships)
            connection_list = self.relationship_analyzer.create_connection_list(relationships)
            print(f"Connection list: {len(connection_list)} connections")
            print(connection_list)
            
        except Exception as e:
            print(f"Error in relationship analysis: {e}")
            relationships = []
            connection_list = []
        
        # Process gaze data for this frame
        frame_id = img_name.replace('.png', '').replace('.jpg', '').replace('.jpeg', '')
        gaze_positions = []
        detected_goal = "unknown"
        
        if not self.gaze_df.empty:
            gaze_positions = self.gaze_processor.get_gaze_positions_for_frame(self.gaze_df, frame_id)
            
            # Get action for goal detection
            action = self.gaze_processor.get_action_for_frame(self.gaze_df, frame_id)
            
            # Detect goal based on gaze data and game state
            if gaze_positions:
                detected_goal = self.goal_detector.detect_goal(
                    gaze_positions, detected_objects, action, frame_id
                )
            
            # Update gaze DataFrame with object, relationship, and goal information
            objects_list = self.object_detector.get_all_objects_as_list(detected_objects)
            relationships_text = self.relationship_analyzer.format_relationships_for_dataframe(relationships)
            self.gaze_processor.update_frame_data(self.gaze_df, frame_id, objects_list, relationships_text, detected_goal)
        
        # Create comprehensive visualization
        annotated_image = self.visualizer.create_comprehensive_visualization(
            image, detected_objects, connection_list, gaze_positions, scale_factor=2, detected_goal=detected_goal
        )
        
        # Display the image (unless no_visual is set)
        if not getattr(self, 'no_visual', False):
            wait_for_key = not getattr(self, 'process_all', False)
            window_title = f'Frame: {img_name}'
            key = self.visualizer.display_image(annotated_image, window_title, wait_for_key=wait_for_key)
            
            # Check for ESC key to exit
            if key == 27:  # ESC key
                raise KeyboardInterrupt
        
        else:
            # Just print completion when no visual display
            print(f"  ✓ Completed: {img_name}")
    
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
        detected_goal = "unknown"
        if gaze_data_path and os.path.exists(gaze_data_path):
            gaze_df = self.gaze_processor.load_gaze_data(gaze_data_path)
            frame_id = os.path.basename(image_path).split('.')[0]
            gaze_positions = self.gaze_processor.get_gaze_positions_for_frame(gaze_df, frame_id)
            
            # Detect goal if gaze data is available
            if gaze_positions:
                action = self.gaze_processor.get_action_for_frame(gaze_df, frame_id)
                detected_goal = self.goal_detector.detect_goal(
                    gaze_positions, detected_objects, action, frame_id
                )
        
        # Detect objects
        detected_objects = self.object_detector.detect_all_objects(image)
        
        # Analyze relationships
        relationships = self.relationship_analyzer.analyze_all_relationships(detected_objects)
        connection_list = self.relationship_analyzer.create_connection_list(relationships)
        
        # Create visualization
        annotated_image = self.visualizer.create_comprehensive_visualization(
            image, detected_objects, connection_list, gaze_positions, detected_goal=detected_goal
        )
        
        return {
            'original_image': image,
            'annotated_image': annotated_image,
            'detected_objects': detected_objects,
            'relationships': relationships,
            'connection_list': connection_list,
            'gaze_positions': gaze_positions,
            'detected_goal': detected_goal
        }


def main():
    """Main entry point for the application."""
    parser = argparse.ArgumentParser(
        description='Game object detection and relationship analysis for Atari games',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --data /path/to/images
  python main.py --data /path/to/images --fps 2 --start-frame 100
  python main.py --data /path/to/images --no-visual --process-all
  python main.py --data /path/to/images --game-type seaquest --save-rel
        """
    )
    
    # Required arguments
    parser.add_argument('--data',
                       help='Path to folder containing game images',
                       default='data/seaquest/gaze_data_tmp/54_RZ_2461867_Aug-11-09-35-18')
    # Optional arguments
    parser.add_argument('--output-video', default='test_output.mp4',
                       help='Output video filename (default: test_output.mp4)')
    parser.add_argument('--fps', type=int, default=1,
                       help='Processing frequency - process every fps-th image (default: 1)')
    parser.add_argument('--game-type', default='seaquest',
                       help='Game type to analyze (default: seaquest)')
    parser.add_argument('--start-frame', type=int, default=0,
                       help='Frame index to start processing from (default: 0)')
    parser.add_argument('--no-visual', action='store_true',
                       help='Skip visual display of frames (faster processing)')
    parser.add_argument('--process-all', action='store_true',
                       help='Process all frames without waiting for keypress')
    parser.add_argument('--save-rel', action='store_true',
                       help='Save relationship data to output files')
    
    args = parser.parse_args()
    
    # Create and run the application
    try:
        app = GameAnalysisApp(args.game_type)
        print(f"Starting analysis with:")
        print(f"  Data folder: {args.data}")
        print(f"  Game type: {args.game_type}")
        print(f"  FPS: {args.fps}")
        print(f"  Start frame: {args.start_frame}")
        print(f"  Visual display: {'disabled' if args.no_visual else 'enabled'}")
        print(f"  Process mode: {'automatic' if args.process_all else 'step-by-step'}")
        print(f"  Save relationships: {'yes' if args.save_rel else 'no'}")
        print()
        
        app.run(args.data, args.output_video, args.fps, args.start_frame, 
               args.no_visual, args.process_all, args.save_rel)
               
        print("\nAnalysis completed successfully!")
        
    except Exception as e:
        print(f"\nApplication error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
