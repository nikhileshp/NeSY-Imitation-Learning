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
from tqdm import tqdm

# Add project root to sys.path to allow importing 'models'
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

# Import core modules
from core.gaze_data_processor import GazeDataProcessor
from core.visualization_manager import VisualizationManager
from core.game_object import GameObject
from core.goal_detector import GoalDetector
from core.distance_weight_calculator import DistanceWeightCalculator
from attention_weights import calculate_predicate_weights, calculate_example_weight, create_object_weight_mapping

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
        self.verbose = 1  # Default verbosity level
        self.distance_weight_calculator = None  # Will be initialized when screen dimensions are known
        self.use_alternating_class_weights = False  # Default to standard reciprocal rank weights
        self.use_nearest_only_weights = False  # Default to standard reciprocal rank weights
        self.use_euclidean_distance_weights = False  # Default to standard reciprocal rank weights
        self.zero_second_object_weight = False  # Default to not zeroing second object weights
    
    def run(self, image_folder: str, output_video: str = "test_output.mp4", fps: int = 1, 
            start_frame: int = 0, no_visual: bool = False, process_all: bool = False, 
            save_rel: bool = False, verbose: int = 1, use_alternating_class_weights: bool = False,
            use_nearest_only_weights: bool = False, use_euclidean_distance_weights: bool = False,
            zero_second_object_weight: bool = False):
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
            verbose: Verbosity level (0=quiet, 1=minimal, 2=verbose)
            use_alternating_class_weights: Use alternating class weight calculation
            use_nearest_only_weights: Use nearest-only weight calculation
            use_euclidean_distance_weights: Use euclidean distance-based attention weights
            zero_second_object_weight: Set second object in each class to weight 0
        """
        # Store verbosity level and settings
        self.verbose = verbose
        self.use_alternating_class_weights = use_alternating_class_weights
        self.use_nearest_only_weights = use_nearest_only_weights
        self.use_euclidean_distance_weights = use_euclidean_distance_weights
        self.zero_second_object_weight = zero_second_object_weight
        
        # Validate input folder
        if not os.path.exists(image_folder):
            raise FileNotFoundError(f"Image folder {image_folder} does not exist.")
        
        # Load gaze data
        text_file_path = image_folder + ".txt"
        try:
            self.gaze_df = self.gaze_processor.load_gaze_data(text_file_path)
            if verbose >= 2:
                print(f"Loaded gaze data from {text_file_path}")
        except FileNotFoundError:
            if verbose >= 2:
                print(f"Warning: Gaze data file {text_file_path} not found. Continuing without gaze data.")
            self.gaze_df = pd.DataFrame()
        
        # Get and sort image files
        images = self._get_sorted_images(image_folder)
        if not images:
            raise ValueError(f"No valid image files found in {image_folder}")
        
        if verbose >= 2:
            print(f"Found {len(images)} images to process (starting from frame {start_frame})")
        
        # Get image dimensions from first frame
        first_frame = cv2.imread(os.path.join(image_folder, images[0]))
        height, width, _ = first_frame.shape
        
        # Initialize distance weight calculator with screen dimensions
        self.distance_weight_calculator = DistanceWeightCalculator(width, height)
        
        # Store processing options in instance variables for access in _process_single_image
        self.no_visual = no_visual
        self.process_all = process_all
        self.save_rel = save_rel
        
        # Process each image starting from the specified frame
        try:
            # Filter images to process
            images_to_process = [(i, img_name) for i, img_name in enumerate(images) 
                               if i >= start_frame and i % fps == 0]
            
            # Create progress bar for this trajectory
            trajectory_name = os.path.basename(image_folder)
            pbar_desc = f"Processing {trajectory_name}" if len(trajectory_name) > 0 else "Processing images"
            
            with tqdm(images_to_process, desc=pbar_desc, disable=(verbose == 0)) as pbar:
                for i, img_name in pbar:
                    self._process_single_image(image_folder, img_name, width, height)
                    pbar.set_postfix_str(f"Frame: {img_name}")
                
        except KeyboardInterrupt:
            if verbose >= 1:
                print("\nProcessing interrupted by user")
        except Exception as e:
            if verbose >= 1:
                print(f"Error during processing: {e}")
            raise
        
        # Save updated gaze data if available
        if not self.gaze_df.empty:
            new_path = self.gaze_processor.save_updated_gaze_data(self.gaze_df, text_file_path)
            if verbose >= 1:
                print(f"Saved updated gaze data with relationships to {new_path}")
        
        # Save relationship data if requested
        if save_rel:
            rel_output_path = os.path.join(image_folder, "relationships_output.txt")
            if verbose >= 1:
                print(f"Saving relationship data to {rel_output_path}")
            # This would need to be implemented based on your specific requirements
            # Return the df or save it directly
            self.gaze_df.to_csv(rel_output_path, sep='\t', index=False)

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
        if self.verbose >= 2:
            print(f"Processing {img_name}:")
        
        # Load image
        img_path = os.path.join(image_folder, img_name)
        image = cv2.imread(img_path)
        if image is None:
            if self.verbose >= 1:
                print(f"Failed to load image: {img_path}")
            return
        
        # Detect objects
        detected_objects = self.object_detector.detect_all_objects(image)
        if self.verbose >= 2:
            self._print_detected_objects(detected_objects)
        
        # Analyze relationships
        try:
            relationships = self.relationship_analyzer.analyze_all_relationships(detected_objects)
            if self.verbose >= 2:
                self._print_relationships(relationships)
            
            # Create connection list for visualization
            connection_list = self.relationship_analyzer.create_connection_list(relationships)
            if self.verbose >= 2:
                print(f"Connection list: {len(connection_list)} connections")
                print(connection_list)
            
        except Exception as e:
            if self.verbose >= 1:
                print(f"Error in relationship analysis: {e}")
            relationships = []
            connection_list = []
        
        # Process gaze data for this frame
        frame_id = img_name.replace('.png', '').replace('.jpg', '').replace('.jpeg', '')
        gaze_positions = []
        detected_goal = "unknown"
        distance_weights_text = ""  # Initialize outside the if block
        predicate_weights_text = ""
        example_weight_text = ""
        
        if not self.gaze_df.empty:
            gaze_positions = self.gaze_processor.get_gaze_positions_for_frame(self.gaze_df, frame_id)
            
            # Get action for goal detection
            action = self.gaze_processor.get_action_for_frame(self.gaze_df, frame_id)
            
            # Detect goal based on gaze data and game state
            if gaze_positions:
                detected_goal = self.goal_detector.detect_goal(
                    gaze_positions, detected_objects, action, frame_id
                )
            
            # Calculate distance weights for relationships involving spatial objects
            if self.distance_weight_calculator and gaze_positions:
                predicate_weights_text = ""
                if self.use_euclidean_distance_weights:
                    if len(gaze_positions) > 1:
                        avg_gaze_x = int(sum([pos[0] for pos in gaze_positions]) / len(gaze_positions))
                        avg_gaze_y = int(sum([pos[1] for pos in gaze_positions]) / len(gaze_positions))
                        eye_pos = (avg_gaze_x, avg_gaze_y)
                    else:
                        eye_pos = gaze_positions[0]
                    # Use euclidean distance-based attention weights
                    # First, list all the relationships in relationships_text
                    object_types_no_arg = ['facing_side', 'water_surface', 'diver_state', 'oxygen_state', 'visibility_state']
                    centroids = [(None, (0,0))]  # Placeholder for obj2 centroid
                    for rel in relationships:
                        obj2 = rel.obj2
                        # print(rel)
                        if obj2.object_type not in object_types_no_arg:
                        
                            # Get centroid for game object
                            centroid_obj = obj2.center
                            
                            centroids[0] = (obj2.object_id, centroid_obj)
                            predicate_weight = calculate_predicate_weights(
                        eye_pos, centroids, width, height, k=0.075
                    )
                            
                            rel_text = self.relationship_analyzer.format_relationships_for_dataframe([rel])
                            predicate_weights_text += f"{predicate_weight[0]:.3f} "
                        else:
                            rel_text = self.relationship_analyzer.format_relationships_for_dataframe([rel])
                            predicate_weights_text += f"1.000 "

                            
                            
                    # Relationships
                    relationships_text = self.relationship_analyzer.format_relationships_for_dataframe(relationships)                 

                else:
                    # Use original distance weight calculation
                    distance_weights = self.distance_weight_calculator.calculate_relationship_distance_weights(
                        relationships, gaze_positions, self.use_alternating_class_weights, self.use_nearest_only_weights
                    )
                    distance_weights_text = self.distance_weight_calculator.format_distance_weights_for_dataframe(
                        distance_weights
                    )
                    
                    if self.verbose >= 2 and distance_weights:
                        print(f"\n  Distance weights for relationships:")
                        for rel_identifier, weight in distance_weights.items():
                            print(f"    {rel_identifier}: {weight:.3f}")
                        print(f"  Formatted: {distance_weights_text}")
            
            # Update gaze DataFrame with object, relationship, and goal information
            objects_list = self.object_detector.get_all_objects_as_list(detected_objects)
            relationships_text = self.relationship_analyzer.format_relationships_for_dataframe(relationships)

            # print(relationships_text)
            
            # If using euclidean distance weights, add predicate and example weights
            # print(predicate_weights_text)
            if self.use_euclidean_distance_weights:
                self.gaze_processor.update_frame_data(
                    self.gaze_df, frame_id, objects_list, relationships_text, detected_goal,
                    distance_weights_text, predicate_weights_text, example_weight_text
                )
            else:
                self.gaze_processor.update_frame_data(
                    self.gaze_df, frame_id, objects_list, relationships_text, detected_goal,
                    distance_weights_text
                )
        
        # Create comprehensive visualization
        annotated_image = self.visualizer.create_comprehensive_visualization(
            image, detected_objects, connection_list, gaze_positions, scale_factor=2, 
            detected_goal=detected_goal, distance_weights_text=distance_weights_text
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
            if self.verbose >= 2:
                print(f"  ✓ Completed: {img_name}")

    def list_trajectory_dirs(self, parent_folder: str) -> List[str]:
        """
        List trajectory subfolders inside parent_folder that have a matching .txt file
        alongside them (same base name) and contain at least one image.
        """
        if not os.path.isdir(parent_folder):
            return []
        candidates = []
        for entry in os.listdir(parent_folder):
            full_path = os.path.join(parent_folder, entry)
            if not os.path.isdir(full_path):
                continue
            # Must have corresponding txt next to the folder
            txt_path = os.path.join(parent_folder, f"{entry}.txt")
            if not os.path.exists(txt_path):
                continue
            # Must contain at least one image file
            try:
                imgs = self._get_sorted_images(full_path)
            except Exception:
                imgs = []
            if imgs:
                candidates.append(full_path)
        return sorted(candidates)
    
    def _print_detected_objects(self, detected_objects: Dict[str, List[GameObject]]):
        """Print information about detected objects."""
        if self.verbose >= 2:
            print("Found objects:")
            for object_type, objects in detected_objects.items():
                if objects:
                    object_dict = {obj.object_id: obj.bounding_box for obj in objects}
                    print(f"{object_type.capitalize()} objects: {object_dict}")
    
    def _print_relationships(self, relationships):
        """Print relationship information."""
        if self.verbose >= 2:
            print("\\nRelationship between objects:")
            descriptions = self.relationship_analyzer.get_relationship_descriptions(relationships)
            for description in descriptions:
                print(description)
            print()
    
    def _compute_relationship_predicate_weights(self, relationships, object_weight_map):
        """
        Compute predicate weight for each relationship.
        
        Args:
            relationships: List of SpatialRelationship objects
            object_weight_map: Dictionary mapping object_id to attention weight
            
        Returns:
            Tuple of (list of weights, ordered list of weights matching relationship order)
        """
        weights = []
        
        for rel in relationships:
            # Determine if this relationship has groundings
            obj1_type = rel.obj1.object_type
            obj2_type = rel.obj2.object_type
            
            # Relationships with no grounding objects (virtual objects) get weight 1.0
            # These are relationships where NEITHER object is a real game object
            if obj2_type in ['facing_side', 'water_surface', 'diver_state', 'oxygen_state']:
                weight = 1.0
            elif obj2_type == 'visibility_state':
                # Visibility relationships: visibleDiver(diver_0), visibleEnemy(enemy_1)
                # These are grounded to obj1 (the visible object), not obj2
                obj1_id = rel.obj1.object_id
                weight = object_weight_map.get(obj1_id, 1.0)
            else:
                # Spatial relationships with actual object groundings
                # Use the weight of obj2 (the grounded object)
                obj2_id = rel.obj2.object_id
                weight = object_weight_map.get(obj2_id, 1.0)
            
            weights.append(weight)
        
        return weights
    
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
        
        # Initialize distance weight calculator for this image dimensions if not already done
        if self.distance_weight_calculator is None:
            self.distance_weight_calculator = DistanceWeightCalculator(width, height)
        
        # Calculate distance weights if gaze data is available
        distance_weights_text = ""
        if self.distance_weight_calculator and gaze_positions:
            distance_weights = self.distance_weight_calculator.calculate_relationship_distance_weights(
                relationships, gaze_positions, self.use_alternating_class_weights, self.use_nearest_only_weights
            )
            distance_weights_text = self.distance_weight_calculator.format_distance_weights_for_dataframe(
                distance_weights
            )
        
        # Create visualization
        annotated_image = self.visualizer.create_comprehensive_visualization(
            image, detected_objects, connection_list, gaze_positions, 
            detected_goal=detected_goal, distance_weights_text=distance_weights_text
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
  # Process a single trajectory folder
  python main.py --data /path/to/data/seaquest/gaze_data_tmp/54_RZ_...
  python main.py --data /path/to/trajectory --fps 2 --start-frame 100
  python main.py --data /path/to/trajectory --no-visual --process-all

  # Process all trajectories contained in a parent data folder
  python main.py --data /path/to/data/seaquest/gaze_data_tmp --all-trajectories --no-visual --process-all
  python main.py --data /path/to/data/seaquest/gaze_data_tmp --all-trajectories --game-type seaquest --save-rel
        """
    )
    
    # Required/primary argument
    parser.add_argument('--data',
                       help='Path to folder containing game images OR a parent folder containing multiple trajectory subfolders',
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
    parser.add_argument('--no-visual', action='store_true', default=False,
                       help='Skip visual display of frames (faster processing)')
    parser.add_argument('--process-all', action='store_true', default=False,
                       help='Process all frames without waiting for keypress')
    parser.add_argument('--save-rel', action='store_true',
                       help='Save relationship data to output files')
    parser.add_argument('--all-trajectories', action='store_true',
                       help='If set, treat --data as a parent folder and process all trajectory subfolders that have a matching .txt file')
    parser.add_argument('--verbose', '-v', type=int, default=1, choices=[0, 1, 2],
                       help='Verbosity level: 0=quiet (progress bars only), 1=minimal output, 2=verbose output (default: 1)')
    parser.add_argument('--alternating-class-weights', action='store_true',
                       help='Use alternating class weight calculation (first object of each class gets weight, second gets 0, etc.)')
    parser.add_argument('--nearest-only-weights', action='store_true',
                       help='Use nearest-only weight calculation (only the nearest object gets weight 1, all others get 0)')
    parser.add_argument('--euclidean-distance-weights', action='store_true',
                       help='Use euclidean distance-based Gaussian attention weights for predicates (saves predicate weights and example weight)')
    parser.add_argument('--zero-second-object-weight', action='store_true',
                       help='When using euclidean distance weights, set the second object in each class to weight 0')
    
    args = parser.parse_args()
    
    # Validate argument combinations
    if args.alternating_class_weights and args.nearest_only_weights:
        parser.error("--alternating-class-weights and --nearest-only-weights are mutually exclusive")
    if args.euclidean_distance_weights and (args.alternating_class_weights or args.nearest_only_weights):
        parser.error("--euclidean-distance-weights is mutually exclusive with other weight calculation methods")
    
    # Create and run the application
    try:
        app = GameAnalysisApp(args.game_type)
        print(f"Starting analysis with:")
        print(f"  Data path: {args.data}")
        print(f"  Game type: {args.game_type}")
        print(f"  FPS: {args.fps}")
        print(f"  Start frame: {args.start_frame}")
        print(f"  Visual display: {'disabled' if args.no_visual else 'enabled'}")
        print(f"  Process mode: {'automatic' if args.process_all else 'step-by-step'}")
        print(f"  Save relationships: {'yes' if args.save_rel else 'no'}")
        print(f"  All trajectories mode: {'yes' if args.all_trajectories else 'no'}")
        print(f"  Alternating class weights: {'enabled' if args.alternating_class_weights else 'disabled'}")
        print(f"  Nearest-only weights: {'enabled' if args.nearest_only_weights else 'disabled'}")
        print(f"  Euclidean distance weights: {'enabled' if args.euclidean_distance_weights else 'disabled'}")
        print(f"  Zero second object weight: {'enabled' if args.zero_second_object_weight else 'disabled'}")
        print(f"  Verbosity level: {args.verbose}")
        print()
        
        if args.all_trajectories:
            # Treat args.data as a parent folder containing multiple trajectory subfolders
            traj_dirs = app.list_trajectory_dirs(args.data)
            if not traj_dirs:
                raise ValueError(f"No valid trajectory subfolders with matching .txt files found in {args.data}")
            
            if args.verbose >= 1:
                print(f"Discovered {len(traj_dirs)} trajectory folders. Starting batch processing...\n")
            
            # Collect all trajectory data into a consolidated DataFrame
            consolidated_df = pd.DataFrame()
            
            # Overall progress bar for all trajectories
            with tqdm(traj_dirs, desc="Overall Progress", disable=(args.verbose == 0)) as overall_pbar:
                for traj_dir in overall_pbar:
                    trajectory_name = os.path.basename(traj_dir)
                    overall_pbar.set_postfix_str(f"Processing: {trajectory_name}")
                    
                    try:
                        app.run(traj_dir, args.output_video, args.fps, args.start_frame, 
                                args.no_visual, args.process_all, args.save_rel, args.verbose, 
                                args.alternating_class_weights, args.nearest_only_weights, 
                                args.euclidean_distance_weights, args.zero_second_object_weight)
                        
                        # Add trajectory identifier to the gaze DataFrame
                        if not app.gaze_df.empty:
                            app.gaze_df['trajectory'] = trajectory_name
                            # Concatenate to consolidated DataFrame
                            consolidated_df = pd.concat([consolidated_df, app.gaze_df], ignore_index=True)
                        
                        if args.verbose >= 2:
                            print(f"Completed trajectory: {traj_dir}")
                            
                    except Exception as e:
                        if args.verbose >= 1:
                            print(f"Error processing trajectory {traj_dir}: {e}")
                        # Continue to next trajectory
                        continue
            
            # Save consolidated relationships data
            if not consolidated_df.empty:
                relationships_output_path = os.path.join(args.data, "relationships.txt")
                consolidated_df.to_csv(relationships_output_path, sep='\t', index=False)
                if args.verbose >= 1:
                    print(f"\nSaved consolidated relationships data for all trajectories to {relationships_output_path}")
                    print(f"Total frames processed: {len(consolidated_df)}")
                    print(f"Trajectories included: {consolidated_df['trajectory'].nunique()}")
            else:
                if args.verbose >= 1:
                    print("\nNo data to save - all trajectories were empty or failed to process.")
            
            if args.verbose >= 1:
                print("\nBatch processing completed!")
        else:
            # Process a single trajectory folder
            app.run(args.data, args.output_video, args.fps, args.start_frame, 
                    args.no_visual, args.process_all, args.save_rel, args.verbose, 
                    args.alternating_class_weights, args.nearest_only_weights, 
                    args.euclidean_distance_weights, args.zero_second_object_weight)
            if args.verbose >= 1:
                print("\nAnalysis completed successfully!")
        
    except Exception as e:
        print(f"\nApplication error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
