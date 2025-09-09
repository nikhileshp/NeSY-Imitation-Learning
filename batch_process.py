#!/usr/bin/env python3
"""
Batch processing script for game object detection and relationship analysis.
Automatically processes all images in sorted order and creates updated gaze data with relationships.
"""
import sys
import os
import cv2
import pandas as pd
from typing import List, Dict, Optional, Tuple
import time

# Import core modules
from core.gaze_data_processor import GazeDataProcessor
from core.visualization_manager import VisualizationManager
from core.game_object import GameObject

# Import Seaquest-specific modules
from env.seaquest.object_detector import SeaquestObjectDetector
from env.seaquest.relationship_analyzer import SeaquestRelationshipAnalyzer
from env.seaquest.config import VISUALIZATION_COLORS


class BatchGameAnalysisApp:
    """Batch processing application for game object detection and relationship analysis."""
    
    def __init__(self, game_type: str = "seaquest", save_visualizations: bool = False):
        """
        Initialize the batch game analysis application.
        
        Args:
            game_type: Type of game ("seaquest" for now, extensible for other games)
            save_visualizations: Whether to save visualization images to disk
        """
        if game_type == "seaquest":
            self.object_detector = SeaquestObjectDetector()
            self.relationship_analyzer = SeaquestRelationshipAnalyzer()
            self.visualizer = VisualizationManager(VISUALIZATION_COLORS)
        else:
            raise ValueError(f"Game type '{game_type}' not supported yet")
        
        self.gaze_processor = GazeDataProcessor()
        self.gaze_df = pd.DataFrame()
        self.save_visualizations = save_visualizations
    
    def run_batch(self, image_folder: str, fps: int = 1, output_folder: Optional[str] = None):
        """
        Run the batch analysis pipeline on all images.
        
        Args:
            image_folder: Path to folder containing game images
            fps: Processing frequency (process every fps-th image)
            output_folder: Optional folder to save visualization images
        """
        # Validate input folder
        if not os.path.exists(image_folder):
            raise FileNotFoundError(f"Image folder {image_folder} does not exist.")
        
        # Create output folder if needed
        if output_folder and not os.path.exists(output_folder):
            os.makedirs(output_folder)
            print(f"Created output folder: {output_folder}")
        
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
        print(f"Processing every {fps} image(s)")
        if self.save_visualizations:
            print(f"Visualizations will be saved to: {output_folder or 'same as input folder'}")
        
        # Get image dimensions from first frame
        first_frame = cv2.imread(os.path.join(image_folder, images[0]))
        height, width, _ = first_frame.shape
        
        # Initialize progress tracking
        images_to_process = [img for i, img in enumerate(images) if i % fps == 0]
        total_images = len(images_to_process)
        start_time = time.time()
        
        print(f"\nStarting batch processing of {total_images} images...")
        print("-" * 60)
        
        # Process each image
        for idx, img_name in enumerate(images_to_process):
            try:
                self._process_single_image(
                    image_folder, img_name, width, height, 
                    idx + 1, total_images, output_folder
                )
            except KeyboardInterrupt:
                print("\nProcessing interrupted by user")
                break
            except Exception as e:
                print(f"Error processing {img_name}: {e}")
                continue
        
        # Calculate processing time
        elapsed_time = time.time() - start_time
        print(f"\nBatch processing completed in {elapsed_time:.2f} seconds")
        print(f"Average time per image: {elapsed_time/total_images:.2f} seconds")
        
        # Save updated gaze data if available
        if not self.gaze_df.empty:
            new_path = self.gaze_processor.save_updated_gaze_data(self.gaze_df, text_file_path)
            print(f"Saved updated gaze data with relationships to {new_path}")
        
        self.visualizer.close_all_windows()
        print("Batch processing completed successfully!")
    
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
                            width: int, height: int, current_idx: int, 
                            total_images: int, output_folder: Optional[str] = None):
        """Process a single image through the complete pipeline."""
        print(f"[{current_idx}/{total_images}] Processing {img_name}")
        
        # Load image
        img_path = os.path.join(image_folder, img_name)
        image = cv2.imread(img_path)
        if image is None:
            print(f"  ❌ Failed to load image: {img_path}")
            return
        
        # Detect objects
        detected_objects = self.object_detector.detect_all_objects(image)
        object_count = sum(len(objects) for objects in detected_objects.values())
        print(f"  🔍 Detected {object_count} objects")
        
        # Analyze relationships
        try:
            relationships = self.relationship_analyzer.analyze_all_relationships(detected_objects)
            connection_list = self.relationship_analyzer.create_connection_list(relationships)
            print(f"  🔗 Found {len(relationships)} relationships, {len(connection_list)} connections")
        except Exception as e:
            print(f"  ⚠️  Error in relationship analysis: {e}")
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
            print(f"  👁️  Updated gaze data with {len(gaze_positions)} gaze points")
        
        # Create visualization if requested
        if self.save_visualizations:
            annotated_image = self.visualizer.create_comprehensive_visualization(
                image, detected_objects, connection_list, gaze_positions, scale_factor=2
            )
            
            # Save visualization
            output_path = os.path.join(
                output_folder or image_folder, 
                f"annotated_{img_name}"
            )
            cv2.imwrite(output_path, annotated_image)
            print(f"  💾 Saved visualization to {output_path}")
    
    def run_quick_analysis(self, image_folder: str, fps: int = 1) -> Dict:
        """
        Run a quick analysis to get summary statistics without saving visualizations.
        
        Args:
            image_folder: Path to folder containing game images
            fps: Processing frequency (process every fps-th image)
            
        Returns:
            Dictionary with summary statistics
        """
        print(f"Running quick analysis on {image_folder}")
        
        # Get and sort image files
        images = self._get_sorted_images(image_folder)
        if not images:
            raise ValueError(f"No valid image files found in {image_folder}")
        
        images_to_process = [img for i, img in enumerate(images) if i % fps == 0]
        
        # Initialize counters
        stats = {
            'total_images': len(images_to_process),
            'total_objects': 0,
            'total_relationships': 0,
            'object_counts': {},
            'relationship_counts': {},
            'processing_time': 0
        }
        
        start_time = time.time()
        
        for idx, img_name in enumerate(images_to_process):
            if idx % 10 == 0:  # Progress update every 10 images
                print(f"  Progress: {idx}/{len(images_to_process)} images")
            
            try:
                img_path = os.path.join(image_folder, img_name)
                image = cv2.imread(img_path)
                if image is None:
                    continue
                
                # Detect objects
                detected_objects = self.object_detector.detect_all_objects(image)
                
                # Count objects by type
                for obj_type, objects in detected_objects.items():
                    count = len(objects)
                    stats['total_objects'] += count
                    stats['object_counts'][obj_type] = stats['object_counts'].get(obj_type, 0) + count
                
                # Analyze relationships
                try:
                    relationships = self.relationship_analyzer.analyze_all_relationships(detected_objects)
                    stats['total_relationships'] += len(relationships)
                    
                    # Count relationships by type
                    for rel in relationships:
                        rel_type = rel.relationship_type
                        stats['relationship_counts'][rel_type] = stats['relationship_counts'].get(rel_type, 0) + 1
                
                except Exception:
                    continue
                    
            except Exception:
                continue
        
        stats['processing_time'] = time.time() - start_time
        
        # Print summary
        print(f"\n📊 Analysis Summary:")
        print(f"   Processed: {stats['total_images']} images")
        print(f"   Total objects: {stats['total_objects']}")
        print(f"   Total relationships: {stats['total_relationships']}")
        print(f"   Processing time: {stats['processing_time']:.2f} seconds")
        
        print(f"\n📦 Object counts:")
        for obj_type, count in sorted(stats['object_counts'].items()):
            print(f"   {obj_type}: {count}")
        
        print(f"\n🔗 Top relationship types:")
        sorted_rels = sorted(stats['relationship_counts'].items(), key=lambda x: x[1], reverse=True)
        for rel_type, count in sorted_rels[:10]:  # Top 10
            print(f"   {rel_type}: {count}")
        
        return stats


def main():
    """Main entry point for the batch processing application."""
    if len(sys.argv) < 2:
        print('Usage: python batch_process.py image_folder [options]')
        print('  image_folder: Path to folder containing game images')
        print('')
        print('Options:')
        print('  --fps N              Process every N-th image (default: 1)')
        print('  --game_type TYPE     Game type (default: seaquest)')
        print('  --save-viz           Save visualization images')
        print('  --output-folder DIR  Output folder for visualizations')
        print('  --quick-analysis     Run quick analysis only (no gaze data update)')
        print('')
        print('Examples:')
        print('  python batch_process.py data/seaquest/frames')
        print('  python batch_process.py data/seaquest/frames --fps 5 --save-viz')
        print('  python batch_process.py data/seaquest/frames --quick-analysis')
        sys.exit(1)
    
    # Parse arguments
    image_folder = sys.argv[1]
    fps = 1
    game_type = "seaquest"
    save_visualizations = False
    output_folder = None
    quick_analysis = False
    
    i = 2
    while i < len(sys.argv):
        if sys.argv[i] == '--fps' and i + 1 < len(sys.argv):
            fps = int(sys.argv[i + 1])
            i += 2
        elif sys.argv[i] == '--game_type' and i + 1 < len(sys.argv):
            game_type = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == '--save-viz':
            save_visualizations = True
            i += 1
        elif sys.argv[i] == '--output-folder' and i + 1 < len(sys.argv):
            output_folder = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == '--quick-analysis':
            quick_analysis = True
            i += 1
        else:
            print(f"Unknown option: {sys.argv[i]}")
            sys.exit(1)
    
    try:
        # Create and run the application
        app = BatchGameAnalysisApp(game_type, save_visualizations)
        
        if quick_analysis:
            app.run_quick_analysis(image_folder, fps)
        else:
            app.run_batch(image_folder, fps, output_folder)
            
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        sys.exit(0)
    except Exception as e:
        print(f"Application error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
