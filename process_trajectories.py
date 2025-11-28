"""
Process multiple Freeway trajectories with DEBUG logging.
"""

import os
import pandas as pd
import numpy as np
import cv2
import json
from pathlib import Path
from tqdm import tqdm
import sys

# Add to path
sys.path.append('/Users/varun/Desktop/NeSy-Imitation-Learning/')
sys.path.append('/Users/varun/Desktop/NeSy-Imitation-Learning/src')

from env.freeway.object_detector import FreewayObjectDetector
from env.freeway.relationship_analyzer import FreewayRelationshipAnalyzer



def read_custom_csv(filename):
    """Read the custom CSV file with gaze positions."""
    data = []
    with open(filename, 'r') as f:
        header = f.readline().strip().split(',')
        
        for line in f:
            parts = line.strip().split(',')
            if len(parts) >= 6:
                row_data = {
                    'qframe_id': parts[0],
                    'episode_id': parts[1],
                    'score': parts[2],
                    'duration(ms)': parts[3],
                    'unclipped_reward': parts[4],
                    'action': parts[5],
                    'gaze_positions': ','.join(parts[6:])
                }
                data.append(row_data)
    
    return pd.DataFrame(data)


def detect_objects_in_frame(detector, frame_path, debug=False):
    """
    Detect objects in a single frame with DEBUG mode.
    
    Returns:
        tuple: (detected_objects_list, bboxes_list)
    """
    if not os.path.exists(frame_path):
        if debug:
            print(f"    DEBUG: Frame not found: {frame_path}")
        return [], []
    
    # Read image
    image = cv2.imread(str(frame_path))
    if image is None:
        if debug:
            print(f"    DEBUG: Failed to read image: {frame_path}")
        return [], []
    
    if debug:
        print(f"    DEBUG: Image loaded successfully: {image.shape}")
    
    # Convert to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    if debug:
        print(f"    DEBUG: Image converted to RGB")
    
    # Detect objects
    try:
        detected_objects = detector.detect_all_objects(image_rgb)
        if debug:
            print(f"    DEBUG: Detection complete. Found {len(detected_objects)} object types")
            for obj_type, objs in detected_objects.items():
                if objs:
                    print(f"      - {obj_type}: {len(objs)} objects")
    except Exception as e:
        if debug:
            print(f"    DEBUG: Detection failed with error: {e}")
            import traceback
            traceback.print_exc()
        return [], []
    
    # Process detections
    objects_list = []
    bboxes_list = []
    
    for object_type, objects in detected_objects.items():
        if not objects:
            continue
        
        for obj in objects:
            # Skip right chicken (player 2)
            if object_type == 'chicken' and hasattr(obj, 'characteristics'):
                if obj.characteristics.get('player') == 2:
                    if debug:
                        print(f"      DEBUG: Skipping player 2 chicken")
                    continue
            
            # Store object info
            x, y, w, h = obj.xywh
            
            obj_info = {
                'type': object_type,
                'id': obj.object_id
            }
            
            # Add characteristics
            if hasattr(obj, 'characteristics') and obj.characteristics:
                obj_info.update(obj.characteristics)
            
            bbox_info = {
                'type': object_type,
                'x': int(x),
                'y': int(y),
                'w': int(w),
                'h': int(h)
            }
            
            objects_list.append(obj_info)
            bboxes_list.append(bbox_info)
    
    if debug:
        print(f"    DEBUG: Final count: {len(objects_list)} objects after filtering")
    analyzer = FreewayRelationshipAnalyzer()
    relationships = analyzer.analyze_all_relationships(detected_objects)
    relationships_str = analyzer.format_relationships_for_dataframe(relationships)
    
    return objects_list, bboxes_list, relationships_str


def process_trajectory(data_folder, txt_file, detector, trajectory_id, debug_frames=5):
    """
    Process a single trajectory with DEBUG mode for first N frames.
    
    Returns:
        DataFrame with added object detection columns
    """
    print(f"\nProcessing trajectory: {trajectory_id}")
    print(f"  Folder: {data_folder}")
    print(f"  Data file: {txt_file}")
    
    # Check if folder exists
    if not os.path.exists(data_folder):
        print(f"  ERROR: Folder does not exist: {data_folder}")
        return None
    
    # Check if txt file exists
    if not os.path.exists(txt_file):
        print(f"  ERROR: Text file does not exist: {txt_file}")
        return None
    
    # Read data
    df = read_custom_csv(txt_file)
    
    # Convert numeric columns
    df['duration(ms)'] = pd.to_numeric(df['duration(ms)'], errors='coerce')
    df['action'] = pd.to_numeric(df['action'], errors='coerce')
    
    # Add trajectory ID
    df['trajectory_id'] = trajectory_id
    
    # Initialize lists for new columns
    detected_objects_list = []
    bounding_boxes_list = []
    relationships_list = []
    num_objects_list = []
    
    # Check frame naming pattern from first row
    if len(df) > 0:
        first_frame_id = df.iloc[0]['qframe_id']
        print(f"  First frame ID: {first_frame_id}")
        
        # Try to determine the correct frame filename pattern
        numeric_id = first_frame_id.split('_')[-1]
        
        # Try different patterns
        test_patterns = [
            f"RZ_2464601_{numeric_id}.png",
            f"{first_frame_id}.png",
            f"{trajectory_id.split('_')[0]}_{trajectory_id.split('_')[1]}_{trajectory_id.split('_')[2]}_{numeric_id}.png"
        ]
        
        print(f"  Testing frame patterns:")
        found_pattern = None
        for pattern in test_patterns:
            test_path = os.path.join(data_folder, pattern)
            print(f"    - {pattern}: ", end="")
            if os.path.exists(test_path):
                print("FOUND ✓")
                found_pattern = pattern
                break
            else:
                print("NOT FOUND ✗")
        
        if found_pattern is None:
            # List actual files in directory
            print(f"\n  Listing actual files in {data_folder}:")
            files = sorted(os.listdir(data_folder))[:10]
            for f in files:
                print(f"    - {f}")
            print(f"  (showing first 10 of {len(os.listdir(data_folder))} files)")
    
    # Process each frame
    print(f"\n  Processing {len(df)} frames...")
    for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"  {trajectory_id}"):
        frame_id = row['qframe_id']
        
        # Construct frame filename - use the pattern from qframe_id
        numeric_id = frame_id.split('_')[-1]
        
        # Try the original pattern first
        frame_filename = f"RZ_2464601_{numeric_id}.png"
        frame_path = os.path.join(data_folder, frame_filename)
        
        # If doesn't exist, try using the full qframe_id
        if not os.path.exists(frame_path):
            frame_filename = f"{frame_id}.png"
            frame_path = os.path.join(data_folder, frame_filename)
        
        # Debug first few frames
        debug = (idx < debug_frames)
        if debug:
            print(f"\n  DEBUG Frame {idx}:")
            print(f"    Frame ID: {frame_id}")
            print(f"    Frame filename: {frame_filename}")
            print(f"    Full path: {frame_path}")
            print(f"    Exists: {os.path.exists(frame_path)}")
        
        # Detect objects
        # Detect objects and relationships
        objects, bboxes, relationships = detect_objects_in_frame(detector, frame_path, debug=debug)
        
        # Store data
        detected_objects_list.append(json.dumps(objects, separators=(',', ':')))
        bounding_boxes_list.append(json.dumps(bboxes, separators=(',', ':')))
        relationships_list.append(relationships)  # NEW
        num_objects_list.append(len(objects))
    
    # Add new columns
    df['detected_objects'] = detected_objects_list
    df['bounding_boxes'] = bounding_boxes_list
    df['relationships'] = relationships_list 
    df['num_objects'] = num_objects_list
    
    print(f"\n  Completed! Total objects detected: {sum(num_objects_list)}")
    print(f"  Frames with 0 objects: {sum(1 for x in num_objects_list if x == 0)}")
    print(f"  Frames with objects: {sum(1 for x in num_objects_list if x > 0)}")
    
    return df


def process_all_trajectories(base_path, output_file='combined_freeway_data.txt'):
    """
    Process all 10 trajectories and create combined dataframe.
    
    Args:
        base_path: Base path containing trajectory folders
        output_file: Output file name for combined data
    """
    # Initialize detector
    print("Initializing object detector...")
    try:
        detector = FreewayObjectDetector()
        print("  Detector initialized successfully!")
    except Exception as e:
        print(f"  ERROR initializing detector: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Find all trajectory folders and files
    print(f"\nScanning directory: {base_path}")
    trajectory_folders = []
    
    for item in os.listdir(base_path):
        item_path = os.path.join(base_path, item)
        if os.path.isdir(item_path) and 'RZ_' in item or 'JAW_' in item or 'KM_' in item:
            # Check if corresponding txt file exists
            txt_file = item + '.txt'
            txt_path = os.path.join(base_path, txt_file)
            if os.path.exists(txt_path):
                trajectory_folders.append((item_path, txt_path, item))
                print(f"  Found trajectory: {item}")
            else:
                print(f"  Skipping {item}: No matching .txt file")
    
    print(f"\nFound {len(trajectory_folders)} trajectories to process")
    
    if len(trajectory_folders) == 0:
        print("No trajectories found! Please check the base path.")
        print(f"Base path contents:")
        for item in os.listdir(base_path)[:20]:
            print(f"  - {item}")
        return
    
    # Process all trajectories
    all_dataframes = []
    
    for idx, (folder, txt_file, traj_id) in enumerate(trajectory_folders, 1):
        print(f"\n{'='*70}")
        print(f"Trajectory {idx}/{len(trajectory_folders)}")
        print(f"{'='*70}")
        
        try:
            df_traj = process_trajectory(folder, txt_file, detector, traj_id)
            if df_traj is not None:
                all_dataframes.append(df_traj)
        except Exception as e:
            print(f"ERROR processing {traj_id}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    if len(all_dataframes) == 0:
        print("\nERROR: No trajectories were processed successfully!")
        return
    
    # Combine all dataframes
    print(f"\n{'='*70}")
    print("Combining all trajectories...")
    print(f"{'='*70}")
    
    combined_df = pd.concat(all_dataframes, ignore_index=True)
    
    # Add global frame index
    combined_df['global_frame_id'] = range(len(combined_df))
    
    # Reorder columns
    columns_order = [
        'global_frame_id',
        'trajectory_id',
        'qframe_id',
        'episode_id',
        'score',
        'duration(ms)',
        'unclipped_reward',
        'action',
        'num_objects',
        'detected_objects',
        'bounding_boxes',
        'relationships',
        'gaze_positions'
    ]
    
    combined_df = combined_df[columns_order]
    
    # Save to file
    output_path = os.path.join(base_path, output_file)
    combined_df.to_csv(output_path, index=False, sep='\t')
    
    print(f"\n{'='*70}")
    print("PROCESSING COMPLETE!")
    print(f"{'='*70}")
    print(f"Total trajectories: {len(all_dataframes)}")
    print(f"Total frames: {len(combined_df)}")
    print(f"Total objects detected: {combined_df['num_objects'].sum()}")
    print(f"Output saved to: {output_path}")
    print(f"{'='*70}")
    
    # Print summary statistics
    print(f"\nSummary Statistics:")
    print(f"  Average objects per frame: {combined_df['num_objects'].mean():.2f}")
    print(f"  Max objects in a frame: {combined_df['num_objects'].max()}")
    print(f"  Min objects in a frame: {combined_df['num_objects'].min()}")
    print(f"  Frames with 0 objects: {sum(combined_df['num_objects'] == 0)}")
    print(f"  Frames with objects: {sum(combined_df['num_objects'] > 0)}")
    print(f"\n  Frames by trajectory:")
    for traj_id in combined_df['trajectory_id'].unique():
        traj_df = combined_df[combined_df['trajectory_id'] == traj_id]
        count = len(traj_df)
        obj_count = traj_df['num_objects'].sum()
        print(f"    {traj_id}: {count} frames, {obj_count} objects")
    
    return combined_df


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        base_path = sys.argv[1]
    else:
        base_path = "."
    
    output_file = sys.argv[2] if len(sys.argv) > 2 else "combined_freeway_data.txt"
    
    print(f"\n{'='*70}")
    print("Freeway Multi-Trajectory Object Detection Processor (DEBUG)")
    print(f"{'='*70}")
    print(f"Base path: {base_path}")
    print(f"Output file: {output_file}")
    
    combined_df = process_all_trajectories(base_path, output_file)
