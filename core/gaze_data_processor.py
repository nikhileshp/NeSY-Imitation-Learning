"""
Gaze data processing module for handling gaze tracking data.
"""
import os
import pandas as pd
from typing import List, Dict, Tuple, Optional, Any


class GazeDataProcessor:
    """Handles reading and processing of gaze tracking data from text files."""
    
    def __init__(self):
        """Initialize the gaze data processor."""
        pass
    
    def load_gaze_data(self, text_file_path: str) -> pd.DataFrame:
        """
        Load gaze data from a text file into a pandas DataFrame.
        
        Args:
            text_file_path: Path to the gaze data text file
            
        Returns:
            DataFrame containing gaze data
            
        Raises:
            FileNotFoundError: If the text file doesn't exist
            ValueError: If the file format is invalid
        """
        if not os.path.exists(text_file_path):
            raise FileNotFoundError(f"Text file {text_file_path} does not exist.")
        
        gaze_info = []
        
        with open(text_file_path, 'r') as f:
            gaze_data = f.readlines()
        
        for line in gaze_data:
            parsed_line = self._parse_gaze_line(line)
            if parsed_line:
                gaze_info.append(parsed_line)
        
        return pd.DataFrame(gaze_info)
    
    def _parse_gaze_line(self, line: str) -> Optional[Dict[str, Any]]:
        """
        Parse a single line of gaze data.
        
        Args:
            line: Single line from the gaze data file
            
        Returns:
            Dictionary containing parsed gaze data or None if invalid
        """
        parts = line.strip().split(',')
        
        # Skip lines with insufficient data
        if len(parts) <= 7:
            return None
        
        frameid = parts[0]
        episode_id = parts[1]
        score = parts[2]
        duration = float(parts[3])
        unclipped_reward = float(parts[4])
        action = int(parts[5])
        gaze_positions = parts[6:]
        
        # Convert gaze positions to a list of tuples (x, y)
        if len(gaze_positions) % 2 != 0:
            print(f"Warning: Odd number of gaze positions in line: {line.strip()}")
            return None
        
        # Convert gaze positions to integers and group them as (x, y) pairs
        try:
            gaze_positions = [float(pos) for pos in gaze_positions]
            gaze_positions = [(int(gaze_positions[i]), int(gaze_positions[i+1])) 
                             for i in range(0, len(gaze_positions), 2)]
        except ValueError as e:
            print(f"Warning: Invalid gaze position format in line: {line.strip()}")
            return None
        
        return {
            'frameid': frameid,
            'episode_id': episode_id,
            'score': score,
            'duration': duration,
            'unclipped_reward': unclipped_reward,
            'action': action,
            'gaze_positions': gaze_positions,
            'objects': "",
            'relationships': ""
        }
    
    def update_frame_data(self, gaze_df: pd.DataFrame, frame_id: str, 
                         objects_list: List[str], relationships_text: str) -> None:
        """
        Update gaze DataFrame with object and relationship information for a specific frame.
        
        Args:
            gaze_df: Gaze data DataFrame to update
            frame_id: Frame identifier
            objects_list: List of detected object IDs
            relationships_text: Formatted relationships text
        """
        if gaze_df.empty:
            return
        
        # Find the row with matching frame_id
        frame_mask = gaze_df['frameid'] == frame_id
        if not frame_mask.any():
            return
        
        # Update objects column
        objects_text = " , ".join(objects_list) + " , " if objects_list else ""
        gaze_df.loc[frame_mask, 'objects'] = objects_text
        
        # Update relationships column
        gaze_df.loc[frame_mask, 'relationships'] = relationships_text
    
    def get_gaze_positions_for_frame(self, gaze_df: pd.DataFrame, 
                                   frame_id: str) -> List[Tuple[int, int]]:
        """
        Get gaze positions for a specific frame.
        
        Args:
            gaze_df: Gaze data DataFrame
            frame_id: Frame identifier
            
        Returns:
            List of (x, y) gaze position tuples
        """
        if gaze_df.empty:
            return []
        
        frame_data = gaze_df[gaze_df['frameid'] == frame_id]
        if frame_data.empty:
            return []
        
        gaze_positions = frame_data['gaze_positions'].values
        if len(gaze_positions) > 0 and gaze_positions[0]:
            return gaze_positions[0]
        
        return []
    
    def save_updated_gaze_data(self, gaze_df: pd.DataFrame, original_path: str) -> str:
        """
        Save the updated gaze DataFrame to a new file.
        
        Args:
            gaze_df: Updated gaze data DataFrame
            original_path: Path to the original text file
            
        Returns:
            Path to the new saved file
        """
        # Create new filename with _with_relationships suffix
        new_path = original_path.replace(".txt", "_with_relationships.txt")
        
        # Remove gaze_positions column before saving (as it contains complex data)
        df_to_save = gaze_df.drop(columns=['gaze_positions'], errors='ignore')
        
        # Save to CSV format
        df_to_save.to_csv(new_path, index=False)
        
        return new_path
    
    def get_frame_info(self, gaze_df: pd.DataFrame, frame_id: str) -> Optional[Dict[str, Any]]:
        """
        Get all information for a specific frame.
        
        Args:
            gaze_df: Gaze data DataFrame
            frame_id: Frame identifier
            
        Returns:
            Dictionary containing frame information or None if not found
        """
        if gaze_df.empty:
            return None
        
        frame_data = gaze_df[gaze_df['frameid'] == frame_id]
        if frame_data.empty:
            return None
        
        return frame_data.iloc[0].to_dict()
    
    def validate_gaze_data(self, gaze_df: pd.DataFrame) -> bool:
        """
        Validate that the gaze data has the expected format.
        
        Args:
            gaze_df: Gaze data DataFrame to validate
            
        Returns:
            True if valid, False otherwise
        """
        required_columns = ['frameid', 'episode_id', 'score', 'duration', 
                           'unclipped_reward', 'action', 'gaze_positions']
        
        if gaze_df.empty:
            return False
        
        for col in required_columns:
            if col not in gaze_df.columns:
                return False
        
        return True
