import os
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

# Define paths
data_folder = "55_RZ_2464601_Aug-11-10-18-09"
txt_file = "55_RZ_2464601_Aug-11-10-18-09.txt"

# Read the file manually to handle the complex gaze_positions column
def read_custom_csv(filename):
    data = []
    with open(filename, 'r') as f:
        header = f.readline().strip().split(',')
        # Expected columns: qframe_id,episode_id,score,duration(ms),unclipped_reward,action,gaze_positions
        
        for line in f:
            parts = line.strip().split(',')
            if len(parts) >= 6:
                # First 6 columns are fixed
                row_data = {
                    'qframe_id': parts[0],
                    'episode_id': parts[1],
                    'score': parts[2],
                    'duration(ms)': parts[3],
                    'unclipped_reward': parts[4],
                    'action': parts[5],
                    'gaze_positions': ','.join(parts[6:])  # Rest are gaze positions
                }
                data.append(row_data)
    
    return pd.DataFrame(data)

# Read the data
df = read_custom_csv(txt_file)

# Convert numeric columns
df['duration(ms)'] = pd.to_numeric(df['duration(ms)'], errors='coerce')
df['action'] = pd.to_numeric(df['action'], errors='coerce')

print(f"Total rows in data: {len(df)}")
print(f"Columns: {df.columns.tolist()}")
print(f"\nFirst few qframe_id values:")
print(df['qframe_id'].head())
print("\nControls: Press SPACE for next frame, ESC to exit\n")

# Function to parse gaze positions from comma-separated values
def parse_gaze_positions(gaze_str):
    """Parse gaze_positions string to extract (x, y) coordinate pairs"""
    try:
        # Check if gaze_str is null or empty
        if pd.isna(gaze_str) or str(gaze_str).strip().lower() in ['null', 'nan', '']:
            return None
        
        # Split by commas and convert to float, filtering out null/invalid values
        values = []
        for x in str(gaze_str).split(','):
            x = x.strip()
            if x and x.lower() not in ['null', 'nan']:
                try:
                    values.append(float(x))
                except ValueError:
                    continue  # Skip invalid values
        
        # Group into (x, y) pairs
        gaze_points = []
        for i in range(0, len(values), 2):
            if i + 1 < len(values):
                gaze_points.append((values[i], values[i+1]))
        
        return gaze_points if gaze_points else None
    except Exception as e:
        print(f"Error parsing gaze positions: {e}")
        return None

# Global variables for keyboard control and previous gaze data
current_idx = 0
should_exit = False
previous_gaze_points = None

def on_key(event):
    """Handle keyboard events"""
    global current_idx, should_exit
    
    if event.key == ' ':  # Space key
        plt.close()  # Close current figure to move to next
    elif event.key == 'escape':  # Escape key
        should_exit = True
        plt.close()

# Iterate through each frame
for idx, row in df.iterrows():
    if should_exit:
        print("\nExiting visualization...")
        break
        
    frame_id = row['qframe_id']
    gaze_positions = row['gaze_positions']
    
    # Construct frame filename - extract just the numeric ID
    # frame_id looks like "RZ_2464601_1", we want just "1"
    numeric_id = frame_id.split('_')[-1]
    frame_filename = f"RZ_2464601_{numeric_id}.png"
    frame_path = os.path.join(data_folder, frame_filename)
    
    # Check if frame exists
    if not os.path.exists(frame_path):
        print(f"Frame {frame_filename} not found, skipping...")
        continue
    
    # Load the frame
    img = Image.open(frame_path)
    
    # Parse gaze positions
    gaze_points = parse_gaze_positions(gaze_positions)
    
    # Check if we should use previous frame's gaze data
    use_previous = False
    if gaze_points is None or len(gaze_points) < 45:
        if previous_gaze_points is not None and len(previous_gaze_points) >= 45:
            gaze_points = previous_gaze_points
            use_previous = True
            print(f"Frame {numeric_id}: Using previous frame gaze data ({len(gaze_points)} points)")
        else:
            print(f"Frame {numeric_id}: Insufficient gaze data ({len(gaze_points) if gaze_points else 0} points) and no valid previous data")
    else:
        # Update previous_gaze_points only if current frame has sufficient data
        previous_gaze_points = gaze_points
        print(f"Frame {numeric_id}: {len(gaze_points)} gaze points")
    
    # Create visualization with fixed figure size
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)
    ax.imshow(img)
    
    # Plot gaze positions if available
    if gaze_points is not None and len(gaze_points) > 0:
        # Extract x and y coordinates
        x_coords = [point[0] for point in gaze_points]
        y_coords = [point[1] for point in gaze_points]
        
        # Use different color if using previous frame data
        color = 'orange' if use_previous else 'red'
        
        # Plot all gaze points
        ax.scatter(x_coords, y_coords, c=color, s=30, alpha=0.6, 
                  edgecolors='white', linewidths=1)
        
        # Optionally, draw a path connecting the gaze points
        ax.plot(x_coords, y_coords, color=color, alpha=0.3, linewidth=1)
    
    # Add title with frame info
    action = row['action']
    duration = row['duration(ms)']
    gaze_count = len(gaze_points) if gaze_points else 0
    status = " (from previous frame)" if use_previous else ""
    title = f"Frame {frame_id} | Duration: {duration}ms | Action: {action} | Gaze Points: {gaze_count}{status}"
    ax.set_title(title, fontsize=10)
    ax.axis('off')
    
    # Add controls text at bottom - position relative to figure, not axes
    controls_text = 'Press SPACE for next frame | Press ESC to exit'
    if use_previous:
        controls_text += ' | Orange = Previous Frame Data'
    fig.text(0.5, 0.02, controls_text, ha='center', fontsize=10, style='italic')
    
    # Ensure tight layout to maintain consistent spacing
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    
    # Connect keyboard event handler
    fig.canvas.mpl_connect('key_press_event', on_key)
    
    # Display the plot
    plt.show()
    
    if idx % 5 == 0:
        print(f"Processed frame {idx+1}/{len(df)}")

print("\nVisualization complete!")
