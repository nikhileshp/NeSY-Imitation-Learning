#!/usr/bin/env python3
"""
Script to create train_bk_pi.txt files for all actions in seaquest/all.
Each train_bk_pi.txt will contain all modes from train_bk.txt plus additional 
inradius modes for privileged information.
"""

import os
import sys

# Define the base directory
base_dir = "data/seaquest/all"

# Define the primitive actions
primitive_actions = ["noop", "fire", "up", "right", "left", "down"]

# Define the additional privileged modes to add
additional_pi_modes = [
    "mode: inradiusenemy(+state, -enemy).",
    "mode: inradiusmissile(+state, -missile).",
    "mode: inradiussubmarine(+state, -submarine).",
    "mode: inradiusdiver(+state, -diver)."
]

def create_train_bk_pi(action):
    """Create train_bk_pi.txt for a given action"""
    train_bk_path = os.path.join(base_dir, action, "train", "train_bk.txt")
    train_bk_pi_path = os.path.join(base_dir, action, "train", "train_bk_pi.txt")
    
    if not os.path.exists(train_bk_path):
        print(f"Warning: {train_bk_path} does not exist. Skipping {action}.")
        return
    
    # Read the original train_bk.txt
    with open(train_bk_path, 'r') as f:
        original_content = f.read()
    
    # Find where to insert the new modes (after the last mode: line, before bridger: lines)
    lines = original_content.split('\n')
    
    # Find the last line that starts with "mode:"
    last_mode_index = -1
    for i in range(len(lines) - 1, -1, -1):
        if lines[i].strip().startswith("mode:"):
            last_mode_index = i
            break
    
    if last_mode_index == -1:
        print(f"Warning: No mode declarations found in {train_bk_path}")
        return
    
    # Insert the additional PI modes after the last mode
    new_lines = lines[:last_mode_index + 1] + additional_pi_modes + lines[last_mode_index + 1:]
    
    # Write to train_bk_pi.txt
    with open(train_bk_pi_path, 'w') as f:
        f.write('\n'.join(new_lines))
    
    print(f"Created {train_bk_pi_path}")

def main():
    for action in primitive_actions:
        create_train_bk_pi(action)
    
    print("\nAll train_bk_pi.txt files have been created successfully!")

if __name__ == "__main__":
    main()
