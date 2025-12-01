#Change one line of the bk file to set max depth

import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--max_depth", type=int, required=True, help="Maximum depth for reasoning")
parser.add_argument("--base_dir", type=str, default="data/seaquest/all", help="Base directory containing action folders")
args = parser.parse_args()

base_dir = args.base_dir

# List all directories in base_dir
for action in os.listdir(base_dir):
    action_dir = os.path.join(base_dir, action)
    if os.path.isdir(action_dir):
        # Check for train/bk.pl
        train_dir = os.path.join(action_dir, "train")
        bk_file_path = os.path.join(train_dir, "train_bk.txt")
        
        if os.path.exists(bk_file_path):
            # Read the existing bk.pl file
            with open(bk_file_path, "r") as file:
                lines = file.readlines()
            
            # Modify the max_depth line
            with open(bk_file_path, "w") as file:
                for line in lines:
                    if line.startswith("setParam: max"):
                        file.write(f"setParam: maxTreeDepth={args.max_depth}.\n")
                    else:
                        file.write(line)
            
            print(f"Updated max_depth in {bk_file_path} to {args.max_depth}")
        else:
            print(f"bk.txt not found in {train_dir}")

        # Also check for test/bk.pl
        test_dir = os.path.join(action_dir, "test")
        bk_file_path = os.path.join(test_dir, "test_bk.txt")

        if os.path.exists(bk_file_path):
            # Read the existing bk.pl file
            with open(bk_file_path, "r") as file:
                lines = file.readlines()
            
            # Modify the max_depth line
            with open(bk_file_path, "w") as file:
                for line in lines:
                    if line.startswith("setParam: max"):
                        file.write(f"setParam: maxTreeDepth={args.max_depth}.\n")
                    else:
                        file.write(line)
            
            print(f"Updated max_depth in {bk_file_path} to {args.max_depth}")
        else:
            print(f"test_bk.txt not found in {test_dir}")

    