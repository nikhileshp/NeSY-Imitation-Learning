import subprocess
import os
import argparse

# Configuration
ACTIONS = ['noop', 'fire', 'up', 'right', 'left', 'down']
RATIO = 2.0
SEED = 42

def run_command(cmd):
    # Use the specific python executable for nesy-il environment
    # Assuming standard conda path structure based on previous output
    # /home/nikhilesh/software/miniconda3/envs/nesy-il/bin/python
    if cmd[0] == "python":
        cmd[0] = "/home/nikhilesh/software/miniconda3/envs/nesy-il/bin/python"
        
    print(f"Running: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {e}")
        return False

def train_rgb_resnet():
    print("\n=== Training RGB ResNet ===")
    for action in ACTIONS:
        cmd = [
            "python", "train_per_action.py",
            "--action", action,
            "--seed", str(SEED),
            "--ratio", str(RATIO),
            "--model_type", "resnet18",
            "--epochs", "1",
        ]
        run_command(cmd)

def train_rgb_cnn():
    print("\n=== Training RGB CNN ===")
    for action in ACTIONS:
        cmd = [
            "python", "train_per_action.py",
            "--action", action,
            "--seed", str(SEED),
            "--ratio", str(RATIO),
            "--model_type", "cnn",
            "--epochs", "1",
            "--max_steps", "20"
        ]
        run_command(cmd)

def train_mlp_bc():
    print("\n=== Training MLP BC ===")
    # experiments/cloning_rdn.py handles all actions if --action is not specified, 
    # but let's run per action to be safe/consistent or just run once if it supports all.
    # Looking at cloning_rdn.py, it takes --action.
    for action in ACTIONS:
        cmd = [
            "python", "experiments/cloning_rdn.py",
            "--action", action,
            "--seed", str(SEED),
            "--save_dir", "trained_models/seaquest/all" 
        ]
        # Note: cloning_rdn.py might need modification if it doesn't support save_dir exactly as we want
        # or we just use its default and move files. 
        # Based on my reading, it uses save_dir/negpos_{ratio}_mlp_64_32_bc/{action}/seed_{seed}
        # which matches our target structure in trained_models.
        run_command(cmd)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip_rgb_resnet", action="store_true")
    parser.add_argument("--skip_rgb_cnn", action="store_true")
    parser.add_argument("--skip_mlp", action="store_true")
    args = parser.parse_args()
    
    if not args.skip_rgb_resnet:
        train_rgb_resnet()
        
    if not args.skip_rgb_cnn:
        train_rgb_cnn()
        
    if not args.skip_mlp:
        train_mlp_bc()
        
    print("\nAll training tasks completed.")

if __name__ == "__main__":
    main()
