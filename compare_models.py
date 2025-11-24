import os
import glob
import re
import pandas as pd

# Define the configurations to compare
configs = [
    {
        "name": "Grounding Penalty 0.1",
        "base_dir": "rdn_models/seaquest/all",
        "pattern": "negpos_2_trees_{trees}_depth_3_grounding_penalty_0.1",
        "trees": [1, 10, 20]
    },
    {
        "name": "New Runs (No Penalty)",
        "base_dir": "rdn_models/seaquest/new_runs",
        "pattern": "negpos_2_trees_{trees}_depth_3_new_all",
        "trees": [1]
    },
    {
        "name": "Old Runs (Per Example Weight)",
        "base_dir": "rdn_models/seaquest/old_runs",
        "pattern": "negpos_2_trees_{trees}_depth_4_per_example_weight_all",
        "trees": [1, 10, 20]
    }
]

actions = ["fire", "up", "down", "left", "right", "noop"]
results = []

for config in configs:
    for trees in config["trees"]:
        dir_name = config["pattern"].format(trees=trees)
        
        for action in actions:
            action_dir = os.path.join(config["base_dir"], dir_name, action)
            
            if not os.path.exists(action_dir):
                # print(f"Warning: Directory {action_dir} not found")
                continue
                
            # Find log files (handle variable naming: action_test_infer.log, fire_test_infer.log, etc.)
            log_files = glob.glob(os.path.join(action_dir, "*test_infer*.log"))
            
            if not log_files:
                # print(f"Warning: No log files found in {action_dir}")
                continue
                
            # Sort and pick the last one
            log_files.sort()
            log_path = log_files[-1]
            
            with open(log_path, "r") as f:
                content = f.read()
                
            # Extract F1
            f1_match = re.search(r"%   F1        = ([\d\.]+)", content)
            
            if f1_match:
                results.append({
                    "Configuration": config["name"],
                    "Trees": trees,
                    "Action": action,
                    "F1": float(f1_match.group(1))
                })

df = pd.DataFrame(results)

if df.empty:
    print("No results found!")
else:
    # Pivot for comparison
    # We want rows to be Action and columns to be (Configuration, Trees)
    pivot_df = df.pivot_table(index="Action", columns=["Configuration", "Trees"], values="F1")
    
    print("\nF1 Score Comparison:")
    print(pivot_df.to_string())
    
    # Also print a flattened version for easier reading if needed
    print("\nDetailed Results:")
    print(df.sort_values(["Configuration", "Trees", "Action"]).to_string(index=False))
