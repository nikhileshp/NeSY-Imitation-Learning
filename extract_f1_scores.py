import os
import re
import glob
import pandas as pd
import numpy as np

base_dir = "rdn_models/seaquest/all"
tree_counts = [1, 10, 20]
actions = ["fire", "up", "down", "left", "right", "noop"]

# Seeds to analyze (should match the seeds used in run_all.sh)
seeds = [1729, 42, 123, 456, 789]

results = []

for trees in tree_counts:
    dir_name = f"negpos_2_trees_{trees}_depth_3_grounding_penalty_0.1"
    for action in actions:
        action_dir = os.path.join(base_dir, dir_name, action)
        
        if not os.path.exists(action_dir):
            print(f"Warning: Directory {action_dir} not found")
            continue
        
        # Collect F1 scores for all seeds
        f1_scores = []
        
        for seed in seeds:
            log_path = os.path.join(action_dir, f"action_test_infer_seed_{seed}.log")
            
            if not os.path.exists(log_path):
                print(f"Warning: Log file not found: {log_path}")
                continue
            
            with open(log_path, "r") as f:
                content = f.read()
            
            # Extract F1 score
            f1_match = re.search(r"%   F1        = ([\d\.]+)", content)
            if f1_match:
                f1_scores.append(float(f1_match.group(1)))
        
        # Calculate statistics if we have data
        if f1_scores:
            results.append({
                "Trees": trees,
                "Action": action,
                "F1_Mean": np.mean(f1_scores),
                "F1_Std": np.std(f1_scores),
                "F1_Min": np.min(f1_scores),
                "F1_Max": np.max(f1_scores),
                "N_Seeds": len(f1_scores)
            })

# Create DataFrame
df = pd.DataFrame(results)

if df.empty:
    print("No results found!")
else:
    print("\n" + "="*80)
    print("F1 Score Statistics Across Seeds (depth=3, grounding_penalty=0.1)")
    print("="*80)
    print(df.to_string(index=False))
    
    # Per-action statistics
    print("\n" + "="*80)
    print("Per-Action Statistics (Mean ± Std Dev)")
    print("="*80)
    for action in actions:
        action_data = df[df["Action"] == action]
        if not action_data.empty:
            print(f"\n{action.upper()}:")
            for _, row in action_data.iterrows():
                print(f"  {row['Trees']:2d} trees: {row['F1_Mean']:.4f} ± {row['F1_Std']:.4f} "
                      f"(min={row['F1_Min']:.4f}, max={row['F1_Max']:.4f}, n={int(row['N_Seeds'])})")
    
    # Pivot for easier comparison by trees
    print("\n" + "="*80)
    print("F1 Mean by Action and Tree Count")
    print("="*80)
    pivot_mean = df.pivot(index="Action", columns="Trees", values="F1_Mean")
    print(pivot_mean.to_string())
    
    print("\n" + "="*80)
    print("F1 Standard Deviation by Action and Tree Count")
    print("="*80)
    pivot_std = df.pivot(index="Action", columns="Trees", values="F1_Std")
    print(pivot_std.to_string())
    
    # Summary statistics
    print("\n" + "="*80)
    print("Summary: Average F1 across all actions")
    print("="*80)
    summary = df.groupby("Trees").agg({
        "F1_Mean": "mean",
        "F1_Std": "mean"
    }).round(4)
    summary.columns = ["Avg_F1_Mean", "Avg_F1_Std"]
    print(summary.to_string())
