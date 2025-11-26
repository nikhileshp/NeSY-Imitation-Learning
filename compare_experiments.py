import os
import re
import numpy as np
import pandas as pd
from collections import defaultdict

# Configuration
base_dir = "rdn_models/seaquest/all"
actions = ["fire", "up", "down", "left", "right", "noop"]
seeds = [1729, 42, 123, 456, 789]
depths = [1, 2, 3]
negpos_ratios = [1, 2]
grounding_penalty = "0.1"
trees = 1

def parse_log_file(filepath):
    metrics = {}
    try:
        with open(filepath, 'r') as f:
            content = f.read()
            # Extract metrics using regex
            # Example: "%   F1        = 0.716340"
            # We look for the pattern starting with % to avoid matching "Best F1"
            f1_match = re.search(r'%\s+F1\s*[:=]\s*([\d\.]+)', content)
            auc_pr_match = re.search(r'AUC PR\s*[:=]\s*([\d\.]+)', content)
            auc_roc_match = re.search(r'AUC ROC\s*[:=]\s*([\d\.]+)', content)
            
            if f1_match: metrics['F1'] = float(f1_match.group(1))
            if auc_pr_match: metrics['AUC_PR'] = float(auc_pr_match.group(1))
            if auc_roc_match: metrics['AUC_ROC'] = float(auc_roc_match.group(1))
            
    except FileNotFoundError:
        # print(f"File not found: {filepath}")
        pass
    return metrics

results = []

for depth in depths:
    # Construct directory path
    # Note: The directory name seems to always start with negpos_2 based on ls output
    # "negpos_2_trees_1_depth_2_grounding_penalty_0.1"
    dir_name = f"negpos_2_trees_{trees}_depth_{depth}_grounding_penalty_{grounding_penalty}"
    
    for action in actions:
        for negpos in negpos_ratios:
            f1_scores = []
            auc_pr_scores = []
            auc_roc_scores = []
            
            for seed in seeds:
                # Construct log filename
                # action_test_infer_seed_123_negpos_1.log
                log_filename = f"action_test_infer_seed_{seed}_negpos_{negpos}.log"
                log_path = os.path.join(base_dir, dir_name, action, log_filename)
                
                if not os.path.exists(log_path) and negpos == 2:
                    # Fallback for negpos_2: try without suffix
                    log_filename = f"action_test_infer_seed_{seed}.log"
                    log_path = os.path.join(base_dir, dir_name, action, log_filename)
                
                # Debug print
                # print(f"Checking: {log_path}")
                
                metrics = parse_log_file(log_path)
                
                if metrics:
                    # print(f"Found metrics in {log_path}: {metrics}")
                    if 'F1' in metrics: f1_scores.append(metrics['F1'])
                    if 'AUC_PR' in metrics: auc_pr_scores.append(metrics['AUC_PR'])
                    if 'AUC_ROC' in metrics: auc_roc_scores.append(metrics['AUC_ROC'])
            
            if f1_scores:
                results.append({
                    'Depth': depth,
                    'NegPos': negpos,
                    'Action': action,
                    'F1_Mean': np.mean(f1_scores),
                    'F1_Std': np.std(f1_scores),
                    'AUC_PR_Mean': np.mean(auc_pr_scores),
                    'AUC_PR_Std': np.std(auc_pr_scores),
                    'AUC_ROC_Mean': np.mean(auc_roc_scores),
                    'AUC_ROC_Std': np.std(auc_roc_scores),
                    'Count': len(f1_scores)
                })
            else:
                 # print(f"No results for Depth {depth}, NegPos {negpos}, Action {action}")
                 pass

if not results:
    print("No results found! Checking directory structure...")
    if os.path.exists(base_dir):
        print(f"Base dir {base_dir} exists.")
        print("Subdirs:", os.listdir(base_dir))
    else:
        print(f"Base dir {base_dir} does NOT exist.")
    exit(1)

df = pd.DataFrame(results)

df = pd.DataFrame(results)

# Aggregate across actions for overall stats
overall_results = []
for depth in depths:
    for negpos in negpos_ratios:
        subset = df[(df['Depth'] == depth) & (df['NegPos'] == negpos)]
        if not subset.empty:
            overall_results.append({
                'Depth': depth,
                'NegPos': negpos,
                'F1_Mean': subset['F1_Mean'].mean(),
                'F1_Std': subset['F1_Mean'].std(), # Std of means across actions
                'AUC_PR_Mean': subset['AUC_PR_Mean'].mean(),
                'AUC_PR_Std': subset['AUC_PR_Mean'].std(),
                'AUC_ROC_Mean': subset['AUC_ROC_Mean'].mean(),
                'AUC_ROC_Std': subset['AUC_ROC_Mean'].std()
            })

overall_df = pd.DataFrame(overall_results)

print("\n=== Detailed Results by Action ===")
print(df.to_string(index=False))

print("\n=== Overall Results (Averaged across Actions) ===")
print(overall_df.to_string(index=False))
