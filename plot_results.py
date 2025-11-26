import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Configuration
base_dir = "rdn_models/seaquest/all"
actions = ["fire", "up", "down", "left", "right", "noop"]
seeds = [1729, 42, 123, 456, 789]
depths = [1, 2, 3]
negpos_ratios = [1, 2]
grounding_penalty = "0.1"
trees = 1
output_dir = "plots"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def parse_log_file(filepath):
    metrics = {}
    try:
        with open(filepath, 'r') as f:
            content = f.read()
            # Extract metrics using regex
            # Example: "%   F1        = 0.716340"
            f1_match = re.search(r'%\s+F1\s*[:=]\s*([\d\.]+)', content)
            auc_pr_match = re.search(r'AUC PR\s*[:=]\s*([\d\.]+)', content)
            auc_roc_match = re.search(r'AUC ROC\s*[:=]\s*([\d\.]+)', content)
            
            if f1_match: metrics['F1'] = float(f1_match.group(1))
            if auc_pr_match: metrics['AUC_PR'] = float(auc_pr_match.group(1))
            if auc_roc_match: metrics['AUC_ROC'] = float(auc_roc_match.group(1))
            
    except FileNotFoundError:
        pass
    return metrics

results = []

print("Parsing logs...")
for depth in depths:
    dir_name = f"negpos_2_trees_{trees}_depth_{depth}_grounding_penalty_{grounding_penalty}"
    
    for action in actions:
        for negpos in negpos_ratios:
            f1_scores = []
            auc_pr_scores = []
            auc_roc_scores = []
            
            for seed in seeds:
                log_filename = f"action_test_infer_seed_{seed}_negpos_{negpos}.log"
                log_path = os.path.join(base_dir, dir_name, action, log_filename)
                
                if not os.path.exists(log_path) and negpos == 2:
                    # Fallback for negpos_2: try without suffix
                    log_filename = f"action_test_infer_seed_{seed}.log"
                    log_path = os.path.join(base_dir, dir_name, action, log_filename)
                
                metrics = parse_log_file(log_path)
                
                if metrics:
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
                    'AUC_ROC_Std': np.std(auc_roc_scores)
                })

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
print("Data aggregated. Generating plots...")

metrics_to_plot = [
    ('F1', 'F1 Score'),
    ('AUC_PR', 'AUC PR'),
    ('AUC_ROC', 'AUC ROC')
]

# Define colors for actions to keep them consistent
action_colors = {
    'fire': 'red',
    'up': 'blue',
    'down': 'green',
    'left': 'purple',
    'right': 'orange',
    'noop': 'gray'
}

# Define line styles for NegPos ratios
negpos_styles = {
    1: '-',  # Solid for NegPos 1
    2: '--'  # Dashed for NegPos 2
}

for metric_key, metric_label in metrics_to_plot:
    plt.figure(figsize=(12, 8)) # Larger figure for many lines
    
    for action in actions:
        for negpos in negpos_ratios:
            # Filter data
            subset = df[(df['Action'] == action) & (df['NegPos'] == negpos)]
            subset = subset.sort_values('Depth')
            
            if not subset.empty:
                x = subset['Depth']
                y = subset[f'{metric_key}_Mean']
                
                label = f"{action} (NP={negpos})"
                color = action_colors.get(action, 'black')
                style = negpos_styles.get(negpos, '-')
                
                plt.plot(x, y, marker='o', label=label, color=color, linestyle=style)
    
    plt.title(f'{metric_label} vs Tree Depth by Action')
    plt.xlabel('Tree Depth')
    plt.ylabel(metric_label)
    plt.xticks(depths)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left') # Legend outside
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout() # Adjust layout to fit legend
    
    filename = f"{output_dir}/{metric_key.lower()}_vs_depth_by_action.png"
    plt.savefig(filename)
    print(f"Saved {filename}")
    plt.close()

print("Done.")
