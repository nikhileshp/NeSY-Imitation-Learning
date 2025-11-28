import os
import re
import glob
import pandas as pd
import numpy as np

# Configuration
base_dir_new = "rdn_models/seaquest/new_runs"
base_dir_all = "rdn_models/seaquest/all"

tree_counts = [1]  # Comparing 1 tree configurations
actions = ["fire", "up", "down", "left", "right", "noop"]
seeds = [1729, 42, 123, 456, 789]

results_new = []
results_all = []

# Extract from new_runs
print("="*80)
print("Extracting from NEW_RUNS (depth=3, no penalty)")
print("="*80)
for trees in tree_counts:
    dir_name = f"negpos_2_trees_{trees}_depth_3_new_all"
    for action in actions:
        action_dir = os.path.join(base_dir_new, dir_name, action)
        
        if not os.path.exists(action_dir):
            print(f"Warning: Directory {action_dir} not found")
            continue
        
        f1_scores = []
        auc_pr_scores = []
        
        for seed in seeds:
            log_path = os.path.join(action_dir, f"action_test_infer_seed_{seed}.log")
            
            if not os.path.exists(log_path):
                continue
            
            with open(log_path, "r") as f:
                content = f.read()
            
            f1_match = re.search(r"%   F1        = ([\d\.]+)", content)
            auc_pr_match = re.search(r"%   AUC PR    = ([\d\.]+)", content)
            
            if f1_match:
                f1_scores.append(float(f1_match.group(1)))
            if auc_pr_match:
                auc_pr_scores.append(float(auc_pr_match.group(1)))
        
        if f1_scores:
            results_new.append({
                "Config": "new_runs",
                "Trees": trees,
                "Action": action,
                "F1_Mean": np.mean(f1_scores),
                "F1_Std": np.std(f1_scores),
                "F1_Min": np.min(f1_scores),
                "F1_Max": np.max(f1_scores),
                "AUC_PR_Mean": np.mean(auc_pr_scores) if auc_pr_scores else None,
                "AUC_PR_Std": np.std(auc_pr_scores) if auc_pr_scores else None,
                "N_Seeds": len(f1_scores)
            })

# Extract from all (grounding_penalty_0.1)
print("\nExtracting from ALL (depth=3, grounding_penalty=0.1)")
print("="*80)
for trees in tree_counts:
    dir_name = f"negpos_2_trees_{trees}_depth_3_grounding_penalty_0.1"
    for action in actions:
        action_dir = os.path.join(base_dir_all, dir_name, action)
        
        if not os.path.exists(action_dir):
            continue
        
        f1_scores = []
        auc_pr_scores = []
        
        for seed in seeds:
            log_path = os.path.join(action_dir, f"action_test_infer_seed_{seed}.log")
            
            if not os.path.exists(log_path):
                continue
            
            with open(log_path, "r") as f:
                content = f.read()
            
            f1_match = re.search(r"%   F1        = ([\d\.]+)", content)
            auc_pr_match = re.search(r"%   AUC PR    = ([\d\.]+)", content)
            
            if f1_match:
                f1_scores.append(float(f1_match.group(1)))
            if auc_pr_match:
                auc_pr_scores.append(float(auc_pr_match.group(1)))
        
        if f1_scores:
            results_all.append({
                "Config": "grounding_penalty_0.1",
                "Trees": trees,
                "Action": action,
                "F1_Mean": np.mean(f1_scores),
                "F1_Std": np.std(f1_scores),
                "F1_Min": np.min(f1_scores),
                "F1_Max": np.max(f1_scores),
                "AUC_PR_Mean": np.mean(auc_pr_scores) if auc_pr_scores else None,
                "AUC_PR_Std": np.std(auc_pr_scores) if auc_pr_scores else None,
                "N_Seeds": len(f1_scores)
            })

# Combine results
all_results = results_new + results_all
df = pd.DataFrame(all_results)

if df.empty:
    print("\nNo results found!")
else:
    print("\n" + "="*80)
    print("DETAILED COMPARISON: New Runs vs Grounding Penalty (1 tree, depth=3)")
    print("="*80)
    
    # Side-by-side comparison with both metrics
    for action in actions:
        action_data = df[df["Action"] == action].sort_values("Config")
        if not action_data.empty:
            print(f"\n{action.upper()}:")
            for _, row in action_data.iterrows():
                config_label = "NEW" if row['Config'] == 'new_runs' else "GP_0.1"
                f1_str = f"{row['F1_Mean']:.4f} ± {row['F1_Std']:.4f}"
                auc_pr_str = f"{row['AUC_PR_Mean']:.4f} ± {row['AUC_PR_Std']:.4f}" if pd.notna(row['AUC_PR_Mean']) else "N/A"
                print(f"  {config_label:8s}: F1={f1_str}, AUC_PR={auc_pr_str}")
                print(f"             F1 range=[{row['F1_Min']:.4f}, {row['F1_Max']:.4f}], n={int(row['N_Seeds'])}")
            
            # Calculate differences if both configs exist
            if len(action_data) == 2:
                new_data = action_data[action_data['Config'] == 'new_runs'].iloc[0]
                gp_data = action_data[action_data['Config'] == 'grounding_penalty_0.1'].iloc[0]
                
                f1_diff = new_data['F1_Mean'] - gp_data['F1_Mean']
                f1_pct = (f1_diff / gp_data['F1_Mean']) * 100
                
                if pd.notna(new_data['AUC_PR_Mean']) and pd.notna(gp_data['AUC_PR_Mean']):
                    auc_diff = new_data['AUC_PR_Mean'] - gp_data['AUC_PR_Mean']
                    auc_pct = (auc_diff / gp_data['AUC_PR_Mean']) * 100
                    auc_str = f", AUC_PR: {'+' if auc_diff > 0 else ''}{auc_diff:.4f} ({'+' if auc_pct > 0 else ''}{auc_pct:.2f}%)"
                else:
                    auc_str = ""
                
                print(f"  {'Δ':8s}: F1: {'+' if f1_diff > 0 else ''}{f1_diff:.4f} "
                      f"({'+' if f1_pct > 0 else ''}{f1_pct:.2f}%){auc_str}")
    
    # Summary tables
    print("\n" + "="*80)
    print("Summary Table: F1 Mean")
    print("="*80)
    pivot_f1 = df.pivot_table(index="Action", columns="Config", values="F1_Mean", aggfunc='first')
    if 'new_runs' in pivot_f1.columns and 'grounding_penalty_0.1' in pivot_f1.columns:
        pivot_f1['Difference'] = pivot_f1['new_runs'] - pivot_f1['grounding_penalty_0.1']
        pivot_f1['% Change'] = (pivot_f1['Difference'] / pivot_f1['grounding_penalty_0.1']) * 100
    print(pivot_f1.to_string())
    
    print("\n" + "="*80)
    print("Summary Table: AUC PR Mean")
    print("="*80)
    pivot_auc = df.pivot_table(index="Action", columns="Config", values="AUC_PR_Mean", aggfunc='first')
    if 'new_runs' in pivot_auc.columns and 'grounding_penalty_0.1' in pivot_auc.columns:
        pivot_auc['Difference'] = pivot_auc['new_runs'] - pivot_auc['grounding_penalty_0.1']
        pivot_auc['% Change'] = (pivot_auc['Difference'] / pivot_auc['grounding_penalty_0.1']) * 100
    print(pivot_auc.to_string())
    
    print("\n" + "="*80)
    print("Summary Table: F1 Standard Deviation (Stability)")
    print("="*80)
    pivot_std = df.pivot_table(index="Action", columns="Config", values="F1_Std", aggfunc='first')
    if 'new_runs' in pivot_std.columns and 'grounding_penalty_0.1' in pivot_std.columns:
        pivot_std['Difference'] = pivot_std['new_runs'] - pivot_std['grounding_penalty_0.1']
    print(pivot_std.to_string())
    print("(Negative difference = NEW is more stable)")
    
    # Overall comparison
    print("\n" + "="*80)
    print("Overall Statistics")
    print("="*80)
    summary = df.groupby("Config").agg({
        "F1_Mean": "mean",
        "F1_Std": "mean",
        "AUC_PR_Mean": "mean",
        "AUC_PR_Std": "mean"
    }).round(4)
    print(summary.to_string())
    
    # Winner analysis
    print("\n" + "="*80)
    print("Winner Analysis")
    print("="*80)
    if 'new_runs' in pivot_f1.columns and 'grounding_penalty_0.1' in pivot_f1.columns:
        new_wins_f1 = (pivot_f1['new_runs'] > pivot_f1['grounding_penalty_0.1']).sum()
        gp_wins_f1 = (pivot_f1['grounding_penalty_0.1'] > pivot_f1['new_runs']).sum()
        print(f"F1 Score Winners:")
        print(f"  NEW_RUNS: {new_wins_f1} actions")
        print(f"  GP_0.1:   {gp_wins_f1} actions")
        
    if 'new_runs' in pivot_auc.columns and 'grounding_penalty_0.1' in pivot_auc.columns:
        new_wins_auc = (pivot_auc['new_runs'] > pivot_auc['grounding_penalty_0.1']).sum()
        gp_wins_auc = (pivot_auc['grounding_penalty_0.1'] > pivot_auc['new_runs']).sum()
        print(f"\nAUC PR Winners:")
        print(f"  NEW_RUNS: {new_wins_auc} actions")
        print(f"  GP_0.1:   {gp_wins_auc} actions")
