import os
import re
import pandas as pd
import numpy as np

# Configuration
base_dir = "rdn_models/seaquest/all"
tree_counts = [1]
actions = ["fire", "up", "down", "left", "right", "noop"]
seeds = [1729, 42, 123, 456, 789]

results = []

for trees in tree_counts:
    dir_name_3 = f"negpos_2_trees_{trees}_depth_3_grounding_penalty_0.1"
    dir_name_1 = f"negpos_2_trees_{trees}_depth_1_grounding_penalty_0.1"
    for action in actions:
        action_dir_3 = os.path.join(base_dir, dir_name_3, action)
        action_dir_1 = os.path.join(base_dir, dir_name_1, action)
        if not os.path.exists(action_dir_3):
            print(f"Warning: Directory {action_dir_3} not found")
            continue
        
        # Collect metrics for testNegPosRatio=2 (original)
        f1_scores_ratio2 = []
        auc_pr_scores_ratio2 = []
        
        # Collect metrics for testNegPosRatio=1
        f1_scores_ratio1 = []
        auc_pr_scores_ratio1 = []
        
        for seed in seeds:
            # Check testNegPosRatio=2 logs
            log_path_ratio3 = os.path.join(action_dir_3, f"action_test_infer_seed_{seed}_negpos_1.log")
            if os.path.exists(log_path_ratio3):
                with open(log_path_ratio3, "r") as f:
                    content = f.read()
                f1_match = re.search(r"%   F1        = ([\d\.]+)", content)
                auc_pr_match = re.search(r"%   AUC PR    = ([\d\.]+)", content)
                if f1_match:
                    f1_scores_ratio2.append(float(f1_match.group(1)))
                if auc_pr_match:
                    auc_pr_scores_ratio2.append(float(auc_pr_match.group(1)))
            
            # Check testNegPosRatio=1 logs
            log_path_ratio1 = os.path.join(action_dir_1, f"action_test_infer_seed_{seed}_negpos_1.log")
            if os.path.exists(log_path_ratio1):
                with open(log_path_ratio1, "r") as f:
                    content = f.read()
                f1_match = re.search(r"%   F1        = ([\d\.]+)", content)
                auc_pr_match = re.search(r"%   AUC PR    = ([\d\.]+)", content)
                if f1_match:
                    f1_scores_ratio1.append(float(f1_match.group(1)))
                if auc_pr_match:
                    auc_pr_scores_ratio1.append(float(auc_pr_match.group(1)))
        
        # Add results for ratio=2
        if f1_scores_ratio2:
            results.append({
                "Trees": 3,
                "Action": action,
                "TestRatio": "1:1 (neg:pos)",
                "F1_Mean": np.mean(f1_scores_ratio2),
                "F1_Std": np.std(f1_scores_ratio2),
                "F1_Min": np.min(f1_scores_ratio2),
                "F1_Max": np.max(f1_scores_ratio2),
                "AUC_PR_Mean": np.mean(auc_pr_scores_ratio2) if auc_pr_scores_ratio2 else None,
                "AUC_PR_Std": np.std(auc_pr_scores_ratio2) if auc_pr_scores_ratio2 else None,
                "N_Seeds": len(f1_scores_ratio2)
            })
        
        # Add results for ratio=1
        if f1_scores_ratio1:
            results.append({
                "Trees": 1,
                "Action": action,
                "TestRatio": "1:1 (neg:pos)",
                "F1_Mean": np.mean(f1_scores_ratio1),
                "F1_Std": np.std(f1_scores_ratio1),
                "F1_Min": np.min(f1_scores_ratio1),
                "F1_Max": np.max(f1_scores_ratio1),
                "AUC_PR_Mean": np.mean(auc_pr_scores_ratio1) if auc_pr_scores_ratio1 else None,
                "AUC_PR_Std": np.std(auc_pr_scores_ratio1) if auc_pr_scores_ratio1 else None,
                "N_Seeds": len(f1_scores_ratio1)
            })

df = pd.DataFrame(results)

if df.empty:
    print("No results found!")
else:
    print("\n" + "="*80)
    print("Test Neg:Pos Ratio Comparison (depth=3, grounding_penalty=0.1, 1 tree)")
    print("="*80)
    print(df.to_string(index=False))
    
    # Per-action detailed comparison
    print("\n" + "="*80)
    print("Per-Action Comparison (Mean ± Std Dev)")
    print("="*80)
    for action in actions:
        action_data = df[df["Action"] == action].sort_values("TestRatio", ascending=False)
        if not action_data.empty:
            print(f"\n{action.upper()}:")
            for _, row in action_data.iterrows():
                ratio_label = row['TestRatio']
                f1_str = f"{row['F1_Mean']:.4f} ± {row['F1_Std']:.4f}"
                auc_pr_str = f"{row['AUC_PR_Mean']:.4f} ± {row['AUC_PR_Std']:.4f}" if pd.notna(row['AUC_PR_Mean']) else "N/A"
                print(f"  {ratio_label:15s}: F1={f1_str}, AUC_PR={auc_pr_str}")
                print(f"                   F1 range=[{row['F1_Min']:.4f}, {row['F1_Max']:.4f}], n={int(row['N_Seeds'])}")
            
            # Calculate difference if both ratios exist
            if len(action_data) == 2:
                ratio2_data = action_data[action_data['TestRatio'] == '2:1 (neg:pos)'].iloc[0]
                ratio1_data = action_data[action_data['TestRatio'] == '1:1 (neg:pos)'].iloc[0]
                
                f1_diff = ratio1_data['F1_Mean'] - ratio2_data['F1_Mean']
                f1_pct = (f1_diff / ratio2_data['F1_Mean']) * 100
                
                if pd.notna(ratio1_data['AUC_PR_Mean']) and pd.notna(ratio2_data['AUC_PR_Mean']):
                    auc_diff = ratio1_data['AUC_PR_Mean'] - ratio2_data['AUC_PR_Mean']
                    auc_pct = (auc_diff / ratio2_data['AUC_PR_Mean']) * 100
                    auc_str = f", AUC_PR: {'+' if auc_diff > 0 else ''}{auc_diff:.4f} ({'+' if auc_pct > 0 else ''}{auc_pct:.2f}%)"
                else:
                    auc_str = ""
                
                print(f"  {'Δ (1:1 - 2:1)':15s}: F1: {'+' if f1_diff > 0 else ''}{f1_diff:.4f} "
                      f"({'+' if f1_pct > 0 else ''}{f1_pct:.2f}%){auc_str}")
    
    # Summary tables
    print("\n" + "="*80)
    print("Summary Table: F1 Mean")
    print("="*80)
    pivot_f1 = df.pivot_table(index="Action", columns="TestRatio", values="F1_Mean")
    if '2:1 (neg:pos)' in pivot_f1.columns and '1:1 (neg:pos)' in pivot_f1.columns:
        pivot_f1['Difference'] = pivot_f1['1:1 (neg:pos)'] - pivot_f1['2:1 (neg:pos)']
        pivot_f1['% Change'] = (pivot_f1['Difference'] / pivot_f1['2:1 (neg:pos)']) * 100
    print(pivot_f1.to_string())
    
    print("\n" + "="*80)
    print("Summary Table: AUC PR Mean")
    print("="*80)
    pivot_auc = df.pivot_table(index="Action", columns="TestRatio", values="AUC_PR_Mean")
    if '2:1 (neg:pos)' in pivot_auc.columns and '1:1 (neg:pos)' in pivot_auc.columns:
        pivot_auc['Difference'] = pivot_auc['1:1 (neg:pos)'] - pivot_auc['2:1 (neg:pos)']
        pivot_auc['% Change'] = (pivot_auc['Difference'] / pivot_auc['2:1 (neg:pos)']) * 100
    print(pivot_auc.to_string())
    
    print("\n" + "="*80)
    print("Summary Table: F1 Standard Deviation (Stability)")
    print("="*80)
    pivot_std = df.pivot_table(index="Action", columns="TestRatio", values="F1_Std")
    print(pivot_std.to_string())
    
    # Overall statistics
    print("\n" + "="*80)
    print("Overall Statistics")
    print("="*80)
    summary = df.groupby("TestRatio").agg({
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
    if '2:1 (neg:pos)' in pivot_f1.columns and '1:1 (neg:pos)' in pivot_f1.columns:
        ratio1_wins_f1 = (pivot_f1['1:1 (neg:pos)'] > pivot_f1['2:1 (neg:pos)']).sum()
        ratio2_wins_f1 = (pivot_f1['2:1 (neg:pos)'] > pivot_f1['1:1 (neg:pos)']).sum()
        print(f"F1 Score Winners:")
        print(f"  1:1 ratio: {ratio1_wins_f1} actions")
        print(f"  2:1 ratio: {ratio2_wins_f1} actions")
        
    if '2:1 (neg:pos)' in pivot_auc.columns and '1:1 (neg:pos)' in pivot_auc.columns:
        ratio1_wins_auc = (pivot_auc['1:1 (neg:pos)'] > pivot_auc['2:1 (neg:pos)']).sum()
        ratio2_wins_auc = (pivot_auc['2:1 (neg:pos)'] > pivot_auc['1:1 (neg:pos)']).sum()
        print(f"\nAUC PR Winners:")
        print(f"  1:1 ratio: {ratio1_wins_auc} actions")
        print(f"  2:1 ratio: {ratio2_wins_auc} actions")
