import os
import re

base_path = "/home/nikhilesh/Projects/NeSY-Imitation-Learning/rdn_models/seaquest/all"
configs = {
    "0.01": "negpos_2_trees_1_depth_3_grounding_penalty_0.01",
    "0.1": "negpos_2_trees_1_depth_3_grounding_penalty_0.1_new",
    "None": "negpos_2_trees_1_depth_3"
}
actions = ["down", "fire", "left", "noop", "right", "up"]
seed = "seed_42"

results = {}

for config_name, config_dir in configs.items():
    results[config_name] = {}
    
    # Extract F1 from eval_report
    eval_report_path = os.path.join(base_path, config_dir, f"eval_report_{seed}.txt")
    f1_scores = {}
    if os.path.exists(eval_report_path):
        with open(eval_report_path, "r") as f:
            content = f.read()
            # Look for the uncalibrated method table
            # Pattern: action_name ... f1-score
            # Example: noop       0.85      0.61      0.71      8281
            for action in actions:
                # Regex to find the line starting with the action name and capturing the 3rd number (f1)
                # Assuming format: name precision recall f1 support
                match = re.search(rf"^\s*{action}\s+\S+\s+\S+\s+(\S+)", content, re.MULTILINE)
                if match:
                    f1_scores[action] = float(match.group(1))
    
    for action in actions:
        # Extract AUC
        auc_path = os.path.join(base_path, config_dir, action, seed, "test_AUC", "outputFromAUC_FILTERED.txt")
        auc_pr = None
        auc_roc = None
        if os.path.exists(auc_path):
            with open(auc_path, "r") as f:
                content = f.read()
                pr_match = re.search(r"Area Under the Curve for Precision - Recall is ([\d\.]+)", content)
                roc_match = re.search(r"Area Under the Curve for ROC is ([\d\.]+)", content)
                
                auc_pr = float(pr_match.group(1)) if pr_match else None
                auc_roc = float(roc_match.group(1)) if roc_match else None
        
        results[config_name][action] = {
            "AUC-PR": auc_pr, 
            "AUC-ROC": auc_roc,
            "F1": f1_scores.get(action)
        }

print(f"{'Config':<10} {'Action':<10} {'AUC-PR':<10} {'F1':<10} {'AUC-ROC':<10}")
print("-" * 50)
for config_name in configs:
    for action in actions:
        data = results[config_name].get(action, {})
        auc_pr = data.get("AUC-PR")
        f1 = data.get("F1")
        auc_roc = data.get("AUC-ROC")
        
        auc_pr_str = f"{auc_pr:.4f}" if auc_pr is not None else "N/A"
        f1_str = f"{f1:.4f}" if f1 is not None else "N/A"
        auc_roc_str = f"{auc_roc:.4f}" if auc_roc is not None else "N/A"
        
        print(f"{config_name:<10} {action:<10} {auc_pr_str:<10} {f1_str:<10} {auc_roc_str:<10}")
