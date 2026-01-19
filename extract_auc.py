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
    for action in actions:
        file_path = os.path.join(base_path, config_dir, action, seed, "test_AUC", "outputFromAUC_FILTERED.txt")
        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                content = f.read()
                pr_match = re.search(r"Area Under the Curve for Precision - Recall is ([\d\.]+)", content)
                roc_match = re.search(r"Area Under the Curve for ROC is ([\d\.]+)", content)
                
                auc_pr = float(pr_match.group(1)) if pr_match else None
                auc_roc = float(roc_match.group(1)) if roc_match else None
                
                results[config_name][action] = {"AUC-PR": auc_pr, "AUC-ROC": auc_roc}
        else:
            results[config_name][action] = {"AUC-PR": None, "AUC-ROC": None}

print(f"{'Config':<10} {'Action':<10} {'AUC-PR':<10} {'AUC-ROC':<10}")
print("-" * 40)
for config_name in configs:
    for action in actions:
        data = results[config_name].get(action, {})
        auc_pr = data.get("AUC-PR")
        auc_roc = data.get("AUC-ROC")
        print(f"{config_name:<10} {action:<10} {auc_pr if auc_pr else 'N/A':<10.4f} {auc_roc if auc_roc else 'N/A':<10.4f}")
