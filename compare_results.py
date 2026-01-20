import json
import os
import matplotlib.pyplot as plt
import numpy as np
import re

# Configuration
RESULTS_JSON = "results_ratio_2.0.json"
RDN_LOG_DIR = "rdn_models/seaquest/all/negpos_2_mlp_64_32_bc"
SEED = 42
ACTIONS = [0, 1, 2, 3, 4, 5]
ACTION_NAMES = {
    0: "noop",
    1: "fire",
    2: "up",
    3: "right",
    4: "left",
    5: "down"
}

def load_pixel_results():
    with open(RESULTS_JSON, 'r') as f:
        data = json.load(f)
    return data

def load_rdn_results():
    results = {
        'action': [],
        'rdn_f1': [],
        'rdn_auc': []
    }
    
    for action_idx in ACTIONS:
        action_name = ACTION_NAMES[action_idx]
        log_file = os.path.join(RDN_LOG_DIR, action_name, f"seed_{SEED}", "test_infer.log")
        
        f1 = 0.0
        auc = 0.0
        
        if os.path.exists(log_file):
            with open(log_file, 'r') as f:
                content = f.read()
                # Parse F1 and AUC
                f1_match = re.search(r"F1:\s+([0-9.]+)", content)
                auc_match = re.search(r"AUC PR:\s+([0-9.]+)", content)
                
                if f1_match:
                    f1 = float(f1_match.group(1))
                if auc_match:
                    auc = float(auc_match.group(1))
        else:
            print(f"Warning: Log file not found for {action_name}: {log_file}")
            
        results['action'].append(action_idx)
        results['rdn_f1'].append(f1)
        results['rdn_auc'].append(auc)
        
    return results

def main():
    pixel_data = load_pixel_results()
    rdn_data = load_rdn_results()
    
    # Align data
    actions = pixel_data['action']
    rgb_f1 = pixel_data['rgb_f1']
    gaze_f1 = pixel_data['gaze_f1']
    rgb_auc = pixel_data['rgb_auc']
    gaze_auc = pixel_data['gaze_auc']
    
    rdn_f1 = []
    rdn_auc = []
    
    for action in actions:
        # Find corresponding RDN result
        idx = rdn_data['action'].index(action)
        rdn_f1.append(rdn_data['rdn_f1'][idx])
        rdn_auc.append(rdn_data['rdn_auc'][idx])
        
    # Plotting
    x = np.arange(len(actions))
    width = 0.25
    
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    
    # F1 Plot
    axes[0].bar(x - width, rgb_f1, width, label='RGB (CNN)', color='blue')
    axes[0].bar(x, gaze_f1, width, label='RGB + Gaze (CNN)', color='green')
    axes[0].bar(x + width, rdn_f1, width, label='Symbolic (MLP)', color='purple')
    axes[0].set_ylabel('F1 Score')
    axes[0].set_title('F1 Score per Action (Ratio 2.0)')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([ACTION_NAMES[a] for a in actions])
    axes[0].legend()
    axes[0].grid(axis='y', linestyle='--', alpha=0.7)
    
    # AUC-PR Plot
    axes[1].bar(x - width, rgb_auc, width, label='RGB (CNN)', color='red')
    axes[1].bar(x, gaze_auc, width, label='RGB + Gaze (CNN)', color='orange')
    axes[1].bar(x + width, rdn_auc, width, label='Symbolic (MLP)', color='brown')
    axes[1].set_ylabel('AUC-PR')
    axes[1].set_title('AUC-PR per Action (Ratio 2.0)')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([ACTION_NAMES[a] for a in actions])
    axes[1].legend()
    axes[1].grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('model_comparison_all.png')
    print("\nSaved plot to model_comparison_all.png")

if __name__ == "__main__":
    main()
