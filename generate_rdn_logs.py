import os
import re
import glob

# Configuration
RDN_MODELS_DIR = "trained_models/seaquest/all"
ACTIONS = ["fire", "up", "down", "left", "right", "noop"]
SEEDS = [42] # Add more seeds if needed

def extract_metrics_from_auc_output(auc_file):
    metrics = {}
    if os.path.exists(auc_file):
        with open(auc_file, 'r') as f:
            content = f.read()
            # Parse AUC PR
            match_pr = re.search(r"Area Under the Curve for Precision - Recall is ([0-9.]+)", content)
            if match_pr:
                metrics['AUC PR'] = float(match_pr.group(1))
            
            # Parse AUC ROC
            match_roc = re.search(r"Area Under the Curve for ROC is ([0-9.]+)", content)
            if match_roc:
                metrics['AUC ROC'] = float(match_roc.group(1))
    return metrics

def extract_metrics_from_log(log_file):
    metrics = {}
    if os.path.exists(log_file):
        with open(log_file, 'r') as f:
            content = f.read()
            # Parse F1, Precision, Recall if available in the log
            # The log format seen earlier:
            # %   AUC ROC   = 0.820772
            # %   AUC PR    = 0.660964
            # %   Precision = 0.730483 at threshold = 0.178
            # %   Recall    = 0.604615
            # %   F1        = 0.661616
            
            match_f1 = re.search(r"%\s+F1\s+=\s+([0-9.]+)", content)
            if match_f1:
                metrics['F1'] = float(match_f1.group(1))
                
            match_prec = re.search(r"%\s+Precision\s+=\s+([0-9.]+)", content)
            if match_prec:
                metrics['Precision'] = float(match_prec.group(1))
                
            match_rec = re.search(r"%\s+Recall\s+=\s+([0-9.]+)", content)
            if match_rec:
                metrics['Recall'] = float(match_rec.group(1))
                
            # Also try to get AUCs from here if not in AUC file
            if 'AUC PR' not in metrics:
                match_pr = re.search(r"%\s+AUC PR\s+=\s+([0-9.]+)", content)
                if match_pr:
                    metrics['AUC PR'] = float(match_pr.group(1))
                    
            if 'AUC ROC' not in metrics:
                match_roc = re.search(r"%\s+AUC ROC\s+=\s+([0-9.]+)", content)
                if match_roc:
                    metrics['AUC ROC'] = float(match_roc.group(1))
                    
    return metrics

def create_standard_log(model_dir, action, seed, metrics):
    log_file = os.path.join(model_dir, "test_infer.log")
    with open(log_file, "w") as f:
        f.write(f"Results for {action} (Seed {seed}, Ratio 2.0):\n")
        if 'AUC PR' in metrics:
            f.write(f"AUC PR:    {metrics['AUC PR']:.4f}\n")
        if 'AUC ROC' in metrics:
            f.write(f"AUC ROC:   {metrics['AUC ROC']:.4f}\n")
        if 'F1' in metrics:
            f.write(f"F1:        {metrics['F1']:.4f}\n")
        if 'Precision' in metrics:
            f.write(f"Precision: {metrics['Precision']:.4f}\n")
        if 'Recall' in metrics:
            f.write(f"Recall:    {metrics['Recall']:.4f}\n")
            
    print(f"Created {log_file}")

def process_rdn_models():
    # Find all RDN model directories
    # Pattern: negpos_2_trees_1_depth_{1,2,3}
    # Also check for other variations if needed
    
    patterns = [
        "negpos_2_trees_1_depth_1",
        "negpos_2_trees_1_depth_2",
        "negpos_2_trees_1_depth_3"
    ]
    
    for pattern in patterns:
        base_path = os.path.join(RDN_MODELS_DIR, pattern)
        if not os.path.exists(base_path):
            print(f"Skipping {base_path} (not found)")
            continue
            
        for action in ACTIONS:
            for seed in SEEDS:
                model_dir = os.path.join(base_path, action, f"seed_{seed}")
                if not os.path.exists(model_dir):
                    continue
                    
                print(f"Processing {model_dir}...")
                
                metrics = {}
                
                # 1. Try outputFromAUC.txt
                auc_file = os.path.join(model_dir, "test_AUC", "outputFromAUC.txt")
                metrics.update(extract_metrics_from_auc_output(auc_file))
                
                # 2. Try test_infer_seed_{seed}.log
                log_file = os.path.join(model_dir, f"test_infer_seed_{seed}.log")
                metrics.update(extract_metrics_from_log(log_file))
                
                if metrics:
                    create_standard_log(model_dir, action, seed, metrics)
                else:
                    print(f"No metrics found for {model_dir}")

if __name__ == "__main__":
    process_rdn_models()
