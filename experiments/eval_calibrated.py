import os
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
import argparse
import gzip

parser = argparse.ArgumentParser(description="Evaluate with calibrated probabilities")
parser.add_argument("--model_dir", type=str, default="", help="Model directory path (base path containing action subdirectories)")
parser.add_argument("--data_base", type=str, default="data/seaquest/all", help="Base directory for data")
parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456, 789, 1729], help="Seeds to evaluate")
parser.add_argument("--negpos", type=float, default=2.0, help="NegPos ratio used for testing (default: 2.0)")
parser.add_argument("--output_file", type=str, default="eval_calibrated_report.txt", help="Output report filename")

args = parser.parse_args()

primitive_actions = ["noop","fire","up","right","left","down"]
seeds = args.seeds

print(f"Evaluating seeds: {seeds}")
print(f"Model Base Directory: {args.model_dir}")
print(f"Data Base Directory: {args.data_base}")

# Load training data for calibration
# Try to find train.csv
train_csv_path = "train.csv"
if not os.path.exists(train_csv_path):
    train_csv_path = os.path.join(args.data_base, "train.csv")
    if not os.path.exists(train_csv_path):
        # Fallback to root
        train_csv_path = os.path.join(os.path.dirname(args.data_base), "train.csv")
        if not os.path.exists(train_csv_path):
             # Fallback to project root (assuming script is run from project root)
             train_csv_path = "train.csv"

if os.path.exists(train_csv_path):
    print(f"Loading training data from: {train_csv_path}")
    train_df = pd.read_csv(train_csv_path)
    train_df['state_id'] = train_df['frameid'].apply(lambda x: "s" + str(x).lower().replace("_",""))
else:
    print("Error: train.csv not found!")
    exit(1)

# Load test data
test_csv_path = "test.csv"
if not os.path.exists(test_csv_path):
    test_csv_path = os.path.join(args.data_base, "test.csv")
    if not os.path.exists(test_csv_path):
         test_csv_path = os.path.join(os.path.dirname(args.data_base), "test.csv")

if os.path.exists(test_csv_path):
    print(f"Loading test data from: {test_csv_path}")
    test_df = pd.read_csv(test_csv_path)
    test_df['state_id'] = test_df['frameid'].apply(lambda x: "s" + str(x).lower().replace("_",""))
else:
    print("Error: test.csv not found!")
    exit(1)

# Helper function to load state IDs from query file
def load_state_ids(query_file):
    ids = [[], []] # [positive_ids, negative_ids]
    if query_file.endswith(".gz"):
        opener = gzip.open
        mode = "rt"
    else:
        opener = open
        mode = "r"
        
    try:
        with opener(query_file, mode) as f:
            lines = f.read().splitlines()
            for line in lines:
                try:
                    # Parse "action(state_id, ...)" or "action(state_id)"
                    content = line.split("(")[1]
                    if "," in content:
                        state_id = content.split(",")[0]
                    else:
                        state_id = content.split(")")[0]
                    
                    if "!" in line:
                        ids[1].append(state_id)
                    else:
                        ids[0].append(state_id)
                except IndexError:
                    continue
    except FileNotFoundError:
        print(f"Warning: Query file not found: {query_file}")
        return [[], []]
    return ids

# Helper function to load probabilities from AUC file
def load_probs(auc_file, num_pos):
    probs = [[], []] # [positive_probs, negative_probs]
    try:
        with open(auc_file, "r") as f:
            lines = f.read().splitlines()
            for i, line in enumerate(lines):
                parts = line.split()
                if len(parts) > 0:
                    prob = float(parts[0])
                    if i < num_pos:
                        probs[0].append(prob)
                    else:
                        probs[1].append(prob)
    except FileNotFoundError:
        print(f"Warning: AUC file not found: {auc_file}")
        return [[], []]
    return probs

# ==============================================================================
# LOAD AND COMBINE PREDICTIONS
# ==============================================================================

# Structure to store combined probabilities: {state_id: [prob_seed1, prob_seed2, ...]}
# We will average them later.
# We need to do this for both TRAIN (for calibration) and TEST.

# 1. TRAIN PREDICTIONS
print("\nLoading Training Predictions...")
train_combined_probs = {action: {} for action in primitive_actions} # action -> state_id -> list of probs
train_state_ids_map = {action: [[], []] for action in primitive_actions} # Store IDs to ensure order

for action in primitive_actions:
    # Load IDs from the first seed (assuming all seeds use same data/query file)
    # Actually, query file is in train_infer directory.
    # We updated run_all.sh to put train_infer in MODEL_DIR/train_infer.
    # But we can also load from the original data directory if it's the same.
    # However, the order in aucTemp.txt corresponds to the query file used during inference.
    # Since we copied the query file to MODEL_DIR/train_infer, we should use that one.
    
    first_seed = seeds[0]
    # Check both new location (MODEL_DIR/train_infer) and old location (DATA_BASE/action/train/train_infer)
    # The script run_all.sh now puts it in MODEL_DIR/train_infer
    
    query_file = f"{args.model_dir}/{action}/seed_{first_seed}/train_infer/query_action.db"
    if not os.path.exists(query_file):
        query_file = f"{args.model_dir}/{action}/seed_{first_seed}/train_infer/query_action.db.gz"
        if not os.path.exists(query_file):
            # Fallback to old location if running on old results
            query_file = f"{args.data_base}/{action}/train/train_infer/query_{action}.db.gz"
            if not os.path.exists(query_file):
                 query_file = f"{args.data_base}/{action}/train/train_infer/query_{action}.db"
    
    print(f"  Action {action}: Loading IDs from {query_file}")
    train_state_ids_map[action] = load_state_ids(query_file)
    
    num_pos = len(train_state_ids_map[action][0])
    num_neg = len(train_state_ids_map[action][1])
    print(f"    Found {num_pos} pos and {num_neg} neg examples.")

    # Initialize lists for each state_id
    for state_id in train_state_ids_map[action][0] + train_state_ids_map[action][1]:
        train_combined_probs[action][state_id] = []

    # Load probs for each seed
    for seed in seeds:
        # Try new location first: MODEL_DIR/train_infer/AUC/aucTemp.txt
        auc_file = f"{args.model_dir}/{action}/seed_{seed}/train_infer/AUC/aucTemp.txt"
        if not os.path.exists(auc_file):
             # Fallback
             auc_file = f"{args.data_base}/{action}/train/train_infer/AUC/aucTemp.txt"
        
        if os.path.exists(auc_file):
            probs = load_probs(auc_file, num_pos)
            # Flatten and map to state_ids
            all_probs = probs[0] + probs[1]
            all_ids = train_state_ids_map[action][0] + train_state_ids_map[action][1]
            
            if len(all_probs) != len(all_ids):
                print(f"    WARNING: Seed {seed} has {len(all_probs)} probs but expected {len(all_ids)} IDs. Skipping.")
                continue
                
            for i, state_id in enumerate(all_ids):
                train_combined_probs[action][state_id].append(all_probs[i])
        else:
             print(f"    WARNING: AUC file for seed {seed} not found at {auc_file}")

# 2. TEST PREDICTIONS
print("\nLoading Test Predictions...")
test_combined_probs = {action: {} for action in primitive_actions}
test_state_ids_map = {action: [[], []] for action in primitive_actions}

for action in primitive_actions:
    # Load IDs
    # Test query file is usually in data/seaquest/all/{action}/test/query_action.db
    query_file = f"{args.data_base}/{action}/test/query_action.db"
    
    print(f"  Action {action}: Loading IDs from {query_file}")
    test_state_ids_map[action] = load_state_ids(query_file)
    
    num_pos = len(test_state_ids_map[action][0])
    num_neg = len(test_state_ids_map[action][1])
    print(f"    Found {num_pos} pos and {num_neg} neg examples.")
    
    for state_id in test_state_ids_map[action][0] + test_state_ids_map[action][1]:
        test_combined_probs[action][state_id] = []
        
    # Load probs for each seed
    for seed in seeds:
        # New location: MODEL_DIR/test_AUC/aucTemp.txt
        auc_file = f"{args.model_dir}/{action}/seed_{seed}/test_AUC/aucTemp.txt"
        if not os.path.exists(auc_file):
             # Fallback to old location (only works for last seed if overwritten)
             auc_file = f"{args.data_base}/{action}/test/AUC/aucTemp.txt"
        
        if os.path.exists(auc_file):
            probs = load_probs(auc_file, num_pos)
            all_probs = probs[0] + probs[1]
            all_ids = test_state_ids_map[action][0] + test_state_ids_map[action][1]
            
            if len(all_probs) != len(all_ids):
                print(f"    WARNING: Seed {seed} has {len(all_probs)} probs but expected {len(all_ids)} IDs. Skipping.")
                continue

            for i, state_id in enumerate(all_ids):
                test_combined_probs[action][state_id].append(all_probs[i])
        else:
             print(f"    WARNING: AUC file for seed {seed} not found at {auc_file}")

# ==============================================================================
# AVERAGE PREDICTIONS
# ==============================================================================
print("\nAveraging Predictions...")

# Train
train_avg_probs = {action: {} for action in primitive_actions}
for action in primitive_actions:
    for state_id, probs_list in train_combined_probs[action].items():
        if probs_list:
            train_avg_probs[action][state_id] = np.mean(probs_list)
        else:
            train_avg_probs[action][state_id] = 0.0 # Default if no probs found

# Test
test_avg_probs = {action: {} for action in primitive_actions}
for action in primitive_actions:
    for state_id, probs_list in test_combined_probs[action].items():
        if probs_list:
            test_avg_probs[action][state_id] = np.mean(probs_list)
        else:
            test_avg_probs[action][state_id] = 0.0

# ==============================================================================
# CALIBRATION AND EVALUATION
# ==============================================================================

# Build training dataset for calibration
train_state_id_list = []
for action in primitive_actions:
    train_state_id_list.extend(train_state_ids_map[action][0])
    train_state_id_list.extend(train_state_ids_map[action][1])

# Filter duplicates if any (though state_ids should be unique per frame, but here we iterate actions)
# Actually, a state_id belongs to one frame, and one frame has one action.
# But we have 6 binary classifiers.
# For calibration, we want to map (Prob_Action1, Prob_Action2, ...) -> True_Action
# So for each state_id, we need a vector of 6 probabilities.

print("\nPreparing Calibration Dataset...")

# Get unique state IDs across all actions in training
all_train_state_ids = set()
for action in primitive_actions:
    all_train_state_ids.update(train_avg_probs[action].keys())

# Filter to those present in train_df
train_df_filtered = train_df[train_df['state_id'].isin(all_train_state_ids)].copy()
train_df_filtered = train_df_filtered.set_index('state_id')

X_train = []
y_train = []
valid_train_state_ids = []

for state_id in train_df_filtered.index:
    # Construct feature vector: [prob_noop, prob_fire, ...]
    probs_vector = []
    for action in primitive_actions:
        probs_vector.append(train_avg_probs[action].get(state_id, 0.0))
    
    X_train.append(probs_vector)
    y_train.append(train_df_filtered.loc[state_id, 'action'])
    valid_train_state_ids.append(state_id)

X_train = np.array(X_train)
y_train = np.array([primitive_actions[y] for y in y_train])

print(f"Training calibration model on {len(X_train)} samples...")

# Fit Platt scaling (logistic regression)
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Compute class weights
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = dict(enumerate(class_weights))
# Map class indices to weights correctly
unique_classes = np.unique(y_train)
class_weight_dict = {cls: weight for cls, weight in zip(unique_classes, class_weights)}

print(f"Class weights: {class_weight_dict}")

calibrator = LogisticRegression(max_iter=1000, random_state=42, class_weight=class_weight_dict)
calibrator.fit(X_train_scaled, y_train)

print(f"Calibration model trained!")

# ==============================================================================
# TEST EVALUATION
# ==============================================================================
print("\nPreparing Test Dataset...")

# Get unique state IDs across all actions in test
all_test_state_ids = set()
for action in primitive_actions:
    all_test_state_ids.update(test_avg_probs[action].keys())

test_df_filtered = test_df[test_df['state_id'].isin(all_test_state_ids)].copy()
test_df_filtered = test_df_filtered.set_index('state_id')

X_test = []
y_true = []
valid_test_state_ids = []

for state_id in test_df_filtered.index:
    probs_vector = []
    for action in primitive_actions:
        probs_vector.append(test_avg_probs[action].get(state_id, 0.0))
    
    X_test.append(probs_vector)
    y_true.append(test_df_filtered.loc[state_id, 'action'])
    valid_test_state_ids.append(state_id)

X_test = np.array(X_test)
y_true = np.array([primitive_actions[y] for y in y_true])

X_test_scaled = scaler.transform(X_test)

# Get calibrated predictions
calibrated_predictions = calibrator.predict(X_test_scaled)

# Get original predictions (argmax of raw probs)
# We need to map index to action name
action_to_idx = {action: i for i, action in enumerate(primitive_actions)}
idx_to_action = {i: action for i, action in enumerate(primitive_actions)}

original_predictions_idx = np.argmax(X_test, axis=1)
original_predictions = [idx_to_action[idx] for idx in original_predictions_idx]

# ==============================================================================
# REPORTING
# ==============================================================================

def print_report(y_true, y_pred, title):
    report = f"\n{title}\n" + "-"*80 + "\n"
    report += classification_report(y_true, y_pred, target_names=primitive_actions, labels=primitive_actions)
    report += "\nConfusion Matrix:\n"
    report += str(confusion_matrix(y_true, y_pred, labels=primitive_actions))
    return report

report_content = ""
report_content += "="*80 + f"\nTEST SET PERFORMANCE (Seed {seeds})\n" + "="*80 + "\n\n"

# Method 1
report_1 = print_report(y_true, original_predictions, "METHOD 1: Direct argmax (non-calibrated)")
print(report_1)
report_content += report_1 + "\n\n"

# Method 2
report_2 = print_report(y_true, calibrated_predictions, "METHOD 2: With logistic regression calibration")
print(report_2)
report_content += report_2 + "\n\n"

# Differences
num_different = np.sum(np.array(original_predictions) != np.array(calibrated_predictions))
diff_msg = f"Predictions differ: {num_different} / {len(y_true)} ({100 * num_different / len(y_true):.2f}%)"
print(diff_msg)
report_content += diff_msg + "\n"

# Save report
if args.model_dir:
    report_file = os.path.join(args.model_dir, args.output_file)
else:
    report_file = args.output_file

with open(report_file, "w") as f:
    f.write(report_content)

print(f"\n✅ Evaluation report saved to: {report_file}")
