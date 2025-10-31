import os
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
import argparse
import gzip

parser = argparse.ArgumentParser(description="Evaluate with calibrated probabilities")
parser.add_argument("--model_dir", type=str, default="", help="Model directory path")

args = parser.parse_args()

primitive_actions = ["noop","fire","up","right","left","down"]

# Load training data for calibration
train_df = pd.read_csv("train.csv")
train_df['state_id'] = train_df['frameid'].apply(lambda x: "s" + str(x).lower().replace("_",""))

# Load test data
test_df = pd.read_csv("test.csv")
test_df['state_id'] = test_df['frameid'].apply(lambda x: "s" + str(x).lower().replace("_",""))

# Load raw model predictions
state_ids = {action: [[],[]] for action in primitive_actions}
for action in primitive_actions:
    test_query_file = f"data/seaquest/all/{action}/test/query_{action}.db"
    
    with open(test_query_file, "r") as f:
        lines = f.read().splitlines()
        for line in lines:
            state_id = line.split("(")[1].split(")")[0]
            if "!" in line:
                state_ids[action][1].append(state_id)
            else:
                state_ids[action][0].append(state_id)

# Load test predictions
pred_prob = {action: [[],[]] for action in primitive_actions}
for action in primitive_actions:
    auc_file = f"data/seaquest/all/{action}/test/AUC/aucTemp.txt"
    with open(auc_file, "r") as f:
        lines = f.read().splitlines()
        for i,line in enumerate(lines):
            parts = line.split()
            if i < len(state_ids[action][0]):
                pred_prob[action][0].append(float(parts[0]))
            else:
                pred_prob[action][1].append(float(parts[0]))

# Load training predictions from train_infer directories
train_state_ids = {action: [[],[]] for action in primitive_actions}
for action in primitive_actions:
    train_query_file = f"data/seaquest/all/{action}/train/train_infer/query_{action}.db.gz"
    
    with gzip.open(train_query_file, "rt") as f:
        lines = f.read().splitlines()
        for line in lines:
            state_id = line.split("(")[1].split(")")[0]
            if "!" in line:
                train_state_ids[action][1].append(state_id)
            else:
                train_state_ids[action][0].append(state_id)

train_pred_prob = {action: [[],[]] for action in primitive_actions}
for action in primitive_actions:
    train_auc_file = f"data/seaquest/all/{action}/train/train_infer/AUC/aucTemp.txt"
    with open(train_auc_file, "r") as f:
        lines = f.read().splitlines()
        for i,line in enumerate(lines):
            parts = line.split()
            if i < len(train_state_ids[action][0]):
                train_pred_prob[action][0].append(float(parts[0]))
            else:
                train_pred_prob[action][1].append(float(parts[0]))

# Build training dataset for calibration
train_state_id_list = [state_id for action in primitive_actions for state_id in train_state_ids[action][0]] + \
                       [state_id for action in primitive_actions for state_id in train_state_ids[action][1]]

train_state_id_probs = {state_id: [] for state_id in train_state_id_list}
for action in primitive_actions:
    for i, state_id in enumerate(train_state_ids[action][0]):
        train_state_id_probs[state_id].append(train_pred_prob[action][0][i])
    for i, state_id in enumerate(train_state_ids[action][1]):
        train_state_id_probs[state_id].append(train_pred_prob[action][1][i])

# Match training state IDs with their true actions
train_df_filtered = train_df[train_df['state_id'].isin(train_state_id_list)].copy()
train_df_filtered = train_df_filtered.set_index('state_id')

X_train = []
y_train = []
for state_id in train_state_id_list:
    if state_id in train_df_filtered.index:
        X_train.append(train_state_id_probs[state_id])
        y_train.append(train_df_filtered.loc[state_id, 'action'])

X_train = np.array(X_train)
y_train = np.array(y_train)

print(f"Training calibration model on {len(X_train)} samples...")

# Fit Platt scaling (logistic regression) on the training probabilities
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Compute class weights for balanced learning
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = dict(enumerate(class_weights))

print(f"Class weights: {class_weight_dict}")

calibrator = LogisticRegression(max_iter=1000, random_state=42, class_weight=class_weight_dict)
calibrator.fit(X_train_scaled, y_train)

print(f"Calibration model trained!")

# Build test dataset
state_id_list = [state_id for action in primitive_actions for state_id in state_ids[action][0]] + \
                [state_id for action in primitive_actions for state_id in state_ids[action][1]]

state_id_action_probs = {state_id: [] for state_id in state_id_list}
for action in primitive_actions:
    for i, state_id in enumerate(state_ids[action][0]):
        state_id_action_probs[state_id].append(pred_prob[action][0][i])
    for i, state_id in enumerate(state_ids[action][1]):
        state_id_action_probs[state_id].append(pred_prob[action][1][i])

# Apply calibration to test set
X_test = []
test_state_ids_ordered = []
for state_id in test_df['state_id']:
    if state_id in state_id_action_probs:
        X_test.append(state_id_action_probs[state_id])
        test_state_ids_ordered.append(state_id)

X_test = np.array(X_test)
X_test_scaled = scaler.transform(X_test)

# Get calibrated probabilities
calibrated_probs = calibrator.predict_proba(X_test_scaled)
calibrated_predictions = np.argmax(calibrated_probs, axis=1)

# Original predictions (argmax of raw probs)
original_predictions = np.argmax(X_test, axis=1)

# Match with test dataframe
test_df_filtered = test_df[test_df['state_id'].isin(test_state_ids_ordered)].copy()
test_df_filtered = test_df_filtered.set_index('state_id')

y_true = []
y_pred_original = []
y_pred_calibrated = []

for i, state_id in enumerate(test_state_ids_ordered):
    if state_id in test_df_filtered.index:
        y_true.append(test_df_filtered.loc[state_id, 'action'])
        y_pred_original.append(original_predictions[i])
        y_pred_calibrated.append(calibrated_predictions[i])

y_true = np.array(y_true)
y_pred_original = np.array(y_pred_original)
y_pred_calibrated = np.array(y_pred_calibrated)

# ==============================================================================
# TRAINING SET EVALUATION
# ==============================================================================
print("\n" + "="*80)
print("TRAINING SET PERFORMANCE")
print("="*80)

# Evaluate on training data
train_y_true = []
train_y_pred_original = []
train_y_pred_calibrated = []

for i, state_id in enumerate(train_state_id_list):
    if state_id in train_df_filtered.index:
        train_y_true.append(train_df_filtered.loc[state_id, 'action'])
        train_probs = train_state_id_probs[state_id]
        train_y_pred_original.append(np.argmax(train_probs))
        # Use calibrator to get calibrated prediction
        train_probs_scaled = scaler.transform([train_probs])
        train_calib_pred = calibrator.predict(train_probs_scaled)[0]
        train_y_pred_calibrated.append(train_calib_pred)

train_y_true = np.array(train_y_true)
train_y_pred_original = np.array(train_y_pred_original)
train_y_pred_calibrated = np.array(train_y_pred_calibrated)

print("\nMETHOD 1: Direct argmax (non-calibrated) - TRAINING")
print("-" * 80)
print(classification_report(train_y_true, train_y_pred_original, target_names=primitive_actions))
print("\nConfusion Matrix:")
print(confusion_matrix(train_y_true, train_y_pred_original))

print("\n" + "-" * 80)
print("METHOD 2: With logistic regression on the classifiers - TRAINING")
print("-" * 80)
print(classification_report(train_y_true, train_y_pred_calibrated, target_names=primitive_actions))
print("\nConfusion Matrix:")
print(confusion_matrix(train_y_true, train_y_pred_calibrated))

# Training set differences
train_num_different = (train_y_pred_original != train_y_pred_calibrated).sum()
print(f"\nTraining: {train_num_different} / {len(train_y_true)} predictions differ ({100 * train_num_different / len(train_y_true):.2f}%)")

# ==============================================================================
# TEST SET EVALUATION
# ==============================================================================
print("\n\n" + "="*80)
print("TEST SET PERFORMANCE")
print("="*80)

print("\nMETHOD 1: Direct argmax (non-calibrated) - TEST")
print("-" * 80)
print(classification_report(y_true, y_pred_original, target_names=primitive_actions))
print("\nConfusion Matrix:")
print(confusion_matrix(y_true, y_pred_original))

print("\n" + "-" * 80)
print("METHOD 2: With logistic regression on the classifiers - TEST")
print("-" * 80)
print(classification_report(y_true, y_pred_calibrated, target_names=primitive_actions))
print("\nConfusion Matrix:")
print(confusion_matrix(y_true, y_pred_calibrated))

# Test set differences
num_different = (y_pred_original != y_pred_calibrated).sum()
print(f"\nTest: {num_different} / {len(y_true)} predictions differ ({100 * num_different / len(y_true):.2f}%)")

print("\n" + "="*80)

# Save report to model directory
if args.model_dir:
    report_file = f"{args.model_dir}/eval_report.txt"
else:
    report_file = "eval_report.txt"

with open(report_file, "w") as f:
    # Training set results
    f.write("="*80 + "\n")
    f.write("TRAINING SET PERFORMANCE\n")
    f.write("="*80 + "\n\n")
    
    f.write("METHOD 1: Direct argmax (non-calibrated) - TRAINING\n")
    f.write("-"*80 + "\n")
    f.write(classification_report(train_y_true, train_y_pred_original, target_names=primitive_actions))
    f.write("\nConfusion Matrix:\n")
    f.write(str(confusion_matrix(train_y_true, train_y_pred_original)))
    f.write("\n\n")
    
    f.write("METHOD 2: With logistic regression on the classifiers - TRAINING\n")
    f.write("-"*80 + "\n")
    f.write(classification_report(train_y_true, train_y_pred_calibrated, target_names=primitive_actions))
    f.write("\nConfusion Matrix:\n")
    f.write(str(confusion_matrix(train_y_true, train_y_pred_calibrated)))
    f.write(f"\n\nTraining: {train_num_different} / {len(train_y_true)} predictions differ ({100 * train_num_different / len(train_y_true):.2f}%)\n")
    
    # Test set results
    f.write("\n\n" + "="*80 + "\n")
    f.write("TEST SET PERFORMANCE\n")
    f.write("="*80 + "\n\n")
    
    f.write("METHOD 1: Direct argmax (non-calibrated) - TEST\n")
    f.write("-"*80 + "\n")
    f.write(classification_report(y_true, y_pred_original, target_names=primitive_actions))
    f.write("\nConfusion Matrix:\n")
    f.write(str(confusion_matrix(y_true, y_pred_original)))
    f.write("\n\n")
    
    f.write("METHOD 2: With logistic regression on the classifiers - TEST\n")
    f.write("-"*80 + "\n")
    f.write(classification_report(y_true, y_pred_calibrated, target_names=primitive_actions))
    f.write("\nConfusion Matrix:\n")
    f.write(str(confusion_matrix(y_true, y_pred_calibrated)))
    f.write(f"\n\nTest: {num_different} / {len(y_true)} predictions differ ({100 * num_different / len(y_true):.2f}%)\n")

print(f"\n✅ Evaluation report saved to: {report_file}")
