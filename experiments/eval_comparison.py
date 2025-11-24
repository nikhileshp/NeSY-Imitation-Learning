import os
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.base import BaseEstimator, ClassifierMixin
import argparse

parser = argparse.ArgumentParser(description="Evaluate with multiple methods")
parser.add_argument("--model_dir", type=str, default="", help="Model directory path")

args = parser.parse_args()

primitive_actions = ["noop","fire","up","right","left","down"]

# ============================================================================
# CUSTOM WEIGHTED CLASSIFIER: Learns linear weights + argmax
# ============================================================================

class WeightedClassifier(BaseEstimator, ClassifierMixin):
    """Learns linear weights over classifiers and uses argmax for prediction."""
    
    def __init__(self, learning_rate=0.01, n_iterations=1000, random_state=42):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.random_state = random_state
        self.weights_ = None
        
    def fit(self, X, y):
        np.random.seed(self.random_state)
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))
        
        # Initialize weights
        self.weights_ = np.random.randn(n_features, n_classes) * 0.01
        self.bias_ = np.zeros(n_classes)
        
        # Gradient descent
        for iteration in range(self.n_iterations):
            # Forward pass: weighted sum
            scores = X @ self.weights_ + self.bias_
            
            # Softmax for training (to get gradients)
            exp_scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))
            probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
            
            # Cross-entropy loss gradient
            y_one_hot = np.zeros((n_samples, n_classes))
            y_one_hot[np.arange(n_samples), y] = 1
            
            grad_scores = probs - y_one_hot
            
            # Update weights
            grad_weights = X.T @ grad_scores / n_samples
            grad_bias = np.mean(grad_scores, axis=0)
            
            self.weights_ -= self.learning_rate * grad_weights
            self.bias_ -= self.learning_rate * grad_bias
            
        return self
    
    def predict(self, X):
        # Weighted sum then argmax
        scores = X @ self.weights_ + self.bias_
        return np.argmax(scores, axis=1)
    
    def predict_proba(self, X):
        scores = X @ self.weights_ + self.bias_
        exp_scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))
        return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

# Load training data
train_df = pd.read_csv("train.csv")
train_df['state_id'] = train_df['frameid'].apply(lambda x: "s" + str(x).lower().replace("_",""))

# Load test data
test_df = pd.read_csv("test.csv")
test_df['state_id'] = test_df['frameid'].apply(lambda x: "s" + str(x).lower().replace("_",""))

# ============================================================================
# LOAD RAW MODEL PREDICTIONS
# ============================================================================

# Load test predictions
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

# Load training predictions
train_state_ids = {action: [[],[]] for action in primitive_actions}
for action in primitive_actions:
    train_query_file = f"data/seaquest/all/{action}/train/train_infer/query_{action}.db"
    
    with open(train_query_file, "r") as f:
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

# ============================================================================
# BUILD TRAINING DATASET
# ============================================================================

train_state_id_list = [state_id for action in primitive_actions for state_id in train_state_ids[action][0]] + \
                       [state_id for action in primitive_actions for state_id in train_state_ids[action][1]]

train_state_id_probs = {state_id: [] for state_id in train_state_id_list}
for action in primitive_actions:
    for i, state_id in enumerate(train_state_ids[action][0]):
        train_state_id_probs[state_id].append(train_pred_prob[action][0][i])
    for i, state_id in enumerate(train_state_ids[action][1]):
        train_state_id_probs[state_id].append(train_pred_prob[action][1][i])

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

print(f"Training set size: {len(X_train)} samples")

# ============================================================================
# BUILD TEST DATASET
# ============================================================================

state_id_list = [state_id for action in primitive_actions for state_id in state_ids[action][0]] + \
                [state_id for action in primitive_actions for state_id in state_ids[action][1]]

state_id_action_probs = {state_id: [] for state_id in state_id_list}
for action in primitive_actions:
    for i, state_id in enumerate(state_ids[action][0]):
        state_id_action_probs[state_id].append(pred_prob[action][0][i])
    for i, state_id in enumerate(state_ids[action][1]):
        state_id_action_probs[state_id].append(pred_prob[action][1][i])

X_test = []
test_state_ids_ordered = []
for state_id in test_df['state_id']:
    if state_id in state_id_action_probs:
        X_test.append(state_id_action_probs[state_id])
        test_state_ids_ordered.append(state_id)

X_test = np.array(X_test)

test_df_filtered = test_df[test_df['state_id'].isin(test_state_ids_ordered)].copy()
test_df_filtered = test_df_filtered.set_index('state_id')

y_true = []
for state_id in test_state_ids_ordered:
    if state_id in test_df_filtered.index:
        y_true.append(test_df_filtered.loc[state_id, 'action'])

y_true = np.array(y_true)

print(f"Test set size: {len(X_test)} samples")

# ============================================================================
# STANDARDIZE FEATURES
# ============================================================================

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ============================================================================
# METHOD 1: Direct argmax (non-calibrated)
# ============================================================================

y_pred_train_argmax = np.argmax(X_train, axis=1)
y_pred_test_argmax = np.argmax(X_test, axis=1)

# ============================================================================
# METHOD 2: Logistic Regression WITHOUT class weights
# ============================================================================

print("\nTraining Logistic Regression WITHOUT class weights...")
lr_no_weights = LogisticRegression(max_iter=1000, random_state=42)
lr_no_weights.fit(X_train_scaled, y_train)

y_pred_train_lr_no_weights = lr_no_weights.predict(X_train_scaled)
y_pred_test_lr_no_weights = lr_no_weights.predict(X_test_scaled)

# ============================================================================
# METHOD 3: Logistic Regression WITH class weights (balanced)
# ============================================================================

print("Training Logistic Regression WITH class weights...")
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = dict(enumerate(class_weights))

lr_with_weights = LogisticRegression(max_iter=1000, random_state=42, class_weight=class_weight_dict)
lr_with_weights.fit(X_train_scaled, y_train)

y_pred_train_lr_with_weights = lr_with_weights.predict(X_train_scaled)
y_pred_test_lr_with_weights = lr_with_weights.predict(X_test_scaled)

# ============================================================================
# METHOD 4: Single Layer Perceptron (MLP with one hidden layer)
# ============================================================================

print("Training Single Layer Perceptron (10 hidden neurons)...")
# Use a small hidden layer that learns weights over classifiers
mlp = MLPClassifier(hidden_layer_sizes=(10,), max_iter=1000, random_state=42, 
                    early_stopping=True, validation_fraction=0.1)
mlp.fit(X_train_scaled, y_train)

y_pred_train_mlp = mlp.predict(X_train_scaled)
y_pred_test_mlp = mlp.predict(X_test_scaled)

# ============================================================================
# METHOD 5: Weighted Classifier (learns linear weights + argmax)
# ============================================================================

print("Training Weighted Classifier (linear weights + argmax)...")
weighted_clf = WeightedClassifier(learning_rate=0.1, n_iterations=2000, random_state=42)
weighted_clf.fit(X_train_scaled, y_train)

y_pred_train_weighted = weighted_clf.predict(X_train_scaled)
y_pred_test_weighted = weighted_clf.predict(X_test_scaled)

# ============================================================================
# CALCULATE PREDICTION DIFFERENCES
# ============================================================================

train_diff_lr_no_weights = (y_pred_train_argmax != y_pred_train_lr_no_weights).sum()
train_diff_lr_with_weights = (y_pred_train_argmax != y_pred_train_lr_with_weights).sum()
train_diff_mlp = (y_pred_train_argmax != y_pred_train_mlp).sum()
train_diff_weighted = (y_pred_train_argmax != y_pred_train_weighted).sum()

test_diff_lr_no_weights = (y_pred_test_argmax != y_pred_test_lr_no_weights).sum()
test_diff_lr_with_weights = (y_pred_test_argmax != y_pred_test_lr_with_weights).sum()
test_diff_mlp = (y_pred_test_argmax != y_pred_test_mlp).sum()
test_diff_weighted = (y_pred_test_argmax != y_pred_test_weighted).sum()

print("\n" + "="*80)
print("PREDICTION DIFFERENCES FROM ARGMAX BASELINE")
print("="*80)
print(f"\nTraining Set:")
print(f"  LR without weights:  {train_diff_lr_no_weights}/{len(y_train)} ({100*train_diff_lr_no_weights/len(y_train):.2f}%)")
print(f"  LR with weights:     {train_diff_lr_with_weights}/{len(y_train)} ({100*train_diff_lr_with_weights/len(y_train):.2f}%)")
print(f"  MLP (10 hidden):     {train_diff_mlp}/{len(y_train)} ({100*train_diff_mlp/len(y_train):.2f}%)")
print(f"  Weighted + argmax:   {train_diff_weighted}/{len(y_train)} ({100*train_diff_weighted/len(y_train):.2f}%)")

print(f"\nTest Set:")
print(f"  LR without weights:  {test_diff_lr_no_weights}/{len(y_true)} ({100*test_diff_lr_no_weights/len(y_true):.2f}%)")
print(f"  LR with weights:     {test_diff_lr_with_weights}/{len(y_true)} ({100*test_diff_lr_with_weights/len(y_true):.2f}%)")
print(f"  MLP (10 hidden):     {test_diff_mlp}/{len(y_true)} ({100*test_diff_mlp/len(y_true):.2f}%)")
print(f"  Weighted + argmax:   {test_diff_weighted}/{len(y_true)} ({100*test_diff_weighted/len(y_true):.2f}%)")

# ============================================================================
# DISPLAY LEARNED WEIGHTS ON CLASSIFIERS
# ============================================================================

print("\n" + "="*80)
print("LEARNED WEIGHTS ON CLASSIFIERS")
print("="*80)

print("\nLogistic Regression WITHOUT class weights - Coefficients:")
print("  [noop, fire, up, right, left, down]")
for i, action in enumerate(primitive_actions):
    print(f"  {action:6s}: {lr_no_weights.coef_[i]}")
print(f"\nIntercepts: {lr_no_weights.intercept_}")

print("\n" + "-"*80)
print("\nLogistic Regression WITH class weights - Coefficients:")
print("  [noop, fire, up, right, left, down]")
for i, action in enumerate(primitive_actions):
    print(f"  {action:6s}: {lr_with_weights.coef_[i]}")
print(f"\nIntercepts: {lr_with_weights.intercept_}")

print("\n" + "-"*80)
print("\nSingle Layer Perceptron (MLP with 10 hidden neurons):")
print(f"  Input layer -> Hidden layer weights shape: {mlp.coefs_[0].shape}")
print(f"  Hidden layer -> Output layer weights shape: {mlp.coefs_[1].shape}")
print(f"\n  Input -> Hidden weights (first 5 rows):")
for i in range(min(6, mlp.coefs_[0].shape[0])):
    print(f"    Classifier {primitive_actions[i]:6s}: {mlp.coefs_[0][i][:5]}...")
print(f"\n  Hidden -> Output weights (transposed, shows contribution to each action):")
for i in range(mlp.coefs_[1].shape[1]):
    print(f"    To {primitive_actions[i]:6s}: {mlp.coefs_[1][:, i]}")

print("\n" + "-"*80)
print("\nWeighted Classifier (linear weights + argmax):")
print(f"  Weights shape: {weighted_clf.weights_.shape}")
print(f"\n  Learned weights for each (classifier -> action):")
print(f"  {'':12s} " + " ".join([f"{a:>8s}" for a in primitive_actions]))
for i, action in enumerate(primitive_actions):
    weights_str = " ".join([f"{weighted_clf.weights_[i, j]:8.4f}" for j in range(len(primitive_actions))])
    print(f"  {action:12s} {weights_str}")
print(f"\n  Bias terms: {weighted_clf.bias_}")

# ============================================================================
# PRINT AND SAVE RESULTS
# ============================================================================

def print_results(method_name, y_true_train, y_pred_train, y_true_test, y_pred_test):
    print("\n" + "="*80)
    print(f"{method_name}")
    print("="*80)
    
    print("\n--- TRAINING SET ---")
    print(classification_report(y_true_train, y_pred_train, target_names=primitive_actions))
    print("Confusion Matrix:")
    print(confusion_matrix(y_true_train, y_pred_train))
    
    print("\n--- TEST SET ---")
    print(classification_report(y_true_test, y_pred_test, target_names=primitive_actions))
    print("Confusion Matrix:")
    print(confusion_matrix(y_true_test, y_pred_test))
    
    return {
        'train_report': classification_report(y_true_train, y_pred_train, target_names=primitive_actions),
        'train_cm': confusion_matrix(y_true_train, y_pred_train),
        'test_report': classification_report(y_true_test, y_pred_test, target_names=primitive_actions),
        'test_cm': confusion_matrix(y_true_test, y_pred_test)
    }

results = {}
results['argmax'] = print_results(
    "METHOD 1: Direct argmax (non-calibrated)",
    y_train, y_pred_train_argmax,
    y_true, y_pred_test_argmax
)

results['lr_no_weights'] = print_results(
    "METHOD 2: Logistic Regression WITHOUT class weights",
    y_train, y_pred_train_lr_no_weights,
    y_true, y_pred_test_lr_no_weights
)

results['lr_with_weights'] = print_results(
    "METHOD 3: Logistic Regression WITH class weights (balanced)",
    y_train, y_pred_train_lr_with_weights,
    y_true, y_pred_test_lr_with_weights
)

results['mlp'] = print_results(
    "METHOD 4: Single Layer Perceptron (MLP with 10 hidden neurons)",
    y_train, y_pred_train_mlp,
    y_true, y_pred_test_mlp
)

results['weighted'] = print_results(
    "METHOD 5: Weighted Classifier (linear weights + argmax)",
    y_train, y_pred_train_weighted,
    y_true, y_pred_test_weighted
)

# ============================================================================
# SAVE TO FILE
# ============================================================================

if args.model_dir:
    report_file = f"{args.model_dir}/eval_comparison_report.txt"
else:
    report_file = "eval_comparison_report.txt"

with open(report_file, "w") as f:
    # Write prediction differences
    f.write("="*80 + "\n")
    f.write("PREDICTION DIFFERENCES FROM ARGMAX BASELINE\n")
    f.write("="*80 + "\n\n")
    f.write("Training Set:\n")
    f.write(f"  LR without weights:  {train_diff_lr_no_weights}/{len(y_train)} ({100*train_diff_lr_no_weights/len(y_train):.2f}%)\n")
    f.write(f"  LR with weights:     {train_diff_lr_with_weights}/{len(y_train)} ({100*train_diff_lr_with_weights/len(y_train):.2f}%)\n")
    f.write(f"  MLP (10 hidden):     {train_diff_mlp}/{len(y_train)} ({100*train_diff_mlp/len(y_train):.2f}%)\n")
    f.write(f"  Weighted + argmax:   {train_diff_weighted}/{len(y_train)} ({100*train_diff_weighted/len(y_train):.2f}%)\n")
    f.write("\nTest Set:\n")
    f.write(f"  LR without weights:  {test_diff_lr_no_weights}/{len(y_true)} ({100*test_diff_lr_no_weights/len(y_true):.2f}%)\n")
    f.write(f"  LR with weights:     {test_diff_lr_with_weights}/{len(y_true)} ({100*test_diff_lr_with_weights/len(y_true):.2f}%)\n")
    f.write(f"  MLP (10 hidden):     {test_diff_mlp}/{len(y_true)} ({100*test_diff_mlp/len(y_true):.2f}%)\n")
    f.write(f"  Weighted + argmax:   {test_diff_weighted}/{len(y_true)} ({100*test_diff_weighted/len(y_true):.2f}%)\n")
    f.write("\n\n")
    
    # Write learned weights
    f.write("="*80 + "\n")
    f.write("LEARNED WEIGHTS ON CLASSIFIERS\n")
    f.write("="*80 + "\n\n")
    
    f.write("Logistic Regression WITHOUT class weights - Coefficients:\n")
    f.write("  [noop, fire, up, right, left, down]\n")
    for i, action in enumerate(primitive_actions):
        f.write(f"  {action:6s}: {lr_no_weights.coef_[i]}\n")
    f.write(f"\nIntercepts: {lr_no_weights.intercept_}\n")
    
    f.write("\n" + "-"*80 + "\n\n")
    f.write("Logistic Regression WITH class weights - Coefficients:\n")
    f.write("  [noop, fire, up, right, left, down]\n")
    for i, action in enumerate(primitive_actions):
        f.write(f"  {action:6s}: {lr_with_weights.coef_[i]}\n")
    f.write(f"\nIntercepts: {lr_with_weights.intercept_}\n")
    
    f.write("\n" + "-"*80 + "\n\n")
    f.write("Single Layer Perceptron (MLP with 10 hidden neurons):\n")
    f.write(f"  Input layer -> Hidden layer weights shape: {mlp.coefs_[0].shape}\n")
    f.write(f"  Hidden layer -> Output layer weights shape: {mlp.coefs_[1].shape}\n")
    f.write(f"\n  Input -> Hidden weights (first 5 rows):\n")
    for i in range(min(6, mlp.coefs_[0].shape[0])):
        f.write(f"    Classifier {primitive_actions[i]:6s}: {mlp.coefs_[0][i][:5]}...\n")
    f.write(f"\n  Hidden -> Output weights (transposed, shows contribution to each action):\n")
    for i in range(mlp.coefs_[1].shape[1]):
        f.write(f"    To {primitive_actions[i]:6s}: {mlp.coefs_[1][:, i]}\n")
    
    f.write("\n" + "-"*80 + "\n\n")
    f.write("Weighted Classifier (linear weights + argmax):\n")
    f.write(f"  Weights shape: {weighted_clf.weights_.shape}\n")
    f.write(f"\n  Learned weights for each (classifier -> action):\n")
    f.write(f"  {'':12s} " + " ".join([f"{a:>8s}" for a in primitive_actions]) + "\n")
    for i, action in enumerate(primitive_actions):
        weights_str = " ".join([f"{weighted_clf.weights_[i, j]:8.4f}" for j in range(len(primitive_actions))])
        f.write(f"  {action:12s} {weights_str}\n")
    f.write(f"\n  Bias terms: {weighted_clf.bias_}\n")
    f.write("\n\n")
    
    # Write performance results
    for method_name, result in [
        ("METHOD 1: Direct argmax (non-calibrated)", results['argmax']),
        ("METHOD 2: Logistic Regression WITHOUT class weights", results['lr_no_weights']),
        ("METHOD 3: Logistic Regression WITH class weights (balanced)", results['lr_with_weights']),
        ("METHOD 4: Single Layer Perceptron (MLP with 10 hidden neurons)", results['mlp']),
        ("METHOD 5: Weighted Classifier (linear weights + argmax)", results['weighted'])
    ]:
        f.write("="*80 + "\n")
        f.write(f"{method_name}\n")
        f.write("="*80 + "\n\n")
        
        f.write("--- TRAINING SET ---\n")
        f.write(result['train_report'])
        f.write("\nConfusion Matrix:\n")
        f.write(str(result['train_cm']))
        f.write("\n\n")
        
        f.write("--- TEST SET ---\n")
        f.write(result['test_report'])
        f.write("\nConfusion Matrix:\n")
        f.write(str(result['test_cm']))
        f.write("\n\n\n")

print(f"\n✅ Evaluation report saved to: {report_file}")
