#!/usr/bin/env python3
"""
Cloning with RDN-style Data Loading
===================================

This script trains binary classifiers (one-vs-rest) using data directly from
the RDN data directories (train_facts.txt, train_pos.txt, train_neg.txt).

It parses Prolog-style facts to create feature vectors (bag-of-predicates).
"""

import os
import re
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import average_precision_score, f1_score, accuracy_score, roc_auc_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
import argparse
from collections import defaultdict
import joblib
import os

# Configuration
ACTIONS = ["fire", "up", "down", "left", "right", "noop"]
BASE_DIR = "data/seaquest/all"

def parse_facts(facts_file):
    """
    Parse train_facts.txt to extract features for each state.
    Returns:
        state_features: dict {state_id: {feature_name: 1}}
        all_features: set of all unique feature names
    """
    print(f"Loading facts from {facts_file}...", flush=True)
    state_features = defaultdict(set)
    all_features = set()
    
    with open(facts_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('%'):
                continue
            
            # Parse predicate(state, args...)
            # Regex to capture predicate name and arguments inside parens
            match = re.match(r'([a-z0-9_]+)\((.*?)\)\.', line)
            if match:
                predicate = match.group(1)
                args_str = match.group(2)
                args = [arg.strip() for arg in args_str.split(',')]
                
                if not args:
                    continue
                    
                state_id = args[0]
                
                # Construct feature name
                # If arity > 1, include other args in feature name
                # e.g., visibleenemy(s1, e1) -> visibleenemy_e1
                # But wait, RDN usually grounds by object existence.
                # For cloning, a simple bag-of-atoms approach is:
                # Feature: "predicate" (if arity 1)
                # Feature: "predicate_arg2_arg3..." (if arity > 1)
                
                # However, object names like 'enemy1', 'enemy2' might be specific instances.
                # We probably want to generalize or keep them if they are consistent.
                # In Seaquest, 'enemy1' is likely a specific slot.
                
                if len(args) == 1:
                    feature = predicate
                else:
                    # Join remaining args
                    feature = f"{predicate}_{'_'.join(args[1:])}"
                
                state_features[state_id].add(feature)
                all_features.add(feature)
                
    print(f"Found {len(all_features)} unique features across {len(state_features)} states.", flush=True)
    return state_features, sorted(list(all_features))

def load_examples(pos_file, neg_file):
    """
    Load positive and negative examples.
    Returns:
        pos_states: list of state_ids
        neg_states: list of state_ids
    """
    pos_states = []
    neg_states = []
    
    # Regex for action(state, action_name).
    # Note: action name might be in the file, but we assume the file itself 
    # corresponds to the binary classification task for that action.
    
    with open(pos_file, 'r') as f:
        for line in f:
            match = re.search(r'action\((.*?),\s*.*?\)\.', line)
            if match:
                pos_states.append(match.group(1))
                
    with open(neg_file, 'r') as f:
        for line in f:
            match = re.search(r'action\((.*?),\s*.*?\)\.', line)
            if match:
                neg_states.append(match.group(1))
                
    return pos_states, neg_states

def create_dataset(state_features, all_features, pos_states, neg_states):
    """
    Create X and y matrices.
    """
    feature_to_idx = {f: i for i, f in enumerate(all_features)}
    num_features = len(all_features)
    
    states = pos_states + neg_states
    y = np.array([1] * len(pos_states) + [0] * len(neg_states))
    
    X = np.zeros((len(states), num_features), dtype=np.float32)
    
    for i, state_id in enumerate(states):
        if state_id in state_features:
            for feat in state_features[state_id]:
                if feat in feature_to_idx:
                    X[i, feature_to_idx[feat]] = 1.0
                    
    return X, y

def train_and_evaluate(action, seed, save_dir, negpos_ratio, debug=False):
    print(f"\n{'='*40}", flush=True)
    print(f"Processing Action: {action} (Seed: {seed}, Ratio: {negpos_ratio})", flush=True)
    print(f"{'='*40}", flush=True)
    
    # Create output directory structure
    # Structure: rdn_models/seaquest/all/negpos_{NEGPOS}_mlp_64_32_bc/{action}/seed_{seed}/
    experiment_dir = os.path.join(save_dir, f"negpos_{int(negpos_ratio)}_mlp_64_32_bc", action, f"seed_{seed}")
    os.makedirs(experiment_dir, exist_ok=True)
    
    # --- Training Data Loading ---
    print("Loading Training Data...", flush=True)
    train_facts_file = f"data/seaquest/all/{action}/train/train_facts.txt"
    train_pos_file = f"data/seaquest/all/{action}/train/train_pos.txt"
    train_neg_file = f"data/seaquest/all/{action}/train/train_neg.txt"
    
    # Parse facts
    feature_map, unique_features = parse_facts(train_facts_file)
    
    # Load examples
    pos_examples, neg_examples = load_examples(train_pos_file, train_neg_file)
    
    if debug:
        pos_examples = pos_examples[:100]
        neg_examples = neg_examples[:200]
        print("DEBUG MODE: Reduced training examples.", flush=True)

    # Downsample training negatives
    n_pos = len(pos_examples)
    n_neg_keep = int(n_pos * negpos_ratio)
    if len(neg_examples) > n_neg_keep:
        # Use seed for reproducibility
        rng = np.random.RandomState(seed)
        neg_indices = rng.choice(len(neg_examples), n_neg_keep, replace=False)
        neg_examples = [neg_examples[i] for i in neg_indices]
        print(f"Downsampling Training Negatives to ratio {negpos_ratio}:1...", flush=True)
    
    print(f"Training Examples: {len(pos_examples)} Pos, {len(neg_examples)} Neg", flush=True)

    # Create dataset
    X_train, y_train = create_dataset(feature_map, unique_features, pos_examples, neg_examples)
    
    # Scale training data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # --- Training ---
    print("Training MLP Classifier...", flush=True)
    # Use seed for MLP initialization
    clf = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=200 if not debug else 10, 
                        random_state=seed, verbose=True)
    clf.fit(X_train_scaled, y_train)
    
    print(f"Model Parameters:", flush=True)
    print(f"  Hidden Layers: {clf.hidden_layer_sizes}", flush=True)
    print(f"  Max Iterations: {clf.max_iter}", flush=True)
    print(f"  Solver: {clf.solver}", flush=True)
    print(f"  Activation: {clf.activation}", flush=True)
    
    # Save Model
    model_path = os.path.join(experiment_dir, "model.ckpt")
    joblib.dump(clf, model_path)
    print(f"Model saved to {model_path}", flush=True)

    # --- Testing ---
    print("Loading Test Data...", flush=True)
    test_facts_file = f"data/seaquest/all/{action}/test/test_facts.txt"
    test_pos_file = f"data/seaquest/all/{action}/test/test_pos.txt"
    test_neg_file = f"data/seaquest/all/{action}/test/test_neg.txt"
    
    test_feature_map, _ = parse_facts(test_facts_file)
    test_pos_examples, test_neg_examples_all = load_examples(test_pos_file, test_neg_file)
    
    if debug:
        test_pos_examples = test_pos_examples[:50]
        test_neg_examples_all = test_neg_examples_all[:100]
        print("DEBUG MODE: Reduced test examples.", flush=True)

    # Evaluate on the specific ratio
    print(f"\nEvaluating with NegPos Ratio: {negpos_ratio}", flush=True)
    
    # Downsample test negatives
    n_test_pos = len(test_pos_examples)
    n_test_neg_keep = int(n_test_pos * negpos_ratio)
    
    current_test_neg = test_neg_examples_all
    if len(test_neg_examples_all) > n_test_neg_keep:
        # Use same seed for test sampling consistency within this run
        rng = np.random.RandomState(seed) 
        neg_indices = rng.choice(len(test_neg_examples_all), n_test_neg_keep, replace=False)
        current_test_neg = [test_neg_examples_all[i] for i in neg_indices]
        print(f"Downsampling Test Negatives to ratio {negpos_ratio}:1...", flush=True)
    
    print(f"Test Examples: {len(test_pos_examples)} Pos, {len(current_test_neg)} Neg", flush=True)
    
    X_test, y_test = create_dataset(test_feature_map, unique_features, test_pos_examples, current_test_neg)
    
    # Scale test data using the *trained* scaler
    X_test_scaled = scaler.transform(X_test)

    # Predict
    if len(X_test_scaled) > 0:
        y_pred_prob = clf.predict_proba(X_test_scaled)[:, 1]
        y_pred = clf.predict(X_test_scaled)
        
        auc_pr = average_precision_score(y_test, y_pred_prob)
        f1 = f1_score(y_test, y_pred)
        acc = accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_prob)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        threshold = 0.5
    else:
        auc_pr, f1, acc, roc_auc, precision, recall, threshold = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5
        
    print(f"Results for {action} (Ratio {negpos_ratio}):", flush=True)
    print(f"  AUC PR:    {auc_pr:.4f}", flush=True)
    print(f"  AUC ROC:   {roc_auc:.4f}", flush=True)
    print(f"  F1:        {f1:.4f}", flush=True)
    print(f"  Precision: {precision:.4f}", flush=True)
    print(f"  Recall:    {recall:.4f}", flush=True)
    print(f"  Acc:       {acc:.4f}", flush=True)
    print(f"  Threshold: {threshold}", flush=True)
    
    # Log to file
    log_file = os.path.join(experiment_dir, "test_infer.log")
    with open(log_file, "w") as f:
        f.write(f"Results for {action} (Seed {seed}, Ratio {negpos_ratio}):\n")
        f.write(f"AUC PR:    {auc_pr:.4f}\n")
        f.write(f"AUC ROC:   {roc_auc:.4f}\n")
        f.write(f"F1:        {f1:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall:    {recall:.4f}\n")
        f.write(f"Acc:       {acc:.4f}\n")
        f.write(f"Threshold: {threshold}\n")
        # You might want to dump more detailed logs here if needed by the user's parsing scripts
        f.write(f"\nModel Parameters:\n{clf.get_params()}\n")

    return auc_pr, f1

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True, help="Random seed")
    parser.add_argument("--action", type=str, default=None, help="Specific action to run (default: all)")
    parser.add_argument("--save_dir", type=str, default="rdn_models/seaquest/all", help="Directory to save models and logs")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode (fewer examples, fewer iters)")
    args = parser.parse_args()
    
    actions_to_run = [args.action] if args.action else ACTIONS
    ratios_to_run = [2.0, 1.0]
    
    results = []
    for action in actions_to_run:
        for ratio in ratios_to_run:
            auc_pr, f1 = train_and_evaluate(action, args.seed, args.save_dir, ratio, args.debug)
            results.append((action, ratio, auc_pr, f1))
        
    print("\n" + "="*40, flush=True)
    print(f"FINAL SUMMARY (Seed {args.seed})", flush=True)
    print("="*40, flush=True)
    print(f"{'Action':<10} {'Ratio':<6} {'AUC PR':<10} {'F1':<10}", flush=True)
    print("-" * 40, flush=True)
    for action, ratio, auc_pr, f1 in results:
        print(f"{action:<10} {ratio:<6} {auc_pr:.4f}     {f1:.4f}", flush=True)

if __name__ == "__main__":
    main()
