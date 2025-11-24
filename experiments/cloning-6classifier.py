#!/usr/bin/env python3
"""
Training 6 Separate Binary Classifiers (One-vs-Rest Approach)
==============================================================

This script:
1. Filters actions > 5 (keeps actions 0-5)
2. Grounds relationships only (ignores objects)
3. Creates 6 separate binary datasets (one per action)
4. Trains 6 separate binary classifiers
5. Combines them into an ensemble for multi-class prediction

Each classifier answers: "Is this action X?" (yes/no)
The ensemble picks the action with highest confidence.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# PART 1: DATA LOADING AND PREPARATION
# ============================================================================

def load_data(filename):
    """Load and parse the RTF file"""
    print("\n" + "="*80)
    print("LOADING DATA")
    print("="*80)

    with open(filename, 'r') as f:
        content = f.read()

    lines = content.split('\n')
    data_start = None
    for i, line in enumerate(lines):
        if 'frameid\tepisode_id' in line:
            data_start = i
            break

    header = lines[data_start].strip()
    header = header.replace('\\f0\\fs24 \\cf0 ', '').replace('\\', '')
    columns = [col.strip() for col in header.split('\t')]

    data_rows = []
    for line in lines[data_start+1:]:
        if line.strip() and '\t' in line:
            data_rows.append(line.strip())

    data = []
    for row in data_rows:
        fields = row.split('\t')
        if len(fields) == len(columns):
            data.append(fields)

    df = pd.DataFrame(data, columns=columns)
    print(f"✓ Loaded {len(df)} samples")
    return df


def parse_relationships(rel_string):
    """Parse relationship string into a list"""
    if pd.isna(rel_string) or rel_string.strip() == '':
        return []
    return [rel.strip() for rel in rel_string.split(',') if rel.strip()]


def ground_relationships(df):
    """Ground relationships into binary features"""
    print("\n" + "="*80)
    print("GROUNDING RELATIONSHIPS")
    print("="*80)

    df['relationships_list'] = df['relationships'].apply(parse_relationships)

    all_relationships = set()
    for rels in df['relationships_list']:
        all_relationships.update(rels)

    all_relationships = sorted(list(all_relationships))
    print(f"✓ Found {len(all_relationships)} unique relationships")

    relationship_features = np.zeros((len(df), len(all_relationships)))
    for i, rels in enumerate(df['relationships_list']):
        for rel in rels:
            if rel in all_relationships:
                j = all_relationships.index(rel)
                relationship_features[i, j] = 1

    print(f"✓ Feature matrix shape: {relationship_features.shape}")
    return relationship_features, all_relationships


# ============================================================================
# PART 2: CREATE BINARY DATASETS
# ============================================================================

def create_binary_datasets(X, y):
    """Create 6 binary datasets (one-vs-rest for each action)"""
    print("\n" + "="*80)
    print("CREATING 6 BINARY DATASETS (One-vs-Rest)")
    print("="*80)

    actions = sorted(np.unique(y))
    binary_datasets = {}

    for action in actions:
        # Create binary labels: 1 if this action, 0 otherwise
        y_binary = (y == action).astype(int)

        num_positive = np.sum(y_binary == 1)
        num_negative = np.sum(y_binary == 0)

        binary_datasets[action] = {
            'X': X,
            'y': y_binary,
            'positive_samples': num_positive,
            'negative_samples': num_negative
        }

        print(f"✓ Action {action}: {num_positive:3d} positive, {num_negative:3d} negative "
              f"(ratio: {num_negative/num_positive:.1f}:1)")

    return binary_datasets, actions


# ============================================================================
# PART 3: TRAIN BINARY CLASSIFIERS
# ============================================================================

def train_binary_classifier(X, y, action_id):
    """Train a single binary classifier for one action"""

    # Train-test split with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train binary classifier
    clf = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation='relu',
        solver='adam',
        alpha=0.0001,
        batch_size=16,
        learning_rate='adaptive',
        learning_rate_init=0.001,
        max_iter=100,
        random_state=42,
        verbose=False,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=10
    )

    clf.fit(X_train_scaled, y_train)

    # Evaluate
    y_train_pred = clf.predict(X_train_scaled)
    y_test_pred = clf.predict(X_test_scaled)

    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)

    # Calculate precision, recall, F1
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_test_pred, average='binary', zero_division=0
    )

    results = {
        'classifier': clf,
        'scaler': scaler,
        'train_acc': train_acc,
        'test_acc': test_acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'X_test': X_test_scaled,
        'y_test': y_test,
        'y_test_pred': y_test_pred
    }

    return results


def train_all_classifiers(binary_datasets, actions):
    """Train all 6 binary classifiers"""
    print("\n" + "="*80)
    print("TRAINING 6 BINARY CLASSIFIERS")
    print("="*80)

    all_results = {}

    for action in actions:
        print(f"\n✓ Training classifier for Action {action}...")

        X_action = binary_datasets[action]['X']
        y_action = binary_datasets[action]['y']

        results = train_binary_classifier(X_action, y_action, action)
        all_results[action] = results

        print(f"   Train: {results['train_acc']*100:>5.2f}% | "
              f"Test: {results['test_acc']*100:>5.2f}% | "
              f"Precision: {results['precision']:.3f} | "
              f"Recall: {results['recall']:.3f} | "
              f"F1: {results['f1']:.3f}")

    return all_results


# ============================================================================
# PART 4: ENSEMBLE PREDICTION
# ============================================================================

def predict_ensemble(X_samples, all_results, actions):
    """
    Predict using all 6 classifiers and choose action with highest confidence.

    For each sample:
    - Get confidence score from each of the 6 binary classifiers
    - Choose the action with the highest confidence
    """
    predictions = []

    for sample in X_samples:
        sample = sample.reshape(1, -1)
        confidences = {}

        for action in actions:
            # Scale the sample
            scaler = all_results[action]['scaler']
            sample_scaled = scaler.transform(sample)

            # Get probability for this action
            clf = all_results[action]['classifier']
            proba = clf.predict_proba(sample_scaled)[0]

            # Confidence that this IS the action (probability of class 1)
            confidences[action] = proba[1] if len(proba) > 1 else proba[0]

        # Choose action with highest confidence
        predicted_action = max(confidences, key=confidences.get)
        predictions.append(predicted_action)

    return np.array(predictions)


def evaluate_ensemble(X, y, all_results, actions):
    """Evaluate the ensemble of all 6 classifiers"""
    print("\n" + "="*80)
    print("ENSEMBLE EVALUATION")
    print("="*80)

    # Split data for ensemble evaluation
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Make ensemble predictions
    y_test_pred = predict_ensemble(X_test, all_results, actions)

    # Evaluate
    ensemble_acc = accuracy_score(y_test, y_test_pred)

    print(f"\n✓ Ensemble Test Accuracy: {ensemble_acc*100:.2f}%")

    print("\n✓ Classification Report:")
    print(classification_report(y_test, y_test_pred))

    print("✓ Confusion Matrix:")
    cm = confusion_matrix(y_test, y_test_pred)
    print(cm)

    return ensemble_acc, y_test, y_test_pred


# ============================================================================
# PART 5: SUMMARY AND VISUALIZATION
# ============================================================================

def print_summary(all_results, actions, ensemble_acc):
    """Print comprehensive summary"""
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    print("\n✓ Individual Binary Classifier Performance:")
    print("-" * 80)
    print(f"{'Action':<8} {'Train Acc':<12} {'Test Acc':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
    print("-" * 80)

    for action in actions:
        r = all_results[action]
        print(f"{action:<8} {r['train_acc']*100:>10.2f}% {r['test_acc']*100:>10.2f}% "
              f"{r['precision']:>10.3f}  {r['recall']:>10.3f}  {r['f1']:>10.3f}")

    print("\n✓ Ensemble Performance:")
    print("-" * 80)
    print(f"   Combined Test Accuracy: {ensemble_acc*100:.2f}%")
    print(f"   ")
    print(f"   The ensemble uses all 6 binary classifiers together.")
    print(f"   For each prediction, it chooses the action with the highest")
    print(f"   confidence score across all classifiers.")


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main(filename='relationships_perplexity.rtf', max_action=5):
    """Main function to run the complete pipeline"""

    print("\n")
    print("╔" + "="*78 + "╗")
    print("║" + " "*15 + "6 SEPARATE BINARY CLASSIFIERS (One-vs-Rest)" + " "*19 + "║")
    print("╚" + "="*78 + "╝")

    # Step 1: Load data
    df = load_data(filename)
    df['action'] = df['action'].astype(int)

    # Step 2: Filter actions
    df_filtered = df[df['action'] <= max_action].copy()
    print(f"\n✓ Filtered to {len(df_filtered)} samples (actions ≤ {max_action})")

    # Step 3: Ground relationships
    X, relationship_names = ground_relationships(df_filtered)
    y = df_filtered['action'].values

    # Step 4: Create binary datasets
    binary_datasets, actions = create_binary_datasets(X, y)

    # Step 5: Train all classifiers
    all_results = train_all_classifiers(binary_datasets, actions)

    # Step 6: Evaluate ensemble
    ensemble_acc, y_test, y_test_pred = evaluate_ensemble(X, y, all_results, actions)

    # Step 7: Print summary
    print_summary(all_results, actions, ensemble_acc)

    print("\n" + "="*80)
    print("COMPLETE")
    print("="*80)
    print(f"\n✓ Trained 6 separate binary classifiers")
    print(f"✓ Each classifier trained on its own binary dataset")
    print(f"✓ Ensemble accuracy: {ensemble_acc*100:.2f}%")
    print()

    return all_results, binary_datasets, actions


if __name__ == "__main__":
    all_results, binary_datasets, actions = main('data/seaquest/gaze_data_tmp/relationships.txt')