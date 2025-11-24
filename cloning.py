#!/usr/bin/env python3
"""
Modified Relationship Grounding and Behavior Cloning Implementation
====================================================================

Changes from original:
1. Filters out rows with actions > 5
2. Only uses relationships column for grounding (ignores objects column)

Ignores: objects, goal, distance_weights, predicate_weights, example_weight, trajectory
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# PART 1: DATA LOADING AND PARSING
# ============================================================================

def load_data(filename):
    """Load and parse the RTF file containing the dataset"""
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

    if data_start is None:
        raise ValueError("Could not find header in file")

    # Get header
    header = lines[data_start].strip()
    header = header.replace('\\f0\\fs24 \\cf0 ', '').replace('\\', '')
    columns = [col.strip() for col in header.split('\t')]

    # Get data rows
    data_rows = []
    for line in lines[data_start+1:]:
        if line.strip() and '\t' in line:
            data_rows.append(line.strip())

    # Create dataframe
    data = []
    for row in data_rows:
        fields = row.split('\t')
        if len(fields) == len(columns):
            data.append(fields)

    df = pd.DataFrame(data, columns=columns)
    print(f"✓ Loaded {len(df)} samples")
    return df


def filter_actions(df, max_action=5):
    """Filter out rows with actions greater than max_action"""
    print("\n" + "="*80)
    print("FILTERING ACTIONS")
    print("="*80)

    df['action'] = df['action'].astype(int)
    initial_count = len(df)
    initial_actions = sorted(df['action'].unique())

    print(f"✓ Initial samples: {initial_count}")
    print(f"✓ Initial actions: {initial_actions}")

    # Filter
    df_filtered = df[df['action'] <= max_action].copy()
    removed_count = initial_count - len(df_filtered)
    final_actions = sorted(df_filtered['action'].unique())

    print(f"✓ Removed {removed_count} rows with actions > {max_action}")
    print(f"✓ Remaining samples: {len(df_filtered)}")
    print(f"✓ Final actions: {final_actions}")

    # Show action distribution
    print(f"\n✓ Action distribution:")
    unique, counts = np.unique(df_filtered['action'], return_counts=True)
    for action, count in zip(unique, counts):
        print(f"   Action {action}: {count:3d} samples ({100*count/len(df_filtered):5.1f}%)")

    return df_filtered


# ============================================================================
# PART 2: RELATIONSHIP GROUNDING (OBJECTS IGNORED)
# ============================================================================

def parse_relationships(rel_string):
    """Parse relationship string into a list of relationships"""
    if pd.isna(rel_string) or rel_string.strip() == '':
        return []

    relationships = []
    for rel in rel_string.split(','):
        rel = rel.strip()
        if rel:
            relationships.append(rel)
    return relationships


def ground_relationships(df):
    """
    Ground ONLY relationships into a binary feature vector.
    IGNORES: objects, goal, distance_weights, predicate_weights, example_weight, trajectory
    """
    print("\n" + "="*80)
    print("GROUNDING RELATIONSHIPS (Objects Column IGNORED)")
    print("="*80)

    # Parse relationships for all samples
    df['relationships_list'] = df['relationships'].apply(parse_relationships)

    # Collect all unique relationships across the dataset
    all_relationships = set()
    for rels in df['relationships_list']:
        all_relationships.update(rels)

    all_relationships = sorted(list(all_relationships))
    print(f"✓ Found {len(all_relationships)} unique relationships")
    print(f"✓ Example relationships:")
    for i, rel in enumerate(all_relationships[:10]):
        print(f"   - {rel}")

    # Create binary feature matrix
    relationship_features = np.zeros((len(df), len(all_relationships)))

    for i, rels in enumerate(df['relationships_list']):
        for rel in rels:
            if rel in all_relationships:
                j = all_relationships.index(rel)
                relationship_features[i, j] = 1

    print(f"✓ Relationship feature matrix shape: {relationship_features.shape}")
    return relationship_features, all_relationships


# ============================================================================
# PART 3: FEATURE PREPARATION
# ============================================================================

def prepare_features_and_targets(df, X_relationships, relationship_names):
    """Prepare features and target variable"""
    print("\n" + "="*80)
    print("PREPARING FEATURES AND TARGETS")
    print("="*80)

    # Use only relationship features
    X = X_relationships
    print(f"✓ Feature matrix shape: {X.shape}")

    # Extract actions (target variable)
    y = df['action'].values
    print(f"✓ Target (action) shape: {y.shape}")
    print(f"✓ Unique actions: {sorted(np.unique(y))}")

    # Feature names
    feature_names = relationship_names
    print(f"\n✓ Total grounded features: {len(feature_names)}")
    print(f"   - Relationship features: {len(relationship_names)}")
    print(f"   - Object features: 0 (IGNORED)")

    return X, y, feature_names


# ============================================================================
# PART 4: BEHAVIOR CLONING
# ============================================================================

def train_behavior_cloning(X_train, y_train, X_test, y_test):
    """Train behavior cloning model using neural network"""
    print("\n" + "="*80)
    print("TRAINING BEHAVIOR CLONING MODEL")
    print("="*80)

    # Standardize features
    print("✓ Standardizing features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Model architecture
    print(f"\n✓ Neural Network Architecture:")
    print(f"   - Input layer: {X_train_scaled.shape[1]} features (relationships only)")
    print(f"   - Hidden layers: (256, 128, 64)")
    print(f"   - Output layer: {len(np.unique(y_train))} actions")
    print(f"   - Activation: ReLU")
    print(f"   - Optimizer: Adam")

    # Train MLPClassifier
    print(f"\n✓ Training model...")
    model = MLPClassifier(
        hidden_layer_sizes=(256, 128, 64),
        activation='relu',
        solver='adam',
        alpha=0.0001,
        batch_size=32,
        learning_rate='adaptive',
        learning_rate_init=0.001,
        max_iter=100,
        random_state=42,
        verbose=False,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=10
    )

    model.fit(X_train_scaled, y_train)

    return model, scaler, X_train_scaled, X_test_scaled


def evaluate_model(model, X_train_scaled, y_train, X_test_scaled, y_test):
    """Evaluate the trained model"""
    print("\n" + "="*80)
    print("MODEL EVALUATION")
    print("="*80)

    # Training accuracy
    y_train_pred = model.predict(X_train_scaled)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    print(f"\n✓ Training Accuracy: {train_accuracy*100:.2f}%")

    # Test accuracy
    y_test_pred = model.predict(X_test_scaled)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    print(f"✓ Test Accuracy: {test_accuracy*100:.2f}%")

    # Detailed classification report
    print(f"\n✓ Classification Report:")
    print(classification_report(y_test, y_test_pred))

    # Confusion Matrix
    print(f"✓ Confusion Matrix:")
    cm = confusion_matrix(y_test, y_test_pred)
    print(cm)

    return train_accuracy, test_accuracy, y_test_pred


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main(filename='relationships_perplexity.rtf', max_action=5):
    """Main function to run the entire pipeline"""

    print("\n")
    print("╔" + "="*78 + "╗")
    print("║" + " "*10 + "MODIFIED RELATIONSHIP GROUNDING & BEHAVIOR CLONING" + " "*17 + "║")
    print("╚" + "="*78 + "╝")
    print("\n✓ Changes: Filter actions > 5, Use relationships only (ignore objects)")

    # Step 1: Load data
    df = load_data(filename)

    # Step 2: Filter actions
    df_filtered = filter_actions(df, max_action=max_action)

    # Step 3: Ground relationships (objects ignored)
    X_relationships, relationship_names = ground_relationships(df_filtered)

    # Step 4: Prepare features and targets
    X, y, feature_names = prepare_features_and_targets(
        df_filtered, X_relationships, relationship_names
    )

    # Step 5: Split data
    print("\n" + "="*80)
    print("SPLITTING DATA")
    print("="*80)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"✓ Training set: {X_train.shape[0]} samples")
    print(f"✓ Test set: {X_test.shape[0]} samples")

    # Step 6: Train behavior cloning model
    model, scaler, X_train_scaled, X_test_scaled = train_behavior_cloning(
        X_train, y_train, X_test, y_test
    )

    # Step 7: Evaluate model
    train_acc, test_acc, predictions = evaluate_model(
        model, X_train_scaled, y_train, X_test_scaled, y_test
    )

    # Final summary
    print("\n")
    print("╔" + "="*78 + "╗")
    print("║" + " "*30 + "SUMMARY" + " "*41 + "║")
    print("╠" + "="*78 + "╣")
    print(f"║  Original Samples: {len(df):<61}║")
    print(f"║  Filtered Samples (action ≤ {max_action}): {len(df_filtered):<47}║")
    print(f"║  Grounded Features: {len(feature_names):<59}║")
    print(f"║    - Relationships: {len(relationship_names):<59}║")
    print(f"║    - Objects: 0 (IGNORED){' '*50}║")
    print(f"║  Training Accuracy: {train_acc*100:5.2f}%{' '*54}║")
    print(f"║  Test Accuracy: {test_acc*100:5.2f}%{' '*58}║")
    print("╚" + "="*78 + "╝")
    print("\n✓ All done!\n")

    return model, scaler, feature_names, df_filtered


if __name__ == "__main__":
    # Run the complete pipeline
    model, scaler, feature_names, df = main('data/seaquest/gaze_data_tmp/relationships.txt', max_action=5)
