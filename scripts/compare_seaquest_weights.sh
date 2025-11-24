#!/bin/bash

ACTION="fire"  # or any other action: noop, up, down, left, right
DATA_DIR="data/seaquest/all/${ACTION}"

cd /home/nikhilesh/Projects/NeSY-Imitation-Learning/rdnboost

echo "=========================================="
echo "BASELINE: Training WITHOUT custom weights"
echo "=========================================="

# Backup weight files
mv ../${DATA_DIR}/train/train_pos_weights.txt ../${DATA_DIR}/train/train_pos_weights.txt.bak 2>/dev/null
mv ../${DATA_DIR}/train/train_neg_weights.txt ../${DATA_DIR}/train/train_neg_weights.txt.bak 2>/dev/null

# Run training
java -cp "target/classes:lib/*" edu.wisc.cs.will.Boosting.RDN.RunBoostedRDN -l -train ../${DATA_DIR}/train/ -target ${ACTION} -trees 2 2>&1 | grep -E "(Loaded.*weight|covers [0-9]+|weighted_variance|TRUE branch|FALSE branch)" | head -20

echo ""
echo "=========================================="
echo "WITH WEIGHTS: Training WITH custom weights"
echo "=========================================="

# Restore weight files
mv ../${DATA_DIR}/train/train_pos_weights.txt.bak ../${DATA_DIR}/train/train_pos_weights.txt
mv ../${DATA_DIR}/train/train_neg_weights.txt.bak ../${DATA_DIR}/train/train_neg_weights.txt

# Run training
java -cp "target/classes:lib/*" edu.wisc.cs.will.Boosting.RDN.RunBoostedRDN -l -train ../${DATA_DIR}/train/ -target ${ACTION} -trees 2 2>&1 | grep -E "(Loaded.*weight|covers [0-9]+|weighted_variance|TRUE branch|FALSE branch)" | head -20

echo ""
echo "=========================================="
echo "COMPARISON COMPLETE"
echo "=========================================="
