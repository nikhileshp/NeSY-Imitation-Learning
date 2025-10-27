#!/bin/bash

cd /home/nikhilesh/Projects/NeSY-Imitation-Learning/rdnboost

echo "=========================================="
echo "BASELINE: Training WITHOUT custom weights"
echo "=========================================="

# Backup weight files
mv sample/ICML/train/train_pos_weights.txt sample/ICML/train/train_pos_weights.txt.bak 2>/dev/null
mv sample/ICML/train/train_neg_weights.txt sample/ICML/train/train_neg_weights.txt.bak 2>/dev/null

# Run training
java -cp "target/classes:lib/*" edu.wisc.cs.will.Boosting.RDN.RunBoostedRDN -l -train sample/ICML/train/ -target CoAuthor -trees 2 2>&1 | grep -E "(Loaded.*weight|covers [0-9]+|weighted_variance|TRUE branch|FALSE branch)" | head -15

echo ""
echo "=========================================="
echo "WITH WEIGHTS: Training WITH custom weights"
echo "=========================================="

# Restore weight files
mv sample/ICML/train/train_pos_weights.txt.bak sample/ICML/train/train_pos_weights.txt
mv sample/ICML/train/train_neg_weights.txt.bak sample/ICML/train/train_neg_weights.txt

# Run training
java -cp "target/classes:lib/*" edu.wisc.cs.will.Boosting.RDN.RunBoostedRDN -l -train sample/ICML/train/ -target CoAuthor -trees 2 2>&1 | grep -E "(Loaded.*weight|covers [0-9]+|weighted_variance|TRUE branch|FALSE branch)" | head -15

echo ""
echo "=========================================="
echo "COMPARISON COMPLETE"
echo "=========================================="
