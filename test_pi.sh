#!/bin/bash
set -e

# Configuration
JAR="rdnboost-pi/target/boostsrl-weights-2.0.0.jar"
AUC_JAR="rdnboost-pi/src/edu/wisc/cs/will/DataSetUtils/"
TEST_DIR="pi_test_data/train"

# Clean up previous test
rm -rf pi_test_data

# Create test directories
mkdir -p "$TEST_DIR"

# Create dummy data files
echo "dummy_fact(a)." > "$TEST_DIR/train_facts.txt"
echo "target(a)." > "$TEST_DIR/train_pos.txt"
echo "target(b)." > "$TEST_DIR/train_neg.txt"
echo "mode: dummy_fact(+a)." > "$TEST_DIR/train_bk.txt"
echo "mode: target(+a)." >> "$TEST_DIR/train_bk.txt"
echo "pi_fact(a)." > "$TEST_DIR/train_facts_pi.txt"

echo "Running RDNBoost with -pi flag..."

# Run Java command
/home/nikhilesh/software/miniconda3/envs/nesy-il/bin/java -Xmx1G \
     -jar "$JAR" \
     -l \
     -train "$TEST_DIR" \
     -target "target" \
     -trees 1 \
     -aucJarPath "$AUC_JAR" \
     -pi \
     > test_pi_output.txt 2>&1 || true

# Check output
if grep -q "Loading privileged facts from" test_pi_output.txt; then
    echo "SUCCESS: Privileged facts loading message found."
else
    echo "FAILURE: Privileged facts loading message NOT found."
    cat test_pi_output.txt
    exit 1
fi

if grep -q "Privileged facts file not found" test_pi_output.txt; then
    echo "FAILURE: Privileged facts file not found message detected."
    exit 1
fi

echo "Test passed!"
