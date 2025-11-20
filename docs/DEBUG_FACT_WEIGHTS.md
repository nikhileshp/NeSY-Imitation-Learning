# Debugging Fact Weights and Grounding Penalty

## Quick Checklist

If grounding penalties are not being applied, check these in order:

### 1. ✅ fact_weights.txt File Exists

```bash
ls -la data/seaquest/single_t/54_RZ_2461867/fire/train/fact_weights.txt
```

Should show a file with size > 0 bytes.

### 2. ✅ -use-distance-weights Flag is Present

The Java command MUST include `-use-distance-weights`:

```bash
java -Dgrounding.penalty.threshold=0.7 \
     -Dgrounding.penalty.alpha=0.1 \
     -Dgrounding.penalty.beta=0.5 \
     -Dgrounding.penalty.strategy=min \
     -jar rdnboost/target/boostsrl-1.1.1.jar \
     -l \
     -train data/seaquest/single_t/54_RZ_2461867/fire/train \
     -target action \
     -trees 10 \
     -use-distance-weights \
     -negPosRatio 2 \
     -model models/test
```

**Common mistake**: Forgetting the `-use-distance-weights` flag causes fact_weights.txt to be ignored.

### 3. ✅ System Properties are Set

The grounding penalty parameters must be passed as Java system properties (`-D` flags) BEFORE the `-jar`:

```bash
java -Dgrounding.penalty.threshold=0.7 \    # ← Must come BEFORE -jar
     -Dgrounding.penalty.alpha=0.1 \
     -Dgrounding.penalty.beta=0.5 \
     -Dgrounding.penalty.strategy=min \
     -jar rdnboost/target/boostsrl-1.1.1.jar \
     -l \
     -train ...
```

### 4. ✅ Verify Loading in Output

Check for these messages in the console output:

```
% Distance weights enabled with XXXXX fact weights.
% Grounding penalty configured: threshold=0.7 alpha=0.1 beta=0.5 strategy=min
```

If you see:
```
% Distance weights flag set but weights not loaded. Using default weight 1.0.
```

This means the `-use-distance-weights` flag was set but `fact_weights.txt` was not found or failed to load.

## Detailed Debugging Steps

### Step 1: Check File Format

View the first few lines:

```bash
head -20 data/seaquest/single_t/54_RZ_2461867/fire/train/fact_weights.txt
```

Expected format:
```
facingleft(srz24618676426). 1.000
visibleenemy(srz24618676426,enemy1). 1.000
belowwatersurface(srz24618676426). 1.000
leftofenemy(srz24618676426,enemy1). 0.609
```

Each line should have:
- A fact (predicate with arguments)
- A period and space
- A weight value (0.0 to 1.0)

### Step 2: Test with Debug Mode

Run with debug mode to see if penalties are being computed:

```bash
./run_single_t_54_RZ.sh 3 1 true 2>&1 | grep -A5 "Grounding penalty"
```

You should see lines like:
```
%     Grounding penalty = 0.013456 for clause: action(State) :- near(State, submarine, fish)
```

If grounding penalty is always 0.0 or never appears, the weights aren't being loaded.

### Step 3: Check Java Classpath

Ensure you're using the correct JAR:

```bash
ls -lh rdnboost/target/boostsrl-1.1.1.jar
```

If the file doesn't exist or is old, rebuild:

```bash
cd rdnboost
mvn clean package
```

### Step 4: Verify Code Flow

Add this debug output to check if weights are loaded:

In `WILLSetup.java` after line 572, the console should print:
```
% Distance weights enabled with N fact weights.
```

Where N is the number of facts loaded from fact_weights.txt.

## Common Issues

### Issue 1: Flag Order Matters

**Wrong**:
```bash
java -jar boostsrl.jar -Dgrounding.penalty.threshold=0.7 ...  # ❌ -D flags after -jar won't work
```

**Correct**:
```bash
java -Dgrounding.penalty.threshold=0.7 -jar boostsrl.jar ...  # ✓ -D flags before -jar
```

### Issue 2: Missing -use-distance-weights

Even with -D flags, fact weights won't load without this flag:

```bash
java -Dgrounding.penalty.threshold=0.7 \
     -jar boostsrl.jar \
     -l \
     -train dir \
     # ❌ Missing: -use-distance-weights
```

### Issue 3: Wrong File Path

The code looks for `fact_weights.txt` in the training directory specified by `-train`:

```bash
-train data/seaquest/single_t/54_RZ_2461867/fire/train
```

The file must be at:
```
data/seaquest/single_t/54_RZ_2461867/fire/train/fact_weights.txt
```

Not in parent directories or other locations.

### Issue 4: File Permissions

Check that the file is readable:

```bash
ls -l data/seaquest/single_t/54_RZ_2461867/fire/train/fact_weights.txt
```

Should show read permissions (r--) for your user.

## Verification Commands

### Quick Test Command

```bash
cd /home/nikhilesh/Projects/NeSY-Imitation-Learning

java -Dgrounding.penalty.threshold=0.7 \
     -Dgrounding.penalty.alpha=0.1 \
     -Dgrounding.penalty.beta=0.5 \
     -Dgrounding.penalty.strategy=min \
     -jar rdnboost/target/boostsrl-1.1.1.jar \
     -l \
     -train data/seaquest/single_t/54_RZ_2461867/fire/train \
     -target action \
     -trees 1 \
     -aucJarPath rdnboost/src/edu/wisc/cs/will/DataSetUtils/ \
     -negPosRatio 2 \
     -model /tmp/test_model \
     -use-distance-weights 2>&1 | grep -E "(Distance weights|Grounding penalty)"
```

Expected output:
```
% Distance weights enabled with XXXXX fact weights.
% Grounding penalty configured: threshold=0.7 alpha=0.1 beta=0.5 strategy=min
```

### Check Penalty Values in Debug Mode

```bash
java -Dgrounding.penalty.threshold=0.7 \
     -Dgrounding.penalty.alpha=0.1 \
     -Dgrounding.penalty.beta=0.5 \
     -Dgrounding.penalty.strategy=min \
     -jar rdnboost/target/boostsrl-1.1.1.jar \
     -l \
     -train data/seaquest/single_t/54_RZ_2461867/fire/train \
     -target action \
     -trees 1 \
     -aucJarPath rdnboost/src/edu/wisc/cs/will/DataSetUtils/ \
     -negPosRatio 2 \
     -model /tmp/test_model \
     -use-distance-weights \
     -debugScoring 2>&1 | grep "Grounding penalty" | head -10
```

Should show non-zero grounding penalty values for various clauses.

## Success Indicators

When everything is working correctly, you'll see:

1. **Console output**:
   ```
   % Distance weights enabled with XXXXX fact weights.
   % Grounding penalty configured: threshold=0.7 alpha=0.1 beta=0.5 strategy=min
   ```

2. **Debug mode output** (if enabled):
   ```
   %     Score = -0.123456 (regressionFit = 0.100000, totalPenalty = 0.023456)
   %       Penalty breakdown:
   %         Length/Singleton = 0.010000
   %         Grounding        = 0.013456    ← Non-zero grounding penalty
   ```

3. **Node debug files** (if debug mode enabled):
   Files like `node_0_root.txt` in the model directory showing penalty information.

4. **Different learned trees**: Trees learned with grounding penalties will differ from those without, preferring clauses involving attended objects.
