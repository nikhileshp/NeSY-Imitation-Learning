# Debug Scoring Command-Line Argument

## Overview

A new command-line argument `-debugScoring` has been added to the BoostSRL/RDN learning system to enable detailed debug output during tree learning. This allows you to see exactly how the scoring function evaluates splits example-by-example.

## Building the JAR with Debug Support

```bash
cd rdnboost
mvn clean package
```

This creates: `rdnboost/target/BoostSRL-0.0.1-SNAPSHOT.jar`

## Usage

### Basic Syntax

```bash
java -jar rdnboost/target/BoostSRL-0.0.1-SNAPSHOT.jar -l -train <train_dir> ... -debugScoring
```

### With Pipeline Script

If you're using the pipeline script, you need to modify `run_full_pipeline.sh` to add the debug flag to the java command.

### Example: Training with Debug Mode

```bash
# Training mode with debug output
java -jar rdnboost/target/BoostSRL-0.0.1-SNAPSHOT.jar \
    -l \
    -train data/seaquest/all/fire/train \
    -target fire \
    -trees 1 \
    -debugScoring
```

### Example: Inference Mode (No Debug Needed)

```bash
# Inference doesn't use the scoring function, so -debugScoring has no effect
java -jar rdnboost/target/BoostSRL-0.0.1-SNAPSHOT.jar \
    -i \
    -test data/seaquest/all/fire/test \
    -model rdn_models/seaquest/... \
    -target fire
```

## What the Debug Output Shows

When `-debugScoring` is enabled during training (`-l` mode), you'll see:

### 1. For Each Example Processed:
```
================================================================================
[TRUE BRANCH] Adding example to branch
================================================================================

CLAUSE BEING EVALUATED:
fire(State) :- oxygen(State,O), O < 50.

EXAMPLE FACTS:
fire(state_42).
submarine(state_42, 120, 80).
oxygen(state_42, 35).
...

CONTRIBUTIONS FROM THIS EXAMPLE:
  num                = 1
  output             = 0.5
  weight             = 1.0
  prob               = 0.9

COMPUTED VALUES:
  num^2 * weight                    = 1.0
  num * output * weight             = 0.5
  output^2 * weight                 = 0.25
  num^2 * weight * deno             = 0.09

RUNNING TOTALS BEFORE ADDING:
  sumOfNumGroundingSquared          = 10.0
  sumOfOutputAndNumGrounding        = 5.5
  sumOfOutputSquared                = 3.25
  numExamples                       = 10.0
  sumOfNumGroundingSquaredWithProb  = 1.2

RUNNING TOTALS AFTER ADDING:
  sumOfNumGroundingSquared          = 11.0
  sumOfOutputAndNumGrounding        = 6.0
  sumOfOutputSquared                = 3.5
  numExamples                       = 11.0
  sumOfNumGroundingSquaredWithProb  = 1.29

CURRENT WEIGHTED VARIANCE:
  Formula: sumOfOutputSquared - (sumOfOutputAndNumGrounding^2 / sumOfNumGroundingSquared)
         = 3.5 - (6.0^2 / 11.0)
         = 3.5 - (36.0 / 11.0)
         = 3.5 - 3.27
         = 0.23
================================================================================
```

### 2. Final Branch Statistics:
```
================================================================================
[TRUE BRANCH] FINAL WEIGHTED VARIANCE CALCULATION
================================================================================

FINAL BRANCH STATISTICS:
  sumOfOutputSquared               = 3.5
  sumOfOutputAndNumGrounding       = 6.0
  sumOfNumGroundingSquared         = 11.0
  numExamples                      = 11.0

FINAL VARIANCE CALCULATION:
  Formula: sumOfOutputSquared - (sumOfOutputAndNumGrounding^2 / sumOfNumGroundingSquared)
         = 3.5 - (6.0^2 / 11.0)
         = 3.5 - (36.0 / 11.0)
         = 3.5 - 3.27
         = 0.23
================================================================================
```

## Implementation Details

### Files Modified

1. **CommandLineArguments.java** - Added `-debugScoring` flag parsing
   - Lines 253-254: Flag definition
   - Lines 850-856: Argument parsing
   - Line 915: Usage string
   - Lines 1343-1352: Getter/setter methods

2. **BranchStats.java** - Added debug output logic
   - Line 8: Public static flag `ENABLE_DETAILED_DEBUG`
   - Lines 11-13: Debug tracking variables
   - Lines 34-50: Setter methods for branch name, clause, and example
   - Lines 54-137: Debug output in `addNumOutput()`
   - Lines 197-220: Debug output in `getWeightedVariance()`

3. **RegressionInfoHolderForRDN.java** - Set debug context
   - Lines 25-26: Set branch names in constructor
   - Lines 115-117: Set clause string
   - Lines 100, 128: Set current example

4. **RunBoostedRDN.java** - Enable debug flag from command-line
   - Lines 486-492: Set static flag based on command-line argument

5. **RunBoostedModels.java** - Enable debug flag from command-line
   - Lines 213-219: Set static flag based on command-line argument

### How It Works

1. User provides `-debugScoring` argument on command line
2. `CommandLineArguments.parseArgs()` sets `enableDebugScoring = true`
3. In `main()`, the static flag is set: `BranchStats.ENABLE_DETAILED_DEBUG = cmd.isEnableDebugScoring()`
4. During tree learning, when `BranchStats.addNumOutput()` is called:
   - If `ENABLE_DETAILED_DEBUG` is true, detailed output is printed
   - Shows the clause being evaluated
   - Shows the facts for the current example
   - Shows all intermediate calculations
5. Same for `getWeightedVariance()` - shows final variance calculation

## Disabling Debug Mode

To disable debug mode, simply omit the `-debugScoring` flag:

```bash
java -jar rdnboost/target/BoostSRL-0.0.1-SNAPSHOT.jar -l -train ... -target fire
```

Or explicitly set it to false:

```bash
java -jar rdnboost/target/BoostSRL-0.0.1-SNAPSHOT.jar -l -train ... -debugScoring false
```

## Performance Impact

- **Debug OFF**: No performance impact (default behavior)
- **Debug ON**: Significant slowdown due to extensive console output
  - Recommended only for debugging small datasets or specific trees
  - Consider redirecting output to a file: `java ... -debugScoring > debug.log 2>&1`

## Tips

1. **Use with minimal trees**: Start with `-trees 1` to see output for just one tree
2. **Redirect output**: Debug output is verbose, save to file for analysis
3. **Filter output**: Use `grep` or similar tools to find specific patterns
4. **Combine with small dataset**: Debug on a subset of your data first

## Example Workflow

```bash
# 1. Build with debug support
cd rdnboost && mvn clean package

# 2. Run training with debug on one tree, save output
java -jar rdnboost/target/BoostSRL-0.0.1-SNAPSHOT.jar \
    -l \
    -train data/seaquest/all/fire/train \
    -target fire \
    -trees 1 \
    -depth 3 \
    -debugScoring \
    > debug_output.log 2>&1

# 3. Analyze specific examples
grep "EXAMPLE FACTS" debug_output.log | less

# 4. Check variance calculations
grep "CURRENT WEIGHTED VARIANCE" debug_output.log | less
```

## Troubleshooting

### Debug output not showing
- Ensure you're running in **learning mode** (`-l` flag), not inference mode (`-i`)
- Check that you've rebuilt the JAR after code changes
- Verify the correct JAR is being used

### Output is too verbose
- Reduce number of trees: `-trees 1`
- Use smaller training set
- Redirect to file and analyze offline
- Filter with grep for specific patterns

### Build fails
- Ensure Java 8 or later: `java -version`
- Clean build: `mvn clean package`
- Check Maven is installed: `mvn -version`

## Related Documentation

- `RDN_SCORING_DETAILED.md` - Mathematical explanation of scoring
- `DEBUG_JAR_BUILD_INSTRUCTIONS.md` - Original debug instructions
- `CODE_CHANGES_SUMMARY.md` - Complete list of code changes

## Command-Line Arguments Summary

```
-debugScoring : Enable detailed debug output for RDN tree scoring 
                (shows clause, examples, and step-by-step variance calculations).
```

This flag is compatible with all other command-line arguments and can be used with:
- `-l` (learn mode) - **This is where debug output appears**
- `-train <dir>` - Training directory
- `-trees <n>` - Number of trees
- `-depth <n>` - Max tree depth  
- And all other standard BoostSRL flags
