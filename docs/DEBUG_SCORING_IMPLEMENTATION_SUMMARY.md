# Debug Scoring Implementation Summary

## What Was Implemented

A command-line argument `-debugScoring` that enables detailed debug output during RDN tree learning, showing example-by-example scoring with all mathematical steps, clause information, and example facts.

## Key Features

1. **Command-Line Controlled**: Enable/disable debug mode without recompiling
2. **Zero Performance Impact When Disabled**: Static boolean flag, no overhead when false
3. **Comprehensive Output**: Shows clause, example facts, contributions, running totals, and variance calculations
4. **Java 8 Compatible**: Uses StringBuilder instead of String.repeat() for portability

## How to Use

```bash
# Build the JAR
cd rdnboost && mvn clean package

# Train with debug output
java -jar rdnboost/target/BoostSRL-0.0.1-SNAPSHOT.jar \
    -l \
    -train data/seaquest/all/fire/train \
    -target fire \
    -trees 1 \
    -debugScoring
```

## Files Modified

### 1. CommandLineArguments.java
- Added `debugScoringFlag` constant and `enableDebugScoring` variable
- Added command-line parsing for `-debugScoring`
- Added getter/setter methods
- Added usage string documentation

**Location**: `rdnboost/src/edu/wisc/cs/will/Boosting/Utils/CommandLineArguments.java`

**Key Lines**:
- 253-254: Flag definition
- 850-856: Argument parsing  
- 915: Usage string
- 1343-1352: Getter/setter

### 2. BranchStats.java
- Added public static `ENABLE_DETAILED_DEBUG` flag
- Added debug context variables (branchName, currentClause, currentExample)
- Added setter methods for debug context
- Added detailed debug output in `addNumOutput()` method
- Added detailed debug output in `getWeightedVariance()` method
- Used Java 8 compatible StringBuilder for separator strings

**Location**: `rdnboost/src/edu/wisc/cs/will/ILP/Regression/BranchStats.java`

**Key Changes**:
- Line 8: Public static debug flag
- Lines 11-13: Debug tracking variables
- Lines 34-50: Setter methods
- Lines 54-137: Debug output in addNumOutput()
- Lines 197-220: Debug output in getWeightedVariance()

### 3. RegressionInfoHolderForRDN.java
- Set branch names ("TRUE" and "FALSE") in constructor
- Set current clause string before processing examples
- Set current example before calling addNumOutput()

**Location**: `rdnboost/src/edu/wisc/cs/will/ILP/Regression/RegressionInfoHolderForRDN.java`

**Key Changes**:
- Lines 25-26: Branch names in constructor
- Lines 115-117: Set clause on both branches
- Lines 100, 128: Set current example

### 4. RunBoostedRDN.java
- Set static BranchStats.ENABLE_DETAILED_DEBUG from command-line argument
- Print notification when debug mode is enabled

**Location**: `rdnboost/src/edu/wisc/cs/will/Boosting/RDN/RunBoostedRDN.java`

**Key Changes**:
- Lines 486-492: Set flag and print notification

### 5. RunBoostedModels.java  
- Set static BranchStats.ENABLE_DETAILED_DEBUG from command-line argument
- Print notification when debug mode is enabled

**Location**: `rdnboost/src/edu/wisc/cs/will/Boosting/Common/RunBoostedModels.java`

**Key Changes**:
- Lines 213-219: Set flag and print notification

## Architecture

### Data Flow

1. User provides `-debugScoring` on command line
2. `CommandLineArguments.parseArgs()` sets `enableDebugScoring = true`
3. In `main()`, static flag is set: `BranchStats.ENABLE_DETAILED_DEBUG = cmd.isEnableDebugScoring()`
4. During tree learning:
   - `RegressionInfoHolderForRDN` sets branch names, clause, and examples
   - `BranchStats.addNumOutput()` checks flag and prints debug output if enabled
   - `BranchStats.getWeightedVariance()` checks flag and prints final variance if enabled

### Design Decisions

1. **Static Flag**: Used a public static boolean in BranchStats for simplicity and zero overhead
2. **Setter Methods**: Added context setters (setBranchName, setCurrentClause, setCurrentExample) to pass information
3. **StringBuilder**: Used StringBuilder instead of String.repeat() for Java 8 compatibility
4. **Main Entry Points**: Set the flag in both RunBoostedRDN.main() and RunBoostedModels.main() to cover all use cases

## Testing

The implementation has been successfully compiled with:
- Java 8 (target version)
- Maven 3.x
- No compilation errors
- No new warnings introduced

To test functionality:

```bash
# Simple test with 1 tree
java -jar rdnboost/target/BoostSRL-0.0.1-SNAPSHOT.jar \
    -l \
    -train data/seaquest/all/fire/train \
    -target fire \
    -trees 1 \
    -depth 3 \
    -debugScoring \
    > test_debug_output.log 2>&1

# Verify output
grep "CLAUSE BEING EVALUATED" test_debug_output.log
grep "CURRENT WEIGHTED VARIANCE" test_debug_output.log
```

## Documentation Created

1. **DEBUG_SCORING_ARG_README.md** - Comprehensive usage guide
   - Usage examples
   - Output format explanation
   - Troubleshooting tips
   - Performance considerations

2. **DEBUG_SCORING_IMPLEMENTATION_SUMMARY.md** (this file)
   - Implementation overview
   - Files modified with line numbers
   - Architecture and design decisions

## Benefits

1. **No Recompilation**: Toggle debug mode via command-line argument
2. **No Performance Cost**: When disabled, zero overhead (single boolean check)
3. **Complete Information**: Shows clause, example facts, and all mathematical steps
4. **Easy to Use**: Simple flag, works with existing commands
5. **Portable**: Java 8 compatible

## Future Enhancements (Optional)

Potential improvements that could be added:

1. **Granular Control**: Flags like `-debugScoringLevel <level>` for different verbosity
2. **File Output**: Automatic logging to file instead of stdout
3. **Filtering**: Debug only specific predicates or examples
4. **Statistics**: Summary statistics at the end of training
5. **Visualization**: Generate graphs from debug output

## Limitations

1. **Only Learning Mode**: Debug output only appears during training (`-l`), not inference (`-i`)
2. **Performance**: Extensive output slows down training significantly
3. **Console Flood**: Output can be overwhelming for large datasets
4. **Single Branch Type**: Currently only implemented for RegressionInfoHolderForRDN

## Related Work

This implementation builds on:
- RDN_SCORING_DETAILED.md - Mathematical documentation
- Previous hardcoded debug versions
- BoostSRL's existing command-line argument infrastructure

## Completion Status

✅ Command-line argument parsing
✅ Static flag implementation
✅ Debug output in BranchStats
✅ Context passing (branch name, clause, example)
✅ Java 8 compatibility
✅ Build successful
✅ Documentation complete
✅ Zero impact when disabled

## How to Revert

If you need to remove this feature:

1. Remove the `-debugScoring` flag from CommandLineArguments.java
2. Remove debug output code from BranchStats.java
3. Remove context setters from BranchStats.java
4. Remove context setting calls from RegressionInfoHolderForRDN.java
5. Remove flag setting from RunBoostedRDN.java and RunBoostedModels.java
6. Rebuild: `cd rdnboost && mvn clean package`

Or simply never use the `-debugScoring` flag (zero impact).

## Contact

For questions or issues with this implementation, refer to:
- DEBUG_SCORING_ARG_README.md for usage
- RDN_SCORING_DETAILED.md for mathematical background
- Code comments in modified files for implementation details
