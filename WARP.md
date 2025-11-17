# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Project Overview

This is a **Neurosymbolic Imitation Learning** system that uses eye-tracking data and object detection to learn action policies for Atari games (currently focused on Seaquest). The system combines:
- **Computer vision**: Object detection and relationship analysis for game frames
- **Gaze analysis**: Eye-tracking data to compute attention weights
- **Relational learning**: Relational Dependency Networks (RDNs) using BoostSRL
- **Calibration**: Logistic regression to combine independent action classifiers

## Common Commands

### Full Pipeline (Training + Testing + Evaluation)
```bash
# Train and evaluate unweighted model
./run_full_pipeline.sh 3 10 false false

# Train and evaluate weighted model (uses eye-tracking attention weights)
./run_full_pipeline.sh 3 10 true false

# Test only (skip training, useful when models already exist)
./run_full_pipeline.sh 3 10 false true
```

**Parameters**: `<max_depth> <num_trees> <weighted> <only_test>`
- `max_depth`: RDN tree depth (typically 3-8)
- `num_trees`: Number of boosting trees (typically 10-20)
- `weighted`: Whether to use per-example attention weights from eye-tracking
- `only_test`: Skip training and only run inference + evaluation

### Building the Java Component (BoostSRL)
```bash
cd rdnboost
mvn clean package
```
This creates `rdnboost/target/boostsrl-weights-2.0.0.jar` used by the pipeline.

### Data Processing
```bash
# Process a single trajectory with visualization
python main.py --image_folder <path/to/images> --fps 1 --verbose 2

# Process without visualization
python main.py --image_folder <path/to/images> --no_visual --process_all --verbose 1
```

### Visualization
```bash
# Visualize RDN decision trees and results
python visualize_rdn_results.py --model_dir <model_path>

# Standalone visualization of processed data
python standalone_visualization.py
```

### Monitoring Progress
```bash
# Check progress during long training runs
./check_calibration_progress.sh
```

## Architecture

### High-Level Flow

1. **Object Detection & Relationship Extraction** (Python)
   - Process game frames to detect objects (submarine, divers, fish, oxygen, etc.)
   - Compute spatial relationships (above, below, left, right, near)
   - Integrate eye-tracking data to compute attention weights

2. **RDN Training** (Java/BoostSRL)
   - Train 6 independent binary classifiers (one per action: fire, up, down, left, right, noop)
   - Generate logical rules in first-order logic
   - Optionally use per-example weights from attention

3. **Inference** (Java/BoostSRL)
   - Run on both training data (for calibration) and test data
   - Output probabilities for each action classifier

4. **Calibration & Evaluation** (Python)
   - Train logistic regression to combine the 6 independent classifiers
   - Compare calibrated vs. non-calibrated predictions

### Code Organization

```
.
├── main.py                      # Main entry point for data processing
├── run_full_pipeline.sh         # Complete training/testing/evaluation pipeline
├── eval_calibrated.py           # Calibration and evaluation logic
├── change_bk.py                 # Utility to update RDN max_depth parameter
├── attention_weights.py         # Eye-tracking attention weight calculation
│
├── core/                        # Core processing modules
│   ├── gaze_data_processor.py  # Load and process eye-tracking data
│   ├── goal_detector.py        # Detect goal conditions in game
│   ├── distance_weight_calculator.py  # Attention weight strategies
│   ├── visualization_manager.py # Visualization utilities
│   ├── game_object.py          # GameObject class definition
│   └── relationship_analyzer.py # Base relationship analysis
│
├── env/seaquest/               # Game-specific modules (extensible to other games)
│   ├── object_detector.py      # Seaquest-specific object detection
│   ├── relationship_analyzer.py # Seaquest-specific relationships
│   └── config.py               # Seaquest configuration (colors, thresholds)
│
├── data/seaquest/all/          # Training and test data
│   ├── fire/train/             # Training data for 'fire' action
│   ├── fire/test/              # Test data for 'fire' action
│   └── [up/down/left/right/noop]/ # Other actions
│
├── rdnboost/                   # Modified BoostSRL for weighted training
│   ├── src/                    # Java source code
│   ├── pom.xml                 # Maven build configuration
│   └── target/                 # Built JAR files
│
├── rdn_models/seaquest/        # Trained models and results
│   └── negpos_2_trees_X_depth_Y_*/  # Model directories
│       ├── fire/fire_test_infer.log
│       ├── up/up_test_infer.log
│       ├── [other actions]/
│       └── eval_report.txt     # Final evaluation results
│
└── docs/                       # Documentation
    ├── PIPELINE_README.md      # Pipeline documentation
    ├── CALIBRATION_README.md   # Calibration explanation
    └── [other docs]
```

### Key Architectural Concepts

#### 1. Per-Example Attention Weights
Eye-tracking data is used to compute attention weights for training examples using a Gaussian function based on distance from gaze point to object centroids. This allows the RDN to prioritize examples the player was attending to.

#### 2. Independent Action Classifiers
Rather than multi-class classification, 6 binary classifiers are trained independently. This is why calibration is needed - their probability scales aren't directly comparable.

#### 3. RDN (Relational Dependency Networks)
First-order logic rules are learned to predict actions based on:
- Object types and positions
- Spatial relationships between objects
- Object properties (e.g., oxygen level)

Example rule:
```prolog
action(State) :- oxygen(State,O), O < 50, near(State, submarine, oxygen_icon).
```

#### 4. Two-Stage Evaluation
- **Method 1**: Direct argmax of raw probabilities from 6 classifiers
- **Method 2**: Logistic regression calibration that learns inter-action dependencies

The calibrator is trained on training set predictions, then applied to test set.

## Data Format

### Input Data Structure
- **Images**: Game frames in PNG/JPG format (e.g., `frame_0.png`, `frame_1.png`)
- **Gaze Data**: Text file with same name as image folder + `.txt` extension
  - Tab-separated: frame index, x-coordinate, y-coordinate
  - Updated with relationship data after processing

### RDN Data Format (BoostSRL)
- `train_facts.txt`: Ground atoms describing the state
- `train_pos.txt`: Positive examples of the target predicate
- `train_neg.txt`: Negative examples of the target predicate
- `bk.pl`: Background knowledge and mode declarations
- `train_pos_weights.txt`: Per-example weights (when using weighted training)
- `train_neg_weights.txt`: Per-example weights for negatives

## Important Notes

### Weight File Management
The pipeline automatically manages weight files:
- **Weighted training**: Ensures `train_pos_weights.txt` and `train_neg_weights.txt` are present
- **Unweighted training**: Moves weight files to `.bak` backups
- Files are automatically restored when switching modes

### Model Naming Convention
Models are saved as: `negpos_<ratio>_trees_<num>_depth_<depth>_[per_example_weight_]all`
- Example: `negpos_2_trees_10_depth_3_per_example_weight_all` (weighted, 10 trees, depth 3)
- Example: `negpos_2_trees_10_depth_3_new_all` (unweighted, 10 trees, depth 3)

### Test Commands
There are no automated unit tests. Testing is done by:
1. Running full pipeline on small datasets
2. Checking evaluation metrics in `eval_report.txt`
3. Visualizing learned trees: `rdn_models/.../WILLtheories/action_learnedWILLregressionTrees.txt`

### Performance
Full pipeline timing (typical):
- Training: 10-30 min per action (60-180 min total)
- Test inference: 2-5 min per action (12-30 min total)
- Train inference: 5-15 min per action (30-90 min total)
- Calibration: < 1 min
- **Total**: 2-4 hours

## Dependencies

### Python
```bash
pip install opencv-python pandas numpy scikit-learn tqdm
```

### Java
- Java 8+ (tested with OpenJDK 1.8.0)
- Maven (for building BoostSRL)

### Required External Model
The object detector depends on models in `models/OC_Atari/` (included via requirements.txt reference).

## Extending to New Games

To add support for a new Atari game:
1. Create `env/<game_name>/` directory
2. Implement `<game_name>ObjectDetector` (inherit from base or use OCAtari models)
3. Implement `<game_name>RelationshipAnalyzer` with game-specific relationships
4. Create `<game_name>Config` with colors and detection parameters
5. Update `main.py` to recognize the new game type

## Common Issues

### Java Heap Space
If RDN training fails with OutOfMemoryError, increase heap:
```bash
# Edit run_full_pipeline.sh, modify java command:
java -Xmx8g -jar "$JAR" ...  # Increases heap to 8GB
```

### Missing AUC Files
If `AUC/aucTemp.txt` is not generated:
- Check that `query_*.db` files are correctly formatted
- Verify model directory exists and contains trained trees
- Check inference logs in `rdn_models/.../action_*_infer.log` for errors

### Calibration Fails
If `eval_calibrated.py` fails:
- Ensure both training and test inference completed
- Check that `train.csv` and `test.csv` exist with correct state_id mappings
- Verify train_infer directories contain `AUC/aucTemp.txt`
