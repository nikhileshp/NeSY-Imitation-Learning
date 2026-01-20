# Neurosymbolic Imitation Learning with Human Eye-Tracking for Atari Games

This project implements a Neurosymbolic Imitation Learning approach for Atari games, specifically Seaquest, leveraging human eye-tracking data to improve agent performance and interpretability.

## Prerequisites

- Python 3.8+
- Conda (Anaconda or Miniconda)

## Installation

To set up the environment, you can use the following commands. This will create a new Conda environment and install all necessary dependencies.

```bash
# 1. Create a new Conda environment
conda create -n nesy-il python=3.9 -y

# 2. Activate the environment
conda activate nesy-il

# 3. Install dependencies
pip install -r requirements.txt
```

## Usage

### Training Per Action

To train a model for a specific action (e.g., "fire"), use the `train_per_action.py` script. This script trains both an RGB-only model and an RGB + Gaze model.

```bash
python train_per_action.py --action fire --model_type resnet18 --epochs 50
```

**Arguments:**
- `--action`: The action to train for (e.g., `fire`, `up`, `down`, `left`, `right`, `noop`).
- `--model_type`: The model architecture to use (`resnet18` or `cnn`). Default: `resnet18`.
- `--ratio`: The ratio of negative to positive samples. Default: `2.0`.
- `--epochs`: Number of training epochs. Default: `50`.
- `--seed`: Random seed. Default: `42`.

### Training Imitation Learning

To train the full imitation learning model, use the `train_imitation.py` script.

```bash
python train_imitation.py
```

This script will train models to predict actions based on game frames and gaze data.

## Project Structure

- `requirements.txt`: List of Python dependencies.
- `data/`: Directory containing game data and gaze recordings.
- `trained_models/`: Directory where trained models and results are saved.
