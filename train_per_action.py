import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import os
import matplotlib.pyplot as plt
import math
from sklearn.metrics import f1_score, average_precision_score, accuracy_score, roc_auc_score, precision_score, recall_score
import argparse
import re
import json

# Configuration
# Using the root train.csv as the source of truth for images/gaze
CSV_FILE = "/home/nikhilesh/Projects/NeSY-Imitation-Learning/train.csv"
# Image directory - assuming images are in the same folder structure as before or we need to find them
# Based on previous exploration, images seem to be in data/seaquest/gaze_data_tmp/...
# But train.csv has a 'trajectory' column which might point to the folder.
# Let's assume a base image dir and try to find them.
# For now, I'll use the one from the previous script as a fallback or try to derive it.
# Actually, the previous script used a specific folder. 
# Let's look at train.csv 'trajectory' column.
# Example: 241_RZ_19306_Feb-12-18-57-06
# The images should be in data/seaquest/gaze_data_tmp/{trajectory}/
BASE_IMAGE_DIR = "/home/nikhilesh/Projects/NeSY-Imitation-Learning/data/seaquest/gaze_data_tmp"

BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001
SEED = 42

# Action mapping
ACTION_NAMES = {
    0: "noop",
    1: "fire",
    2: "up",
    3: "right",
    4: "left",
    5: "down"
}

def get_rdn_id(frame_id):
    """Convert CSV frame_id (RZ_...) to RDN ID (srz...)."""
    return "s" + frame_id.replace("_", "").lower()

def load_split_ids(action, split="train"):
    """Load positive and negative RDN IDs from text files."""
    base_path = f"data/seaquest/all/{action}/{split}"
    pos_file = os.path.join(base_path, f"{split}_pos.txt")
    neg_file = os.path.join(base_path, f"{split}_neg.txt")
    
    pos_ids = set()
    neg_ids = set()
    
    print(f"Loading split IDs from {base_path}...")
    
    if os.path.exists(pos_file):
        with open(pos_file, 'r') as f:
            for line in f:
                match = re.search(r'action\((.*?),\s*.*?\)\.', line)
                if match:
                    pos_ids.add(match.group(1))
    
    if os.path.exists(neg_file):
        with open(neg_file, 'r') as f:
            for line in f:
                match = re.search(r'action\((.*?),\s*.*?\)\.', line)
                if match:
                    neg_ids.add(match.group(1))
                    
    return pos_ids, neg_ids

def process_gaze(gaze_str):
    """Process gaze string from CSV."""
    try:
        # Gaze is stored as "[(x, y), ...]" string
        # We need to parse it. 
        # Simple regex or eval
        # It looks like a list of tuples
        import ast
        points = ast.literal_eval(gaze_str)
        if not points:
            return (160/2, 210/2)
        
        # Strategy: Farthest from previous? 
        # For simplicity/speed in this script, let's just take the average or last point
        # The previous script used "farthest from previous", which requires state.
        # Let's use the mean point for stability if multiple points exist.
        # Or just the first one.
        return points[-1] # Use last point
    except:
        return (160/2, 210/2)

class SeaquestDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform
                
    def _load_image(self, idx):
        row = self.df.iloc[idx]
        traj_folder = row['trajectory']
        img_name = f"{row['frameid']}.png"
        img_path = os.path.join(BASE_IMAGE_DIR, traj_folder, img_name)
        
        try:
            image = Image.open(img_path).convert('RGB')
        except FileNotFoundError:
            image = Image.new('RGB', (210, 160))
            
        return image

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        image = self._load_image(idx)
            
        if self.transform:
            image = self.transform(image)
            
        row = self.df.iloc[idx]
        # Parse gaze
        gaze_point = process_gaze(row['gaze_positions'])
        gaze = torch.tensor([gaze_point[0] / 160.0, gaze_point[1] / 210.0], dtype=torch.float32)
        
        label = torch.tensor(row['label'], dtype=torch.float32)
        
        return image, gaze, label

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )
        
        # Calculate output size
        # 84x84 -> 20x20 -> 9x9 -> 7x7
        # 64 * 7 * 7 = 3136
        # 224x224 -> 55x55 -> 26x26 -> 24x24
        # 64 * 24 * 24 = 36864
        
        self.fc_input_dim = 0 # Will be set dynamically or hardcoded if we know input size
        
    def forward(self, x, gaze=None):
        x = self.features(x)
        return x

class RGBModel(nn.Module):
    def __init__(self, model_type='resnet18', img_size=224):
        super(RGBModel, self).__init__()
        
        if model_type == 'resnet18':
            self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            num_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
            self.feature_dim = num_features
        else: # cnn
            self.backbone = SimpleCNN()
            # Calculate feature dim based on img_size
            # For 84x84: 3136
            # For 224x224: 36864
            if img_size == 224:
                self.feature_dim = 36864
            else:
                self.feature_dim = 3136
        
        self.fc = nn.Sequential(
            nn.Linear(self.feature_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
    def forward(self, x, gaze=None):
        x = self.backbone(x)
        x = self.fc(x)
        return x

class RGBGazeModel(nn.Module):
    def __init__(self, model_type='resnet18', img_size=224):
        super(RGBGazeModel, self).__init__()
        
        if model_type == 'resnet18':
            self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            num_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
            self.feature_dim = num_features
        else: # cnn
            self.backbone = SimpleCNN()
            if img_size == 224:
                self.feature_dim = 36864
            else:
                self.feature_dim = 3136
        
        self.fc = nn.Sequential(
            nn.Linear(self.feature_dim + 2, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
    def forward(self, x, gaze):
        x = self.backbone(x)
        x = torch.cat((x, gaze), dim=1)
        x = self.fc(x)
        return x

def train_model(model, train_loader, val_loader, device, model_name="Model", action_name="unknown"):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    print(f"Training {model_name} for action {action_name}...")
    
    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0
    
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        for i, (images, gazes, labels) in enumerate(train_loader):
            images, gazes, labels = images.to(device), gazes.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images, gazes).squeeze(1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
            if (i + 1) % 10 == 0:
                print(f"[{action_name}] Epoch [{epoch+1}/{EPOCHS}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")
            
            if MAX_STEPS and (i + 1) >= MAX_STEPS:
                print(f"[{action_name}] Reached max steps {MAX_STEPS}, stopping epoch.")
                break
            
        avg_train_loss = running_loss/len(train_loader)
        print(f"[{action_name}] Epoch {epoch+1}/{EPOCHS} Average Loss: {avg_train_loss:.4f}")
        
        # Validation for Early Stopping
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, gazes, labels in val_loader:
                images, gazes, labels = images.to(device), gazes.to(device), labels.to(device)
                outputs = model(images, gazes).squeeze(1)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        print(f"[{action_name}] Val Loss: {avg_val_loss:.4f}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"[{action_name}] Early stopping triggered after {epoch+1} epochs.")
                break
            
    # Final Evaluation
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for i, (images, gazes, labels) in enumerate(val_loader):
            images, gazes, labels = images.to(device), gazes.to(device), labels.to(device)
            outputs = model(images, gazes).squeeze(1)
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            
            if MAX_STEPS and (i + 1) >= MAX_STEPS:
                print(f"Reached max eval steps {MAX_STEPS}, stopping eval.")
                break
            
    f1 = f1_score(all_labels, all_preds)
    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    
    try:
        auc_pr = average_precision_score(all_labels, all_probs)
    except:
        auc_pr = 0.0
        
    try:
        auc_roc = roc_auc_score(all_labels, all_probs)
    except:
        auc_roc = 0.0
        
    metrics = {
        "f1": f1,
        "auc_pr": auc_pr,
        "auc_roc": auc_roc,
        "accuracy": acc,
        "precision": precision,
        "recall": recall
    }
        
    return metrics, model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--action", type=str, required=True, help="Action name (e.g., fire)")
    parser.add_argument("--model_type", type=str, default="resnet18", choices=["resnet18", "cnn"], help="Model architecture")
    parser.add_argument("--ratio", type=float, default=2.0, help="Negative to positive ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--max_steps", type=int, default=None, help="Max steps per epoch")
    args = parser.parse_args()
    
    global EPOCHS, MAX_STEPS
    EPOCHS = args.epochs
    MAX_STEPS = args.max_steps
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Action: {args.action}, Model: {args.model_type}, Ratio: {args.ratio}")
    
    # Load full CSV
    print("Loading train.csv and test.csv...")
    train_df_full = pd.read_csv(CSV_FILE)
    test_csv_file = CSV_FILE.replace("train.csv", "test.csv")
    test_df_full = pd.read_csv(test_csv_file)
    df = pd.concat([train_df_full, test_df_full], ignore_index=True)
    
    # Create RDN ID column
    df['rdn_id'] = df['frameid'].apply(lambda x: "s" + str(x).replace("_", "").lower())
    
    # Load split IDs
    train_pos_ids, train_neg_ids = load_split_ids(args.action, "train")
    test_pos_ids, test_neg_ids = load_split_ids(args.action, "test")
    
    print(f"Found {len(train_pos_ids)} pos, {len(train_neg_ids)} neg training IDs")
    print(f"Found {len(test_pos_ids)} pos, {len(test_neg_ids)} neg test IDs")
    
    # Filter DataFrames
    train_pos_df = df[df['rdn_id'].isin(train_pos_ids)].copy()
    train_pos_df['label'] = 1
    train_neg_df = df[df['rdn_id'].isin(train_neg_ids)].copy()
    train_neg_df['label'] = 0
    
    test_pos_df = df[df['rdn_id'].isin(test_pos_ids)].copy()
    test_pos_df['label'] = 1
    test_neg_df = df[df['rdn_id'].isin(test_neg_ids)].copy()
    test_neg_df['label'] = 0
    
    # Downsample Negatives (Training)
    n_pos = len(train_pos_df)
    n_neg_keep = int(n_pos * args.ratio)
    if len(train_neg_df) > n_neg_keep:
        train_neg_df = train_neg_df.sample(n=n_neg_keep, random_state=args.seed)
        print(f"Downsampled training negatives to {n_neg_keep}")
        
    # Downsample Negatives (Testing) - usually we keep all or same ratio?
    # cloning_rdn.py downsamples test negatives too
    n_test_pos = len(test_pos_df)
    n_test_neg_keep = int(n_test_pos * args.ratio)
    if len(test_neg_df) > n_test_neg_keep:
        test_neg_df = test_neg_df.sample(n=n_test_neg_keep, random_state=args.seed)
        print(f"Downsampled test negatives to {n_test_neg_keep}")
        
    train_df = pd.concat([train_pos_df, train_neg_df])
    test_df = pd.concat([test_pos_df, test_neg_df])
    
    print(f"Train set: {len(train_df)} samples")
    print(f"Test set: {len(test_df)} samples")
    
    # Transforms
    img_size = 224 if args.model_type == 'resnet18' else 84
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])
    
    train_dataset = SeaquestDataset(train_df, transform=transform)
    test_dataset = SeaquestDataset(test_df, transform=transform)
    
    # Use num_workers for faster data loading
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    
    # Define Output Directory
    # trained_models/seaquest/all/negpos_2_{model_type}_64_32_bc/{action}/seed_{seed}/
    model_name_str = f"rgb_{args.model_type}_64_32_bc"
    if args.model_type == 'cnn':
        model_name_str = "rgb_cnn_3_layers_64_32_bc"
        
    output_dir = f"trained_models/seaquest/all/negpos_{int(args.ratio)}_{model_name_str}/{args.action}/seed_{args.seed}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Train RGB Model
    rgb_model = RGBModel(model_type=args.model_type, img_size=img_size).to(device)
    rgb_metrics, rgb_model = train_model(rgb_model, train_loader, test_loader, device, f"RGB Model ({args.model_type})", args.action)
    
    # Save RGB Model
    torch.save(rgb_model.state_dict(), os.path.join(output_dir, "model_rgb.pth"))
    
    # Train RGB + Gaze Model
    gaze_model = RGBGazeModel(model_type=args.model_type, img_size=img_size).to(device)
    gaze_metrics, gaze_model = train_model(gaze_model, train_loader, test_loader, device, f"Gaze Model ({args.model_type})", args.action)
    
    # Save Gaze Model
    torch.save(gaze_model.state_dict(), os.path.join(output_dir, "model_gaze.pth"))
    
    print(f"\nResults for {args.action}:")
    print(f"RGB  - F1: {rgb_metrics['f1']:.4f}, AUC-PR: {rgb_metrics['auc_pr']:.4f}")
    print(f"Gaze - F1: {gaze_metrics['f1']:.4f}, AUC-PR: {gaze_metrics['auc_pr']:.4f}")
    
    # Save Log
    log_file = os.path.join(output_dir, "test_infer.log")
    with open(log_file, "w") as f:
        f.write(f"Results for {args.action} (Seed {args.seed}, Ratio {args.ratio}):\n")
        f.write(f"RGB F1: {rgb_metrics['f1']:.4f}\n")
        f.write(f"RGB AUC PR: {rgb_metrics['auc_pr']:.4f}\n")
        f.write(f"RGB AUC ROC: {rgb_metrics['auc_roc']:.4f}\n")
        f.write(f"RGB Accuracy: {rgb_metrics['accuracy']:.4f}\n")
        f.write(f"RGB Precision: {rgb_metrics['precision']:.4f}\n")
        f.write(f"RGB Recall: {rgb_metrics['recall']:.4f}\n")
        
        f.write(f"Gaze F1: {gaze_metrics['f1']:.4f}\n")
        f.write(f"Gaze AUC PR: {gaze_metrics['auc_pr']:.4f}\n")
        f.write(f"Gaze AUC ROC: {gaze_metrics['auc_roc']:.4f}\n")
        f.write(f"Gaze Accuracy: {gaze_metrics['accuracy']:.4f}\n")
        f.write(f"Gaze Precision: {gaze_metrics['precision']:.4f}\n")
        f.write(f"Gaze Recall: {gaze_metrics['recall']:.4f}\n")
        
    # Save JSON
    results = {
        "action": args.action,
        "rgb_f1": rgb_metrics['f1'],
        "rgb_auc": rgb_metrics['auc_pr'],
        "gaze_f1": gaze_metrics['f1'],
        "gaze_auc": gaze_metrics['auc_pr']
    }
    with open(os.path.join(output_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=4)
        
    print(f"Saved results to {output_dir}")

if __name__ == "__main__":
    main()
