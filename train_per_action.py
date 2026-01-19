import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import os
import matplotlib.pyplot as plt
import math
from sklearn.metrics import f1_score, average_precision_score
from sklearn.preprocessing import label_binarize

# Configuration
IMAGE_DIR = "/home/nikhilesh/Projects/NeSY-Imitation-Learning/data/seaquest/gaze_data_tmp/54_RZ_2461867_Aug-11-09-35-18"
CSV_FILE = "/home/nikhilesh/Projects/NeSY-Imitation-Learning/data/seaquest/54_RZ_2461867_Aug-11-09-35-18.txt"
BATCH_SIZE = 32
EPOCHS = 2
LEARNING_RATE = 0.001
IMG_HEIGHT = 84
IMG_WIDTH = 84
TRAIN_SPLIT = 0.8
SEED = 42
ACTIONS = [0, 1, 2, 3, 4, 5]

# Set seed
torch.manual_seed(SEED)
np.random.seed(SEED)

def load_and_process_data():
    print("Loading data...")
    data = []
    with open(CSV_FILE, 'r') as f:
        header = f.readline().strip().split(',')
        try:
            frame_idx = header.index('frame_id')
            action_idx = header.index('action')
            gaze_start_idx = header.index('gaze_positions')
        except ValueError:
            frame_idx = 0
            action_idx = 5
            gaze_start_idx = 6
            
        for line in f:
            parts = line.strip().split(',')
            if len(parts) < 6:
                continue
            
            frame_id = parts[frame_idx]
            try:
                action = int(parts[action_idx])
            except ValueError:
                continue
            
            if action >= 6:
                continue
                
            gaze_vals = parts[gaze_start_idx:]
            gaze_floats = []
            for g in gaze_vals:
                try:
                    gaze_floats.append(float(g))
                except ValueError:
                    pass
            
            data.append({
                'frame_id': frame_id,
                'action': action,
                'gaze_raw': gaze_floats
            })
    
    df = pd.DataFrame(data)
    print(f"Loaded {len(df)} rows")
    
    # Process gaze
    processed_gaze = []
    prev_gaze = None
    for index, row in df.iterrows():
        vals = row['gaze_raw']
        points = []
        for i in range(0, len(vals), 2):
            if i+1 < len(vals):
                points.append((vals[i], vals[i+1]))
        
        if not points:
            selected_gaze = prev_gaze if prev_gaze else (160/2, 210/2)
        else:
            if prev_gaze is None:
                selected_gaze = points[0]
            else:
                max_dist = -1
                best_point = points[0]
                for p in points:
                    dist = math.sqrt((p[0] - prev_gaze[0])**2 + (p[1] - prev_gaze[1])**2)
                    if dist > max_dist:
                        max_dist = dist
                        best_point = p
                selected_gaze = best_point
        
        processed_gaze.append(selected_gaze)
        prev_gaze = selected_gaze
        
    df['processed_gaze_x'] = [p[0] for p in processed_gaze]
    df['processed_gaze_y'] = [p[1] for p in processed_gaze]
    
    return df

class SeaquestDataset(Dataset):
    def __init__(self, df, image_dir, target_action, transform=None):
        self.df = df
        self.image_dir = image_dir
        self.transform = transform
        self.target_action = target_action
        self.images = []
        self.gazes = []
        self.labels = []
        
        print(f"Caching images for action {target_action}...")
        for idx in range(len(df)):
            row = df.iloc[idx]
            img_name = f"{row['frame_id']}.png"
            img_path = os.path.join(self.image_dir, img_name)
            
            try:
                image = Image.open(img_path).convert('RGB')
            except FileNotFoundError:
                image = Image.new('RGB', (210, 160))
                
            if self.transform:
                image = self.transform(image)
                
            self.images.append(image)
            
            gaze = torch.tensor([row['processed_gaze_x'] / 160.0, row['processed_gaze_y'] / 210.0], dtype=torch.float32)
            self.gazes.append(gaze)
            
            label = 1 if row['action'] == self.target_action else 0
            self.labels.append(torch.tensor(label, dtype=torch.float32))
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        return self.images[idx], self.gazes[idx], self.labels[idx]

class RGBModel(nn.Module):
    def __init__(self):
        super(RGBModel, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )
        self.fc = nn.Sequential(
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, 1) # Binary output
        )
        
    def forward(self, x, gaze=None):
        x = self.features(x)
        x = self.fc(x)
        return x

class RGBGazeModel(nn.Module):
    def __init__(self):
        super(RGBGazeModel, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )
        self.fc = nn.Sequential(
            nn.Linear(3136 + 2, 512),
            nn.ReLU(),
            nn.Linear(512, 1) # Binary output
        )
        
    def forward(self, x, gaze):
        x = self.features(x)
        x = torch.cat((x, gaze), dim=1)
        x = self.fc(x)
        return x

def train_model(model, train_loader, val_loader, device, model_name="Model"):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    print(f"Training {model_name}...")
    
    for epoch in range(EPOCHS):
        model.train()
        for images, gazes, labels in train_loader:
            images, gazes, labels = images.to(device), gazes.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images, gazes).squeeze(1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
    # Evaluation
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for images, gazes, labels in val_loader:
            images, gazes, labels = images.to(device), gazes.to(device), labels.to(device)
            outputs = model(images, gazes).squeeze(1)
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            
    f1 = f1_score(all_labels, all_preds)
    try:
        auc_pr = average_precision_score(all_labels, all_probs)
    except:
        auc_pr = 0.0
        
    return f1, auc_pr

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    df = load_and_process_data()
    
    transform = transforms.Compose([
        transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
        transforms.ToTensor(),
    ])
    
    results = {
        'action': [],
        'rgb_f1': [],
        'rgb_auc': [],
        'gaze_f1': [],
        'gaze_auc': []
    }
    
    for action in ACTIONS:
        print(f"\n--- Processing Action {action} ---")
        dataset = SeaquestDataset(df, IMAGE_DIR, action, transform=transform)
        
        # Check if action exists in dataset
        targets = [dataset[i][2].item() for i in range(len(dataset))]
        if sum(targets) == 0:
            print(f"Action {action} not found in dataset. Skipping.")
            results['action'].append(action)
            results['rgb_f1'].append(0)
            results['rgb_auc'].append(0)
            results['gaze_f1'].append(0)
            results['gaze_auc'].append(0)
            continue

        train_size = int(TRAIN_SPLIT * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
        
        # RGB Model
        rgb_model = RGBModel().to(device)
        rgb_f1, rgb_auc = train_model(rgb_model, train_loader, val_loader, device, f"RGB Model Action {action}")
        
        # Gaze Model
        gaze_model = RGBGazeModel().to(device)
        gaze_f1, gaze_auc = train_model(gaze_model, train_loader, val_loader, device, f"Gaze Model Action {action}")
        
        print(f"Action {action} Results:")
        print(f"RGB  - F1: {rgb_f1:.4f}, AUC-PR: {rgb_auc:.4f}")
        print(f"Gaze - F1: {gaze_f1:.4f}, AUC-PR: {gaze_auc:.4f}")
        
        results['action'].append(action)
        results['rgb_f1'].append(rgb_f1)
        results['rgb_auc'].append(rgb_auc)
        results['gaze_f1'].append(gaze_f1)
        results['gaze_auc'].append(gaze_auc)
        
    # Plotting
    x = np.arange(len(results['action']))
    width = 0.35
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # F1 Plot
    axes[0].bar(x - width/2, results['rgb_f1'], width, label='RGB Only', color='blue')
    axes[0].bar(x + width/2, results['gaze_f1'], width, label='RGB + Gaze', color='green')
    axes[0].set_ylabel('F1 Score')
    axes[0].set_title('F1 Score per Action')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(results['action'])
    axes[0].legend()
    
    # AUC-PR Plot
    axes[1].bar(x - width/2, results['rgb_auc'], width, label='RGB Only', color='red')
    axes[1].bar(x + width/2, results['gaze_auc'], width, label='RGB + Gaze', color='orange')
    axes[1].set_ylabel('AUC-PR')
    axes[1].set_title('AUC-PR per Action')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(results['action'])
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig('per_action_comparison.png')
    print("\nSaved plot to per_action_comparison.png")

if __name__ == "__main__":
    main()
