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

# Configuration
# Image directory
IMAGE_DIR = "/home/nikhilesh/Projects/NeSY-Imitation-Learning/data/seaquest/gaze_data_tmp/54_RZ_2461867_Aug-11-09-35-18"
# CSV File path (located in parent directory based on exploration)
CSV_FILE = "/home/nikhilesh/Projects/NeSY-Imitation-Learning/data/seaquest/54_RZ_2461867_Aug-11-09-35-18.txt"
BATCH_SIZE = 32
EPOCHS = 15
LEARNING_RATE = 0.001
IMG_HEIGHT = 84
IMG_WIDTH = 84
TRAIN_SPLIT = 0.8
SEED = 42

# Set seed for reproducibility
torch.manual_seed(SEED)
np.random.seed(SEED)

def load_and_process_data():
    print("Loading data...")
    
    data = []
    with open(CSV_FILE, 'r') as f:
        header = f.readline().strip().split(',')
        # Identify indices
        try:
            frame_idx = header.index('frame_id')
            action_idx = header.index('action')
            gaze_start_idx = header.index('gaze_positions')
        except ValueError:
            # Fallback if header is different or gaze_positions is just the start of the rest
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
                continue # Skip bad lines
                
            # 3. Clean the dataframe and remove all action greater than equal to 6
            if action >= 6:
                continue
                
            gaze_vals = parts[gaze_start_idx:]
            # Convert to floats
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
    print(f"Loaded {len(df)} rows after filtering actions < 6")
    
    # 4. Process gaze positions
    print("Processing gaze positions...")
    processed_gaze = []
    prev_gaze = None
    
    for index, row in df.iterrows():
        vals = row['gaze_raw']
            
        points = []
        for i in range(0, len(vals), 2):
            if i+1 < len(vals):
                points.append((vals[i], vals[i+1]))
        
        if not points:
            # Fallback if no gaze data, use center or previous
            selected_gaze = prev_gaze if prev_gaze else (160/2, 210/2) # Assuming Atari res
        else:
            if prev_gaze is None:
                # 4.1 First frame: first x,y position
                selected_gaze = points[0]
            else:
                # From second frame: farthest from previous frame
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
    def __init__(self, df, image_dir, transform=None):
        self.df = df
        self.image_dir = image_dir
        self.transform = transform
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_name = f"{row['frame_id']}.png"
        img_path = os.path.join(self.image_dir, img_name)
        
        try:
            image = Image.open(img_path).convert('RGB')
        except FileNotFoundError:
            # Handle missing images if any (though we expect them to exist)
            # Create a black image as placeholder
            image = Image.new('RGB', (210, 160))
            
        if self.transform:
            image = self.transform(image)
            
        # Normalize gaze (assuming 160x210 roughly, but let's just use raw or simple scaling)
        # Atari resolution is usually 160x210 (WxH). Let's scale to [0, 1]
        gaze = torch.tensor([row['processed_gaze_x'] / 160.0, row['processed_gaze_y'] / 210.0], dtype=torch.float32)
        action = torch.tensor(row['action'], dtype=torch.long)
        
        return image, gaze, action

# 6. Define Neural Networks
class RGBModel(nn.Module):
    def __init__(self, num_actions=6):
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
        
        # Calculate linear input size based on 84x84 input
        # 84 -> 20 -> 9 -> 7. 64 * 7 * 7 = 3136
        self.fc = nn.Sequential(
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )
        
    def forward(self, x, gaze=None):
        x = self.features(x)
        x = self.fc(x)
        return x

class RGBGazeModel(nn.Module):
    def __init__(self, num_actions=6):
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
        
        # RGB features (3136) + Gaze features (2)
        self.fc = nn.Sequential(
            nn.Linear(3136 + 2, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )
        
    def forward(self, x, gaze):
        x = self.features(x)
        x = torch.cat((x, gaze), dim=1)
        x = self.fc(x)
        return x

from sklearn.metrics import f1_score, average_precision_score
from sklearn.preprocessing import label_binarize

def train_model(model, train_loader, val_loader, device, model_name="Model", num_classes=6):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    print(f"\nTraining {model_name}...")
    train_losses = []
    
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        for images, gazes, actions in train_loader:
            images, gazes, actions = images.to(device), gazes.to(device), actions.to(device)
            
            optimizer.zero_grad()
            outputs = model(images, gazes)
            loss = criterion(outputs, actions)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        epoch_loss = running_loss / len(train_loader)
        train_losses.append(epoch_loss)
        
        # Validation
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for images, gazes, actions in val_loader:
                images, gazes, actions = images.to(device), gazes.to(device), actions.to(device)
                outputs = model(images, gazes)
                loss = criterion(outputs, actions)
                val_loss += loss.item()
                
                probs = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs.data, 1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(actions.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        avg_val_loss = val_loss / len(val_loader)
        
        # Calculate metrics
        val_acc = 100 * np.mean(np.array(all_preds) == np.array(all_labels))
        val_f1 = f1_score(all_labels, all_preds, average='weighted')
        
        # AUC-PR requires binarized labels for multi-class
        # We need to handle cases where not all classes are present in the batch/split
        # But for validation set it should be fine usually. 
        # If a class is missing, label_binarize might output fewer columns, so we force classes.
        y_bin = label_binarize(all_labels, classes=range(num_classes))
        # Check if y_bin has correct shape (n_samples, n_classes)
        if y_bin.shape[1] != num_classes:
             # This happens if some classes are missing in validation set
             # We need to pad y_bin or handle it. 
             # For simplicity, let's re-binarize ensuring all classes
             pass # label_binarize with classes arg handles this correctly
             
        try:
            val_auc_pr = average_precision_score(y_bin, all_probs, average='weighted')
        except ValueError:
            val_auc_pr = 0.0 # Handle edge cases
            
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {epoch_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%, F1: {val_f1:.4f}, AUC-PR: {val_auc_pr:.4f}")
        
    return val_acc, avg_val_loss, val_f1, val_auc_pr

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    df = load_and_process_data()
    
    transform = transforms.Compose([
        transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
        transforms.ToTensor(),
    ])
    
    dataset = SeaquestDataset(df, IMAGE_DIR, transform=transform)
    
    train_size = int(TRAIN_SPLIT * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Train RGB Model
    rgb_model = RGBModel().to(device)
    rgb_acc, rgb_loss, rgb_f1, rgb_auc = train_model(rgb_model, train_loader, val_loader, device, "RGB Model")
    
    # Train RGB + Gaze Model
    rgb_gaze_model = RGBGazeModel().to(device)
    gaze_acc, gaze_loss, gaze_f1, gaze_auc = train_model(rgb_gaze_model, train_loader, val_loader, device, "RGB + Gaze Model")
    
    print("\nResults:")
    print(f"RGB Model - Acc: {rgb_acc:.2f}%, Loss: {rgb_loss:.4f}, F1: {rgb_f1:.4f}, AUC-PR: {rgb_auc:.4f}")
    print(f"RGB + Gaze Model - Acc: {gaze_acc:.2f}%, Loss: {gaze_loss:.4f}, F1: {gaze_f1:.4f}, AUC-PR: {gaze_auc:.4f}")
    
    # 9. Compare the results with bar plots
    models = ['RGB Only', 'RGB + Gaze']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Accuracy
    axes[0, 0].bar(models, [rgb_acc, gaze_acc], color=['blue', 'green'])
    axes[0, 0].set_title('Validation Accuracy')
    axes[0, 0].set_ylabel('Accuracy (%)')
    for i, v in enumerate([rgb_acc, gaze_acc]):
        axes[0, 0].text(i, v + 0.5, f"{v:.2f}%", ha='center')

    # Loss
    axes[0, 1].bar(models, [rgb_loss, gaze_loss], color=['red', 'orange'])
    axes[0, 1].set_title('Validation Cross Entropy Loss')
    axes[0, 1].set_ylabel('Loss')
    for i, v in enumerate([rgb_loss, gaze_loss]):
        axes[0, 1].text(i, v + 0.01, f"{v:.4f}", ha='center')
        
    # F1 Score
    axes[1, 0].bar(models, [rgb_f1, gaze_f1], color=['purple', 'brown'])
    axes[1, 0].set_title('Weighted F1 Score')
    axes[1, 0].set_ylabel('F1 Score')
    for i, v in enumerate([rgb_f1, gaze_f1]):
        axes[1, 0].text(i, v + 0.01, f"{v:.4f}", ha='center')
        
    # AUC-PR
    axes[1, 1].bar(models, [rgb_auc, gaze_auc], color=['cyan', 'magenta'])
    axes[1, 1].set_title('Weighted AUC-PR')
    axes[1, 1].set_ylabel('AUC-PR')
    for i, v in enumerate([rgb_auc, gaze_auc]):
        axes[1, 1].text(i, v + 0.01, f"{v:.4f}", ha='center')
        
    plt.tight_layout()
    plt.savefig('model_comparison.png')
    print("Comparison plot saved to model_comparison.png")

if __name__ == "__main__":
    main()
