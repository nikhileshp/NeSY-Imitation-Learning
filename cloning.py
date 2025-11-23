import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import ast

# Load your dataset
df = pd.read_csv('data/seaquest/gaze_data_tmp/54_RZ_2461867_Aug-11-09-35-18_with_relationships_and_goals.txt')

class SeaquestDataset:
    """
    Process the Atari Seaquest eye tracking dataset for behavior cloning
    """
    
    def __init__(self, dataframe):
        self.df = dataframe.copy()
        self.process_relationships()
        
    def process_relationships(self):
        """
        Create new columns for each relationship with grounding values
        """
        # Parse the relationships column (assuming it's stored as string representation)
        def parse_relationships(rel_str):
            if pd.isna(rel_str) or rel_str == '':
                return {}
            try:
                # If it's a string representation of dict/list, parse it
                if isinstance(rel_str, str):
                    return ast.literal_eval(rel_str)
                return rel_str
            except:
                return {}
        
        # Parse relationships
        self.df['parsed_relationships'] = self.df['relationships'].apply(parse_relationships)
        
        # Extract unique relationship types
        all_relationships = set()
        for rels in self.df['parsed_relationships']:
            if isinstance(rels, dict):
                all_relationships.update(rels.keys())
            elif isinstance(rels, list):
                for rel in rels:
                    if isinstance(rel, dict) and 'type' in rel:
                        all_relationships.add(rel['type'])
        
        # Create a column for each relationship type with its grounding
        for rel_type in all_relationships:
            self.df[f'rel_{rel_type}'] = self.df['parsed_relationships'].apply(
                lambda x: self.extract_grounding(x, rel_type)
            )
        
        print(f"Created {len(all_relationships)} relationship columns: {all_relationships}")
        
    def extract_grounding(self, relationships, rel_type):
        """
        Extract grounding value for a specific relationship type
        """
        if isinstance(relationships, dict):
            return relationships.get(rel_type, 0)
        elif isinstance(relationships, list):
            for rel in relationships:
                if isinstance(rel, dict):
                    if rel.get('type') == rel_type:
                        return rel.get('grounding', 1)  # Return grounding value
        return 0
    
    def get_processed_dataframe(self):
        # Drop unnecessary columns
        cols_to_drop = ['distance_weights', 'predicate_weights', 'example_weight', 
                        'parsed_relationships', 'relationships']
        return self.df.drop(columns=[col for col in cols_to_drop if col in self.df.columns])


class AtariSeaquestDataset(Dataset):
    """
    PyTorch Dataset for behavior cloning
    """
    
    def __init__(self, dataframe, object_encoder=None, goal_encoder=None, action_encoder=None):
        self.df = dataframe.reset_index(drop=True)
        
        # Encode categorical variables
        self.object_encoder = object_encoder or LabelEncoder()
        self.goal_encoder = goal_encoder or LabelEncoder()
        self.action_encoder = action_encoder or LabelEncoder()
        
        # Handle objects column
        self.df['objects_encoded'] = self.encode_column('objects', self.object_encoder)
        
        # Handle goal column
        self.df['goal_encoded'] = self.encode_column('goal', self.goal_encoder)
        
        # Encode actions (target variable)
        if action_encoder.classes_ is None or len(action_encoder.classes_) == 0:
            self.df['action_encoded'] = self.action_encoder.fit_transform(self.df['action'].astype(str))
        else:
            self.df['action_encoded'] = self.action_encoder.transform(self.df['action'].astype(str))
        
        # Identify relationship columns
        self.rel_columns = [col for col in self.df.columns if col.startswith('rel_')]
        
        # Numerical features
        self.numeric_features = ['score', 'duration', 'unclipped_reward', 
                                 'objects_encoded', 'goal_encoded'] + self.rel_columns
        
    def encode_column(self, col_name, encoder):
        """Encode a column, handling various data types"""
        try:
            col_data = self.df[col_name].astype(str).fillna('none')
            if not hasattr(encoder, 'classes_') or encoder.classes_ is None or len(encoder.classes_) == 0:
                return encoder.fit_transform(col_data)
            else:
                return encoder.transform(col_data)
        except:
            return self.df[col_name].fillna(0)
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        # Get features
        features = self.df.loc[idx, self.numeric_features].values.astype(np.float32)
        
        # Get action (target)
        action = self.df.loc[idx, 'action_encoded']
        
        return torch.tensor(features, dtype=torch.float32), torch.tensor(action, dtype=torch.long)


class BehaviorCloningModel(nn.Module):
    """
    Neural network for behavior cloning
    """
    
    def __init__(self, input_dim, num_actions, hidden_dims=[256, 128, 64]):
        super(BehaviorCloningModel, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        # Build hidden layers
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.3))
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, num_actions))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)


def train_behavior_cloning(model, train_loader, val_loader, num_epochs=50, 
                          learning_rate=0.001, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Train the behavior cloning model
    """
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', 
                                                      factor=0.5, patience=5, verbose=True)
    
    best_val_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for features, actions in train_loader:
            features, actions = features.to(device), actions.to(device)
            
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, actions)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += actions.size(0)
            train_correct += (predicted == actions).sum().item()
        
        train_loss = train_loss / len(train_loader)
        train_acc = 100 * train_correct / train_total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for features, actions in val_loader:
                features, actions = features.to(device), actions.to(device)
                outputs = model(features)
                loss = criterion(outputs, actions)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += actions.size(0)
                val_correct += (predicted == actions).sum().item()
        
        val_loss = val_loss / len(val_loader)
        val_acc = 100 * val_correct / val_total
        
        # Update learning rate
        scheduler.step(val_acc)
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_bc_model.pth')
        
        if (epoch + 1) % 5 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}]')
            print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
            print('-' * 60)
    
    return history, best_val_acc


# Main execution pipeline
def main(data_path):
    """
    Complete pipeline for behavior cloning on Seaquest dataset
    """
    
    # Load data
    print("Loading dataset...")
    df = pd.read_csv(data_path)
    
    # Process relationships
    print("Processing relationships...")
    dataset_processor = SeaquestDataset(df)
    processed_df = dataset_processor.get_processed_dataframe()
    
    print(f"\nProcessed dataset shape: {processed_df.shape}")
    print(f"Columns: {processed_df.columns.tolist()}")
    
    # Split by episode to avoid data leakage
    unique_episodes = processed_df['episode_id'].unique()
    train_episodes, test_episodes = train_test_split(unique_episodes, 
                                                      test_size=0.2, random_state=42)
    
    train_df = processed_df[processed_df['episode_id'].isin(train_episodes)]
    test_df = processed_df[processed_df['episode_id'].isin(test_episodes)]
    
    # Further split train into train and validation
    train_episodes_split, val_episodes = train_test_split(train_episodes, 
                                                           test_size=0.2, random_state=42)
    train_df = processed_df[processed_df['episode_id'].isin(train_episodes_split)]
    val_df = processed_df[processed_df['episode_id'].isin(val_episodes)]
    
    print(f"\nTrain size: {len(train_df)}, Val size: {len(val_df)}, Test size: {len(test_df)}")
    
    # Create PyTorch datasets
    train_dataset = AtariSeaquestDataset(train_df)
    val_dataset = AtariSeaquestDataset(val_df, 
                                       object_encoder=train_dataset.object_encoder,
                                       goal_encoder=train_dataset.goal_encoder,
                                       action_encoder=train_dataset.action_encoder)
    test_dataset = AtariSeaquestDataset(test_df,
                                        object_encoder=train_dataset.object_encoder,
                                        goal_encoder=train_dataset.goal_encoder,
                                        action_encoder=train_dataset.action_encoder)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=4)
    
    # Initialize model
    input_dim = len(train_dataset.numeric_features)
    num_actions = len(train_dataset.action_encoder.classes_)
    
    print(f"\nModel input dimension: {input_dim}")
    print(f"Number of actions: {num_actions}")
    
    model = BehaviorCloningModel(input_dim, num_actions, hidden_dims=[256, 128, 64])
    
    print(f"\nModel architecture:\n{model}")
    
    # Train model
    print("\nTraining behavior cloning model...")
    history, best_val_acc = train_behavior_cloning(model, train_loader, val_loader, 
                                                    num_epochs=50, learning_rate=0.001)
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.load_state_dict(torch.load('best_bc_model.pth'))
    model = model.to(device)
    model.eval()
    
    test_correct = 0
    test_total = 0
    
    with torch.no_grad():
        for features, actions in test_loader:
            features, actions = features.to(device), actions.to(device)
            outputs = model(features)
            _, predicted = torch.max(outputs.data, 1)
            test_total += actions.size(0)
            test_correct += (predicted == actions).sum().item()
    
    test_acc = 100 * test_correct / test_total
    print(f"Test Accuracy: {test_acc:.2f}%")
    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
    
    # Save training history
    history_df = pd.DataFrame(history)
    history_df.to_csv('training_history.csv', index=False)
    print("\nTraining history saved to 'training_history.csv'")
    
    return model, history, train_dataset.action_encoder


# Usage example:
if __name__ == "__main__":
    model, history, action_encoder = main('data/seaquest/gaze_data_tmp/54_RZ_2461867_Aug-11-09-35-18_with_relationships_and_goals.txt')
