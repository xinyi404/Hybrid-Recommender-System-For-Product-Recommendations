import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split # Added for splitting data
import math # Added for sqrt in RMSE

# --- Configuration ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
RATINGS_CSV_PATH = 'C:/Users/mochi/OneDrive/Documents/MMU/Bachelors in Computer Science/FYP/code/cleaned_merged.csv' # Path to your data
USER_COL = 'UserId'            
ITEM_COL = 'ProductId'         
RATING_COL = 'Score'           
USER_ENCODER_PATH = 'user_encoder.pkl'
ITEM_ENCODER_PATH = 'item_encoder.pkl'
EMBEDDING_DIM = 64             
BATCH_SIZE = 64                
EPOCHS = 10                    # You can adjust the number of epochs here
LEARNING_RATE = 0.001          
OUTPUT_MODEL_PATH = 'cf_model.pth' 
VALIDATION_SPLIT = 0.2         # Proportion of data to use for validation

# --- 1. Load Encoders ---
print("Loading encoders...")
with open(USER_ENCODER_PATH, 'rb') as f:
    user_encoder_array = pickle.load(f)
with open(ITEM_ENCODER_PATH, 'rb') as f:
    item_encoder_array = pickle.load(f)

# Create mappings from ID to index
user_id_to_idx = {user_id: idx for idx, user_id in enumerate(user_encoder_array)}
item_id_to_idx = {item_id: idx for idx, item_id in enumerate(item_encoder_array)}

num_users = len(user_encoder_array)
num_items = len(item_encoder_array)

print(f"Found {num_users} unique users and {num_items} unique items.")

# --- 2. Define CF Model (Matrix Factorization) ---
class CFModel(nn.Module):
    def __init__(self, n_users, n_items, embedding_dim):
        super().__init__()
        self.user_embeddings = nn.Embedding(n_users, embedding_dim)
        self.item_embeddings = nn.Embedding(n_items, embedding_dim)
        # Optional: Add user/item biases
        # self.user_bias = nn.Embedding(n_users, 1)
        # self.item_bias = nn.Embedding(n_items, 1)
        
        # Initialize weights
        nn.init.normal_(self.user_embeddings.weight, std=0.01)
        nn.init.normal_(self.item_embeddings.weight, std=0.01)

    def forward(self, user_indices, item_indices):
        user_embedding = self.user_embeddings(user_indices)
        item_embedding = self.item_embeddings(item_indices)
        # Dot product of user and item embeddings
        dot_product = (user_embedding * item_embedding).sum(1)
        # Optional: Add biases
        # user_b = self.user_bias(user_indices).squeeze()
        # item_b = self.item_bias(item_indices).squeeze()
        # return dot_product + user_b + item_b
        return dot_product

# --- 3. Prepare Data ---
class RatingsDataset(Dataset):
    def __init__(self, dataframe, user_map, item_map, user_col, item_col, rating_col):
        # Ensure indices exist before trying to access them
        valid_users = dataframe[user_col].isin(user_map)
        valid_items = dataframe[item_col].isin(item_map)
        valid_df = dataframe[valid_users & valid_items]

        self.users = torch.tensor([user_map[i] for i in valid_df[user_col]], dtype=torch.long)
        self.items = torch.tensor([item_map[i] for i in valid_df[item_col]], dtype=torch.long)
        self.ratings = torch.tensor(valid_df[rating_col].values, dtype=torch.float)

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        return self.users[idx], self.items[idx], self.ratings[idx]

print(f"Loading ratings data from {RATINGS_CSV_PATH}...")
try:
    ratings_df = pd.read_csv(RATINGS_CSV_PATH)
    # Filter out users/items not in the encoders (important!)
    ratings_df = ratings_df[ratings_df[USER_COL].isin(user_id_to_idx)]
    ratings_df = ratings_df[ratings_df[ITEM_COL].isin(item_id_to_idx)]
    print(f"Loaded {len(ratings_df)} ratings after filtering known users/items.")
except FileNotFoundError:
    print(f"Error: Ratings file not found at {RATINGS_CSV_PATH}")
    exit()
except KeyError as e:
    print(f"Error: Column {e} not found in {RATINGS_CSV_PATH}. Adjust column names.")
    exit()

# Split data into training and validation sets
print(f"Splitting data into training ({1-VALIDATION_SPLIT:.0%}) and validation ({VALIDATION_SPLIT:.0%})...")
train_df, val_df = train_test_split(ratings_df, test_size=VALIDATION_SPLIT, random_state=42) # Added random_state for reproducibility
print(f"Training set size: {len(train_df)}, Validation set size: {len(val_df)}")

# Create Datasets and DataLoaders
train_dataset = RatingsDataset(train_df, user_id_to_idx, item_id_to_idx, USER_COL, ITEM_COL, RATING_COL)
val_dataset = RatingsDataset(val_df, user_id_to_idx, item_id_to_idx, USER_COL, ITEM_COL, RATING_COL)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False) # No need to shuffle validation data

# --- 4. Training and Validation Loop ---
def train_and_validate(model, train_loader, val_loader, criterion, optimizer, epochs, device):
    for epoch in range(epochs):
        # --- Training Phase ---
        model.train() # Set model to training mode
        total_train_loss = 0
        print(f'--- Starting Epoch {epoch+1}/{epochs} ---')
        for batch_idx, (users, items, ratings) in enumerate(train_loader):
            users, items, ratings = users.to(device), items.to(device), ratings.to(device)
            
            optimizer.zero_grad()
            predictions = model(users, items)
            loss = criterion(predictions, ratings)
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            
            if batch_idx % 200 == 0: # Print progress less frequently for potentially large datasets
                print(f'  Epoch {epoch+1}, Batch {batch_idx}/{len(train_loader)}, Train Loss: {loss.item():.4f}')
                
        avg_train_loss = total_train_loss / len(train_loader)
        print(f'--- Epoch {epoch+1} Training Finished, Average Train Loss: {avg_train_loss:.4f} ---')

        # --- Validation Phase ---
        model.eval() # Set model to evaluation mode
        total_val_loss = 0
        total_squared_error = 0
        num_samples = 0
        
        with torch.no_grad(): # Disable gradient calculation for validation
            for users, items, ratings in val_loader:
                users, items, ratings = users.to(device), items.to(device), ratings.to(device)
                predictions = model(users, items)
                loss = criterion(predictions, ratings)
                total_val_loss += loss.item()
                
                # Calculate squared error for RMSE
                total_squared_error += torch.sum((predictions - ratings) ** 2).item()
                num_samples += len(ratings)

        avg_val_loss = total_val_loss / len(val_loader)
        rmse = math.sqrt(total_squared_error / num_samples) if num_samples > 0 else 0
        
        print(f'--- Epoch {epoch+1} Validation Finished, Average Val Loss: {avg_val_loss:.4f}, Val RMSE: {rmse:.4f} ---')

# --- 5. Instantiate, Train, and Save ---
print("Initializing CF model...")
cf_model = CFModel(num_users, num_items, EMBEDDING_DIM).to(device)

# Define Loss and Optimizer
criterion = nn.MSELoss() # Mean Squared Error for explicit ratings
optimizer = optim.Adam(cf_model.parameters(), lr=LEARNING_RATE)

print("Starting training and validation...")
train_and_validate(cf_model, train_loader, val_loader, criterion, optimizer, EPOCHS, device)

print(f"Training finished. Saving model to {OUTPUT_MODEL_PATH}...")
torch.save(cf_model.state_dict(), OUTPUT_MODEL_PATH)
print("Model saved successfully.")
