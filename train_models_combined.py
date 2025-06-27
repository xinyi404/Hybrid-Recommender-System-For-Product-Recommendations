#combine CF and CBF
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder
from transformers import BertTokenizer, BertModel
import pickle
import math
import os
import copy

# --- Configuration ---
# General
BASE_DATA_PATH = 'cleaned_merged.csv'
RANDOM_STATE = 42
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

# CBF Configuration
CBF_TEXT_COLUMN = 'CleanedSummary'
CBF_SCORE_COLUMN_TO_CHECK = 'Score' # Using 'Score' directly as per test_data.ipynb for individual review sentiment
CBF_LABEL_COLUMN = 'cbf_sentiment_label'
CBF_BERT_MODEL_NAME = 'bert-base-uncased' # Consistent with model.py
CBF_BATCH_SIZE_EMBEDDING = 32 # For generating embeddings
CBF_MODEL_OUTPUT_PATH = 'cbf_model.pth'
CBF_TRAIN_TEST_SPLIT_SIZE = 0.2
CBF_CLASSIFIER_BATCH_SIZE = 64
CBF_CLASSIFIER_EPOCHS = 15 # Starting point, can be tuned. Original 8020TrainTest used many epochs.
CBF_CLASSIFIER_LR = 1e-5 # Typical learning rate for fine-tuning BERT-based classifiers

# CF Configuration
CF_USER_COL = 'UserId'
CF_ITEM_COL = 'ProductId'
CF_RATING_COL = 'Score'
CF_MODEL_OUTPUT_PATH = 'cf_model.pth'
CF_MAPPINGS_OUTPUT_PATH = 'cf_mappings.pkl'
CF_EMBEDDING_DIM = 64 # Reduced from 512 to combat overfitting, can be tuned
CF_BATCH_SIZE = 256
CF_EPOCHS = 50      # Reduced from 100, with early stopping this is max epochs
CF_LEARNING_RATE = 0.005 # Can be tuned
CF_WEIGHT_DECAY = 1e-4 # Added L2 Regularization
# CF Data Splitting (consistent with test_data.ipynb and original train_cf.ipynb)
# test_data.ipynb implies a 20% test set.
# train_cf.ipynb used a 20% validation split from the remaining 80%.
# So, 0.8 * 0.2 = 16% for validation. 0.8 * 0.8 = 64% for training.
CF_OVERALL_TEST_SPLIT = 0.2 # This portion is conceptually set aside for test_data.ipynb
CF_VALIDATION_SPLIT_FROM_TRAIN_VAL = 0.2 # Val percentage from (1-CF_OVERALL_TEST_SPLIT) data

# --- Helper Functions & Classes ---

# CBF: BERT Embedding Generation (from model.py / precompute_embeddings.py logic)
def get_bert_embeddings_for_cbf(texts, model_name=CBF_BERT_MODEL_NAME, batch_size=CBF_BATCH_SIZE_EMBEDDING, device=DEVICE):
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name).to(device)
    model.eval()
    
    all_embeddings = []
    print(f"Generating CBF BERT embeddings with {model_name} on {device} for {len(texts)} texts...")
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        inputs = tokenizer(batch_texts, return_tensors='pt', padding=True, truncation=True, max_length=512).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        # Use mean pooling of the last hidden state
        embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
        all_embeddings.extend(embeddings)
        if (i // batch_size) % 20 == 0:
             print(f"  CBF Embedding: Processed batch {i // batch_size + 1} / {len(texts) // batch_size + 1}")
    print("CBF BERT embeddings generated.")
    return np.array(all_embeddings)

# CBF: Sentiment Labeling
def cbf_sentiment_label_fn(score):
    if score < 3: return 0  # Negative
    if score == 3: return 1 # Neutral
    return 2  # Positive

# CBF: Classifier Model
class BERTClassifier(nn.Module):
    def __init__(self, bert_embedding_dim, num_classes, dropout_rate=0.3):
        super(BERTClassifier, self).__init__()
        # Assuming pre-computed BERT embeddings are input
        self.fc1 = nn.Linear(bert_embedding_dim, 512) 
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, embeddings):
        x = self.fc1(embeddings)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# # CF: Dataset Class
# class CFRatingsDataset(Dataset):
#     def __init__(self, users, items, ratings):
#         self.users = users
#         self.items = items
#         self.ratings = ratings

#     def __len__(self):
#         return len(self.ratings)

#     def __getitem__(self, idx):
#         return self.users[idx], self.items[idx], self.ratings[idx]

# # CF: Model Class
# class CFModel(nn.Module):
#     def __init__(self, n_users: int, n_items: int, embedding_dim: int):
#         super().__init__()
#         self.user_embeddings = nn.Embedding(n_users, embedding_dim)
#         self.item_embeddings = nn.Embedding(n_items, embedding_dim)
#         self.user_bias = nn.Embedding(n_users, 1)
#         self.item_bias = nn.Embedding(n_items, 1)

#         # Initialize weights
#         nn.init.normal_(self.user_embeddings.weight, std=0.01)
#         nn.init.normal_(self.item_embeddings.weight, std=0.01)
#         nn.init.zeros_(self.user_bias.weight)
#         nn.init.zeros_(self.item_bias.weight)

#     def forward(self, user_indices: torch.Tensor, item_indices: torch.Tensor, apply_sigmoid: bool = False) -> torch.Tensor:
#         user_embedding = self.user_embeddings(user_indices)
#         item_embedding = self.item_embeddings(item_indices)
#         dot_product = (user_embedding * item_embedding).sum(dim=1)

#         user_b = self.user_bias(user_indices).squeeze()
#         item_b = self.item_bias(item_indices).squeeze()

#         prediction = dot_product + user_b + item_b

#         if apply_sigmoid:
#             # Scale to 1–5 range
#             prediction = torch.sigmoid(prediction) * 4 + 1

#         return prediction

# --- Main Script ---
def main():
    # --- Load Data ---
    print(f"Loading data from {BASE_DATA_PATH}...")
    try:
        df_full = pd.read_csv(BASE_DATA_PATH)
        print(f"Data loaded. Shape: {df_full.shape}")
    except FileNotFoundError:
        print(f"Error: File not found at {BASE_DATA_PATH}. Exiting.")
        return
    except Exception as e:
        print(f"Error loading data: {e}. Exiting.")
        return

    # === CBF Model Training ===
    print("\\n--- Starting CBF Model Training ---")
    df_cbf = df_full[[CBF_TEXT_COLUMN, CBF_SCORE_COLUMN_TO_CHECK]].copy()
    df_cbf.dropna(subset=[CBF_TEXT_COLUMN, CBF_SCORE_COLUMN_TO_CHECK], inplace=True)
    df_cbf[CBF_TEXT_COLUMN] = df_cbf[CBF_TEXT_COLUMN].astype(str)
    
    if df_cbf.empty:
        print("CBF data is empty after cleaning. Skipping CBF training.")
    else:
        print(f"Prepared {len(df_cbf)} samples for CBF.")
        
        # 1. CBF: Generate Labels
        print("Generating CBF sentiment labels...")
        df_cbf[CBF_LABEL_COLUMN] = df_cbf[CBF_SCORE_COLUMN_TO_CHECK].apply(cbf_sentiment_label_fn)
        print("CBF Label distribution:")
        print(df_cbf[CBF_LABEL_COLUMN].value_counts())

        # 2. CBF: Generate Embeddings
        texts_for_cbf_embeddings = df_cbf[CBF_TEXT_COLUMN].tolist()
        cbf_initial_labels = df_cbf[CBF_LABEL_COLUMN].values
        
        X_cbf_embeddings = get_bert_embeddings_for_cbf(texts_for_cbf_embeddings)
        y_cbf_labels = cbf_initial_labels # Use corresponding labels

        # 3. CBF: Data Splitting
        print("Splitting CBF data into training and test sets...")
        X_cbf_train, X_cbf_test, y_cbf_train, y_cbf_test = train_test_split(
            X_cbf_embeddings, y_cbf_labels,
            test_size=CBF_TRAIN_TEST_SPLIT_SIZE,
            random_state=RANDOM_STATE,
            stratify=y_cbf_labels
        )
        cbf_train_dataset = TensorDataset(
            torch.tensor(X_cbf_train, dtype=torch.float32),
            torch.tensor(y_cbf_train, dtype=torch.long)
        )
        sm = SMOTE(random_state=RANDOM_STATE)
        X_cbf_train, y_cbf_train = sm.fit_resample(X_cbf_train, y_cbf_train)
        print("SMOTE applied. New class distribution:", np.bincount(y_cbf_train))

        print(f"CBF Train shapes: X={X_cbf_train.shape}, y={y_cbf_train.shape}")
        print(f"CBF Test shapes: X={X_cbf_test.shape}, y={y_cbf_test.shape}")

        cbf_train_dataset = TensorDataset(torch.tensor(X_cbf_train, dtype=torch.float32), torch.tensor(y_cbf_train, dtype=torch.long))
        cbf_test_dataset = TensorDataset(torch.tensor(X_cbf_test, dtype=torch.float32), torch.tensor(y_cbf_test, dtype=torch.long))
        cbf_train_loader = DataLoader(cbf_train_dataset, batch_size=CBF_CLASSIFIER_BATCH_SIZE, shuffle=True)
        cbf_test_loader = DataLoader(cbf_test_dataset, batch_size=CBF_CLASSIFIER_BATCH_SIZE, shuffle=False)

        # 4. CBF: Model, Optimizer, Criterion
        cbf_bert_embedding_dim = X_cbf_embeddings.shape[1]
        num_cbf_classes = len(np.unique(y_cbf_labels))
        cbf_classifier = BERTClassifier(cbf_bert_embedding_dim, num_cbf_classes).to(DEVICE)
        cbf_optimizer = optim.AdamW(cbf_classifier.parameters(), lr=CBF_CLASSIFIER_LR)
        cbf_criterion = nn.CrossEntropyLoss()

        # 5. CBF: Training Loop
        print("Training CBF classifier...")
        for epoch in range(CBF_CLASSIFIER_EPOCHS):
            cbf_classifier.train()
            total_cbf_loss = 0
            correct_cbf_predictions = 0
            total_cbf_samples = 0
            for batch_embeddings, batch_labels in cbf_train_loader:
                batch_embeddings, batch_labels = batch_embeddings.to(DEVICE), batch_labels.to(DEVICE)
                
                cbf_optimizer.zero_grad()
                outputs = cbf_classifier(batch_embeddings)
                loss = cbf_criterion(outputs, batch_labels)
                loss.backward()
                cbf_optimizer.step()
                
                total_cbf_loss += loss.item() * batch_embeddings.size(0)
                _, predicted = torch.max(outputs.data, 1)
                correct_cbf_predictions += (predicted == batch_labels).sum().item()
                total_cbf_samples += batch_labels.size(0)
            
            avg_cbf_train_loss = total_cbf_loss / total_cbf_samples
            cbf_train_accuracy = correct_cbf_predictions / total_cbf_samples
            print(f"CBF Epoch {epoch+1}/{CBF_CLASSIFIER_EPOCHS} - Train Loss: {avg_cbf_train_loss:.4f}, Train Acc: {cbf_train_accuracy:.4f}")

        # 6. CBF: Evaluation
        cbf_classifier.eval()
        total_cbf_test_loss = 0
        correct_cbf_test_predictions = 0
        total_cbf_test_samples = 0
        with torch.no_grad():
            for batch_embeddings, batch_labels in cbf_test_loader:
                batch_embeddings, batch_labels = batch_embeddings.to(DEVICE), batch_labels.to(DEVICE)
                outputs = cbf_classifier(batch_embeddings)
                loss = cbf_criterion(outputs, batch_labels)
                total_cbf_test_loss += loss.item() * batch_embeddings.size(0)
                _, predicted = torch.max(outputs.data, 1)
                correct_cbf_test_predictions += (predicted == batch_labels).sum().item()
                total_cbf_test_samples += batch_labels.size(0)
        
        avg_cbf_test_loss = total_cbf_test_loss / total_cbf_test_samples
        cbf_test_accuracy = correct_cbf_test_predictions / total_cbf_test_samples
        print(f"CBF Test - Loss: {avg_cbf_test_loss:.4f}, Accuracy: {cbf_test_accuracy:.4f}")

        # 7. CBF: Save Model
        try:
            torch.save(cbf_classifier.state_dict(), CBF_MODEL_OUTPUT_PATH)
            print(f"CBF model saved to {CBF_MODEL_OUTPUT_PATH}")
        except Exception as e:
            print(f"Error saving CBF model: {e}")
        print("--- CBF Model Training Finished ---")


    # # === CF Model Training ===
    # print("\\n--- Starting CF Model Training ---")
    # df_cf = df_full[[CF_USER_COL, CF_ITEM_COL, CF_RATING_COL]].copy()
    # df_cf.dropna(subset=[CF_USER_COL, CF_ITEM_COL, CF_RATING_COL], inplace=True)
    # print("Rating distribution:")
    # print(df_cf[CF_RATING_COL].value_counts())
    # if len(df_cf) < (CF_BATCH_SIZE * 2): # Need enough data for train/val split
    #     print(f"Not enough CF data for training (found {len(df_cf)} rows). Skipping CF training.")
    # else:
    #     print(f"Prepared {len(df_cf)} ratings for CF.")

    #     # 1. CF: Encode User and Item IDs
    #     print("Encoding User and Item IDs for CF...")
    #     user_encoder = LabelEncoder()
    #     item_encoder = LabelEncoder()
    #     df_cf['user_encoded'] = user_encoder.fit_transform(df_cf[CF_USER_COL])
    #     df_cf['item_encoded'] = item_encoder.fit_transform(df_cf[CF_ITEM_COL])
        
    #     n_users = df_cf['user_encoded'].nunique()
    #     n_items = df_cf['item_encoded'].nunique()
    #     print(f"CF: Found {n_users} unique users, {n_items} unique items.")

    #     # Save mappings
    #     cf_mappings = {
    #         'user_id_to_idx': {original_id: encoded_id for original_id, encoded_id in zip(user_encoder.classes_, user_encoder.transform(user_encoder.classes_))},
    #         'item_id_to_idx': {original_id: encoded_id for original_id, encoded_id in zip(item_encoder.classes_, item_encoder.transform(item_encoder.classes_))}
    #     }
    #     try:
    #         with open(CF_MAPPINGS_OUTPUT_PATH, 'wb') as f:
    #             pickle.dump(cf_mappings, f)
    #         print(f"CF mappings saved to {CF_MAPPINGS_OUTPUT_PATH}")
    #     except Exception as e:
    #         print(f"Error saving CF mappings: {e}")

    #     # 2. CF: Data Splitting (Train/Validation)
    #     # The final "test" set (20% of original) is conceptually separate and handled by test_data.ipynb
    #     # Here, we split the remaining 80% into our training and validation sets.
    #     print("Splitting CF data into training and validation sets...")
    #     train_val_cf_df, _ = train_test_split(
    #         df_cf,
    #         test_size=CF_OVERALL_TEST_SPLIT, # This part is conceptually for the external test set
    #         random_state=RANDOM_STATE,
    #         stratify=df_cf[CF_RATING_COL] if df_cf[CF_RATING_COL].nunique() > 1 else None # Stratify if possible
    #     )
        
    #     # Now split train_val_cf_df into actual training and validation
    #     train_cf_df, val_cf_df = train_test_split(
    #         train_val_cf_df,
    #         test_size=CF_VALIDATION_SPLIT_FROM_TRAIN_VAL, # e.g., 0.2 of the 80%
    #         random_state=RANDOM_STATE,
    #         stratify=train_val_cf_df[CF_RATING_COL] if train_val_cf_df[CF_RATING_COL].nunique() > 1 else None
    #     )
    #     print(f"CF Train set size: {len(train_cf_df)}")
    #     print(f"CF Validation set size: {len(val_cf_df)}")

    #     train_cf_dataset = CFRatingsDataset(
    #         torch.tensor(train_cf_df['user_encoded'].values, dtype=torch.long),
    #         torch.tensor(train_cf_df['item_encoded'].values, dtype=torch.long),
    #         torch.tensor(train_cf_df[CF_RATING_COL].values, dtype=torch.float)
    #     )
    #     val_cf_dataset = CFRatingsDataset(
    #         torch.tensor(val_cf_df['user_encoded'].values, dtype=torch.long),
    #         torch.tensor(val_cf_df['item_encoded'].values, dtype=torch.long),
    #         torch.tensor(val_cf_df[CF_RATING_COL].values, dtype=torch.float)
    #     )
    #     train_cf_loader = DataLoader(train_cf_dataset, batch_size=CF_BATCH_SIZE, shuffle=True)
    #     val_cf_loader = DataLoader(val_cf_dataset, batch_size=CF_BATCH_SIZE, shuffle=False)

    #     # 3. CF: Model, Optimizer, Criterion
    #     cf_model = CFModel(n_users, n_items, CF_EMBEDDING_DIM).to(DEVICE)
    #     cf_criterion = nn.MSELoss()
    #     cf_optimizer = optim.Adam(cf_model.parameters(), lr=CF_LEARNING_RATE, weight_decay=CF_WEIGHT_DECAY)

    #     # 4. CF: Training Loop with Early Stopping and Best Model Saving
    #     print("Training CF model...")
    #     best_val_rmse = float('inf')
    #     best_model_state = None
    #     epochs_no_improve = 0
    #     patience = 5 # Number of epochs to wait for improvement before stopping

    #     for epoch in range(CF_EPOCHS):
    #         cf_model.train()
    #         total_train_loss = 0
    #         for users, items, ratings in train_cf_loader:
    #             users, items, ratings = users.to(DEVICE), items.to(DEVICE), ratings.to(DEVICE)
    #             cf_optimizer.zero_grad()
    #             outputs = cf_model(users, items)
    #             loss = cf_criterion(outputs, ratings)
    #             loss.backward()
    #             cf_optimizer.step()
    #             total_train_loss += loss.item() * users.size(0)
            
    #         avg_train_loss = total_train_loss / len(train_cf_dataset)
    #         train_rmse = math.sqrt(avg_train_loss)

    #         # Validation
    #         cf_model.eval()
    #         total_val_loss = 0
    #         with torch.no_grad():
    #             for users, items, ratings in val_cf_loader:
    #                 users, items, ratings = users.to(DEVICE), items.to(DEVICE), ratings.to(DEVICE)
    #                 outputs = cf_model(users, items)
    #                 loss = cf_criterion(outputs, ratings)
    #                 total_val_loss += loss.item() * users.size(0)
            
    #         avg_val_loss = total_val_loss / len(val_cf_dataset)
    #         val_rmse = math.sqrt(avg_val_loss)

    #         print(f"CF Epoch {epoch+1}/{CF_EPOCHS} - Train RMSE: {train_rmse:.4f}, Val RMSE: {val_rmse:.4f}")

    #         if epoch == 0 or epoch == CF_EPOCHS - 1:
    #             print("Sample outputs:", outputs[:10].detach().cpu().numpy())

    #         if val_rmse < best_val_rmse:
    #             best_val_rmse = val_rmse
    #             best_model_state = copy.deepcopy(cf_model.state_dict()) # Save copy of best model state
    #             epochs_no_improve = 0
    #             print(f"  -> New best Val RMSE: {best_val_rmse:.4f}. Saving model state.")
    #         else:
    #             epochs_no_improve += 1
            
    #         if epochs_no_improve >= patience:
    #             print(f"Early stopping triggered after {patience} epochs with no improvement.")
    #             break
        
    #     # 5. CF: Save Best Model
    #     if best_model_state:
    #         try:
    #             torch.save(best_model_state, CF_MODEL_OUTPUT_PATH)
    #             print(f"Best CF model (Val RMSE: {best_val_rmse:.4f}) saved to {CF_MODEL_OUTPUT_PATH}")
    #         except Exception as e:
    #             print(f"Error saving best CF model: {e}")
    #     else:
    #         print("No best CF model state found to save (e.g., training didn't improve or complete).")
    #     print("--- CF Model Training Finished ---")

if __name__ == '__main__':
    main() 