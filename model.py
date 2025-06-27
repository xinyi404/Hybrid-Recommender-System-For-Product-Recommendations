# Import necessary libraries
import torch
import pickle
import torch.nn as nn
from transformers import BertTokenizer, BertModel, BertForSequenceClassification
import numpy as np
import traceback 
import os
from typing import Tuple, List, Dict, Optional
import logging
import random

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration --- 
CF_MODEL_PATH = 'cf_model.pth'
CF_MAPPINGS_PATH = 'cf_mappings.pkl' 
ITEM_EMBEDDINGS_PATH = 'item_embeddings.pt' 
CBF_MODEL_PATH = 'cbf_model.pth' 
BERT_MODEL_NAME = 'bert-base-uncased'
CF_EMBEDDING_DIM = 512  # Centralized configuration
CBF_NUM_CLASSES = 3 # Positive, Neutral, Negative as per 8020TrainTest.ipynb


# --- Device Setup --- 
# Use a single source of truth for device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Model script using device: {device}")

# --- Global Variables --- 
cf_model: Optional[nn.Module] = None
cbf_model: Optional[BertForSequenceClassification] = None
item_embeddings: Optional[Dict[str, torch.Tensor]] = None
user_id_to_idx: Optional[Dict[str, int]] = None
item_id_to_idx: Optional[Dict[str, int]] = None
idx_to_item_id: Optional[Dict[int, str]] = None
num_users: int = 0
num_items: int = 0
bert_model: Optional[BertModel] = None
tokenizer: Optional[BertTokenizer] = None
bert_embedding_dim: int = 768

# For Hugging Face pipeline sentiment analysis
sentiment_analyzer = None

# --- Helper: BERT Model and Tokenizer for get_bert_embeddings ---
# These are specifically for the get_bert_embeddings function to keep it self-contained
# and avoid conflicts if the main app's bert_model/tokenizer are different or not yet loaded.
_embedding_tokenizer: Optional[BertTokenizer] = None
_embedding_model: Optional[BertModel] = None

def get_bert_embeddings(texts: List[str], batch_size: int = 32) -> torch.Tensor:
    global _embedding_tokenizer, _embedding_model

    if not _embedding_tokenizer or not _embedding_model:
        logging.info("Initializing dedicated BERT model and tokenizer for get_bert_embeddings...")
        try:
            _embedding_tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)
            _embedding_model = BertModel.from_pretrained(BERT_MODEL_NAME).to(device)
            _embedding_model.eval()  # Set to evaluation mode
            logging.info("Dedicated BERT model and tokenizer for get_bert_embeddings initialized successfully.")
        except Exception as e:
            logging.error(f"Error initializing dedicated BERT model/tokenizer for get_bert_embeddings: {e}")
            raise

    if not texts:
        return torch.empty(0, _embedding_model.config.hidden_size if _embedding_model else 768).cpu()

    try:
        # Ensure tokenizer and model are loaded
        if not _embedding_tokenizer or not _embedding_model:
            raise RuntimeError("Embedding tokenizer or model not initialized.")
            
        inputs = _embedding_tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=512).to(device)
        with torch.no_grad():
            outputs = _embedding_model(**inputs)
        
        # Mean pooling of the last hidden state
        embeddings = outputs.last_hidden_state.mean(dim=1)
        return embeddings.cpu()
    except Exception as e:
        logging.error(f"Error during BERT embedding generation: {e}")
        # Return an empty tensor of appropriate shape on error, or re-raise
        # For precompute_embeddings.py, it might be better to raise to halt the process
        raise

# --- File Validation Function ---
def validate_required_files() -> bool:
    required_files = {
        'CF Model': CF_MODEL_PATH,
        'CF Mappings': CF_MAPPINGS_PATH,
        'Item Embeddings': ITEM_EMBEDDINGS_PATH
        # CBF_MODEL_PATH is optional here, handled in load_models_and_data
    }
    
    missing_files = [f"{name} ({path})" for name, path in required_files.items() 
                    if not os.path.exists(path)]
    
    if missing_files:
        print("Error: The following required files are missing:")
        print("\n".join(f"- {file}" for file in missing_files))
        return False
    return True

# --- Load BERT Model and Tokenizer Globally --- 
def load_bert_model() -> bool:
    global bert_model, tokenizer, bert_embedding_dim
    
    try:
        print(f"Loading BERT tokenizer and model: {BERT_MODEL_NAME}...")
        tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)
        bert_model = BertModel.from_pretrained(BERT_MODEL_NAME).to(device)
        bert_model.eval()
        bert_embedding_dim = bert_model.config.hidden_size
        print("BERT model loaded successfully.")
        return True
    except Exception as e:
        print(f"Error loading BERT model: {e}")
        return False

# --- Define CF Model Class (MUST match the one in train_cf.ipynb) ---
class CFModel(nn.Module):
    def __init__(self, n_users: int, n_items: int, embedding_dim: int):
        super().__init__()
        # Ensure indices are non-zero before creating embeddings
        self.user_embeddings = nn.Embedding(max(n_users, 1), embedding_dim)
        self.item_embeddings = nn.Embedding(max(n_items, 1), embedding_dim)
        self.user_bias = nn.Embedding(max(n_users, 1), 1)
        self.item_bias = nn.Embedding(max(n_items, 1), 1)
        
        # Initialize weights
        nn.init.normal_(self.user_embeddings.weight, std=0.01)
        nn.init.normal_(self.item_embeddings.weight, std=0.01)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)

    def forward(self, user_indices: torch.Tensor, item_indices: torch.Tensor, apply_sigmoid: bool = False) -> torch.Tensor:
        user_embedding = self.user_embeddings(user_indices)
        item_embedding = self.item_embeddings(item_indices)
        dot_product = (user_embedding * item_embedding).sum(1)
        user_b = self.user_bias(user_indices).squeeze()
        item_b = self.item_bias(item_indices).squeeze()

        prediction = dot_product + user_b + item_b

        if apply_sigmoid:
            # Scale to 1–5 range
            prediction = torch.sigmoid(prediction) * 4 + 1

        return prediction
    
# --- Define CBF Model (BERT Classifier) --- 
# Keep this class definition if your hybrid model still uses it
class CBFModel(nn.Module):
    # Based on 8020TrainTest.ipynb (BERT_Classifier)
    def __init__(self, input_size: int = bert_embedding_dim, num_classes: int = 3):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 512)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3) 
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        x = self.fc1(embeddings)
        x = self.relu(x)
        x = self.dropout(x)
        return self.fc2(x)


# --- Central Loading Function --- 
def load_models_and_data() -> bool:
    """Load all required models and data."""
    global cf_model, cbf_model, item_embeddings, user_id_to_idx, item_id_to_idx, idx_to_item_id, num_users, num_items, bert_embedding_dim
    
    logging.info("Attempting to load all models and data...")
    all_loaded_successfully = True

    try:
        # Load CF mappings first
        logging.info(f"Loading CF mappings from: {CF_MAPPINGS_PATH}")
        if not os.path.exists(CF_MAPPINGS_PATH):
            logging.error(f"CF Mappings file not found: {CF_MAPPINGS_PATH}")
            return False # Critical failure
        with open(CF_MAPPINGS_PATH, 'rb') as f:
            mappings = pickle.load(f)
            user_id_to_idx = mappings['user_id_to_idx']
            item_id_to_idx = mappings['item_id_to_idx']
            idx_to_item_id = {v: k for k, v in item_id_to_idx.items()}
            num_users = len(user_id_to_idx)
            num_items = len(item_id_to_idx)
        logging.info("CF mappings loaded successfully.")

        # Load CF model
        logging.info(f"Loading CF model from: {CF_MODEL_PATH}")
        if not os.path.exists(CF_MODEL_PATH):
            logging.error(f"CF Model file not found: {CF_MODEL_PATH}")
            return False # Critical failure
        try:
            logging.info(f"Initializing CFModel with num_users={num_users}, num_items={num_items}, embedding_dim={CF_EMBEDDING_DIM}")
            cf_model = CFModel(num_users, num_items, CF_EMBEDDING_DIM)
            logging.info("Loading CF model state dict...")
            state_dict = torch.load(CF_MODEL_PATH, map_location=device)
            logging.info(f"State dict keys: {state_dict.keys()}")
            cf_model.load_state_dict(state_dict)
            cf_model.to(device)
            cf_model.eval()
            logging.info("CF model loaded successfully.")
        except Exception as e:
            logging.error(f"Error loading CF model: {e}", exc_info=True)
            return False

    except Exception as e:
        logging.error(f"CRITICAL ERROR loading CF model or mappings: {e}", exc_info=True)
        all_loaded_successfully = False
        # Return False here because CF is essential for core functions
        return False 

    try:
        # Load item embeddings
        logging.info(f"Loading item embeddings from: {ITEM_EMBEDDINGS_PATH}")
        if not os.path.exists(ITEM_EMBEDDINGS_PATH):
            logging.error(f"Item embeddings file not found: {ITEM_EMBEDDINGS_PATH}")
            item_embeddings = None # Explicitly set to None
            all_loaded_successfully = False
        else:
            try:
                # Try loading with map_location first
                item_embeddings = torch.load(ITEM_EMBEDDINGS_PATH, map_location=device)
            except RuntimeError as e:
                if "Tried to instantiate class" in str(e):
                    # If we get the class registration error, try loading with pickle
                    logging.warning("Got PyTorch class registration error, trying to load with pickle...")
                    try:
                        with open(ITEM_EMBEDDINGS_PATH, 'rb') as f:
                            item_embeddings = pickle.load(f)
                    except Exception as pickle_e:
                        logging.error(f"Failed to load embeddings with pickle: {pickle_e}")
                        item_embeddings = None
                        all_loaded_successfully = False
                else:
                    raise e

            if not isinstance(item_embeddings, dict) or not item_embeddings:
                logging.error("Item embeddings are not a valid dictionary or are empty.")
                item_embeddings = None # Explicitly set to None on invalid content
                all_loaded_successfully = False
            else:
                logging.info(f"Item embeddings loaded successfully for {len(item_embeddings)} items.")
                # Determine bert_embedding_dim from loaded embeddings if possible
                if item_embeddings and len(item_embeddings) > 0:
                    first_item_key = next(iter(item_embeddings))
                    embedding_shape = item_embeddings[first_item_key].shape
                    if embedding_shape: 
                        bert_embedding_dim = embedding_shape[0] if len(embedding_shape) == 1 else embedding_shape[-1]
                        logging.info(f"Inferred bert_embedding_dim from loaded embeddings: {bert_embedding_dim}")

    except Exception as e:
        logging.error(f"ERROR loading item embeddings: {e}", exc_info=True)
        item_embeddings = None # Ensure it's None on any error
        all_loaded_successfully = False

    try:
        # Load CBF model (trained BERT Classifier)
        logging.info(f"Attempting to load CBF model from: {CBF_MODEL_PATH}")
        if not os.path.exists(CBF_MODEL_PATH):
            logging.warning(f"CBF model file not found at {CBF_MODEL_PATH}. Hybrid model's CBF component will default to 0.")
            cbf_model = None
            # Not necessarily a critical failure for all_loaded_successfully if CBF is optional
        else:
            # Use the bert_embedding_dim possibly inferred from item_embeddings, or default
            actual_input_dim_for_cbf = bert_embedding_dim 
            logging.info(f"Initializing CBFModel with input_size: {actual_input_dim_for_cbf}")
            cbf_model = CBFModel(input_size=actual_input_dim_for_cbf, num_classes=CBF_NUM_CLASSES) # Pass num_classes
            cbf_model.load_state_dict(torch.load(CBF_MODEL_PATH, map_location=device))
            cbf_model.to(device)
            cbf_model.eval()
            logging.info("CBF model loaded successfully.")
            
    except Exception as e:
        logging.error(f"ERROR loading CBF model from {CBF_MODEL_PATH}: {e}. CBF features may be affected.", exc_info=True)
        cbf_model = None # Ensure it's None on any error
        # Not necessarily a critical failure for all_loaded_successfully if CBF is optional

    if all_loaded_successfully:
        logging.info("All models and data relevant for core functionality appear to be loaded.")
    else:
        logging.warning("One or more optional models/data components failed to load. Application functionality may be limited.")
        # Still return True if essential CF parts loaded, False if CF failed earlier
        # The function will have returned False already if CF parts failed.

    return True # Returns True if CF part loaded, even if optional parts (embeddings, CBF model) didn't.
                 # App needs to handle None for optional parts.


# --- Recommendation Functions --- 
def get_cf_recommendations(user_id_str: str, k: int = 10) -> Tuple[List[str], List[float]]:
    """Get collaborative filtering recommendations for a user."""
    if not isinstance(user_id_str, str) or not isinstance(k, int) or k <= 0:
        print(f"Invalid input: user_id_str must be string, k must be positive integer")
        return [], []

    if cf_model is None or user_id_to_idx is None or item_id_to_idx is None or idx_to_item_id is None:
        print("Error: CF model or mappings not loaded")
        return [], []

    if user_id_str not in user_id_to_idx:
        print(f"Warning: User ID '{user_id_str}' not found in mappings")
        return [], []

    try:
        user_idx = user_id_to_idx[user_id_str]
        user_idx_tensor = torch.tensor([user_idx], dtype=torch.long).to(device)
        all_item_indices = torch.arange(num_items, dtype=torch.long).to(device)
        user_indices_repeated = user_idx_tensor.repeat(num_items)

        cf_model.eval()
        with torch.no_grad():
            predicted_scores = cf_model(user_indices_repeated, all_item_indices)

        actual_k = min(k, num_items)
        if actual_k <= 0:
            return [], []

        top_scores, top_indices = torch.topk(predicted_scores, actual_k)
        
        recommended_items = []
        recommended_scores = []
        
        for idx, score in zip(top_indices, top_scores):
            idx_item = idx.item()
            if idx_item in idx_to_item_id:
                recommended_items.append(idx_to_item_id[idx_item])
                recommended_scores.append(score.item())
        
        return recommended_items, recommended_scores
        
    except Exception as e:
        print(f"Error generating CF recommendations: {e}")
        traceback.print_exc()
        return [], []

# --- get_cbf_recommendations (If needed) ---
# Define or update this function if your hybrid model uses separate CBF recommendations
# It should use the globally loaded cbf_model and item_embeddings

def get_cbf_recommendations(item_id_query: str, k: int = 10) -> Tuple[List[str], List[float]]:
    """Get content-based filtering recommendations for an item."""
    if not isinstance(item_id_query, str) or not isinstance(k, int) or k <= 0:
        print(f"Invalid input: item_id_query must be string, k must be positive integer")
        return [], []

    if item_embeddings is None or not item_embeddings:
        print("Error: Item embeddings not loaded")
        return [], []

    if item_id_query not in item_embeddings:
        print(f"Warning: Query item ID '{item_id_query}' not found in embeddings")
        return [], []

    try:
        query_embedding = item_embeddings[item_id_query]
        
        # Compute similarities in batch for efficiency
        embeddings = torch.stack([emb for item_id, emb in item_embeddings.items() 
                                if item_id != item_id_query])
        item_ids = [item_id for item_id in item_embeddings.keys() 
                   if item_id != item_id_query]
        
        with torch.no_grad():
            similarities = torch.nn.functional.cosine_similarity(
                query_embedding.unsqueeze(0), embeddings
            )
        
        # Get top k items
        top_scores, top_indices = torch.topk(similarities, min(k, len(item_ids)))
        
        return [item_ids[i] for i in top_indices], top_scores.tolist()
        
    except Exception as e:
        print(f"Error generating CBF recommendations: {e}")
        traceback.print_exc()
        return [], []


# --- Hybrid Recommendation Function --- 
# Ensure this function uses the globally loaded models and data
def get_hybrid_recommendations(
    user_id: str,
    item_ids: List[str],
    all_item_embeddings: Dict[str, torch.Tensor],
    n_recommendations: int = 100,
    diversity_weight: float = 0.1,
    progress_bar = None
) -> Tuple[List[str], List[float], List[float], List[float]]:
    if cf_model is None or user_id_to_idx is None:
        print("Error: Required CF models or data not loaded for hybrid recommendations.")
        return [], [], [], []
    
    # Fixed weights for CF and CBF
    CF_WEIGHT = 0.5
    CBF_WEIGHT = 0.5
    
    try:
        # Get CF scores
        cf_scores = {}
        if user_id in user_id_to_idx:
            user_idx = user_id_to_idx[user_id]
            user_idx_tensor = torch.tensor([user_idx], dtype=torch.long).to(device)
            
            for item_id in item_ids:
                if item_id in item_id_to_idx:
                    item_idx = item_id_to_idx[item_id]
                    item_idx_tensor = torch.tensor([item_idx], dtype=torch.long).to(device)
                    
                    with torch.no_grad():
                        score = cf_model(user_idx_tensor, item_idx_tensor, apply_sigmoid=True)
                        cf_scores[item_id] = score.item()
                else:
                    cf_scores[item_id] = 0.0
        
        # Get CBF scores using the loaded CBFModel
        cbf_scores = {}
        if cbf_model and all_item_embeddings:
            cbf_model.eval()
            for item_id in item_ids:
                if item_id in all_item_embeddings:
                    item_embedding = all_item_embeddings[item_id].to(device)
                    
                    if item_embedding.ndim == 1:
                        item_embedding = item_embedding.unsqueeze(0)
                    if item_embedding.ndim == 0:
                        print(f"Warning: Skipping 0-dim embedding for item {item_id}")
                        cbf_scores[item_id] = 0.0
                        continue
                    
                    expected_dim = cbf_model.fc1.in_features
                    if item_embedding.shape[-1] != expected_dim:
                        print(f"Warning: Embedding dim mismatch for item {item_id}")
                        cbf_scores[item_id] = 0.0
                        continue

                    with torch.no_grad():
                        cbf_model_output = cbf_model(item_embedding)
                        
                        # Apply temperature scaling
                        temperature = 1.5
                        scaled_output = cbf_model_output / temperature
                        probs = torch.softmax(scaled_output, dim=1)

                        # Weighted sentiment score
                        score = (probs[0, 0] * -1 + probs[0, 1] * 0 + probs[0, 2] * 1).item()
                    cbf_scores[item_id] = score
                else:
                    cbf_scores[item_id] = 0.0
        else:
            if not cbf_model:
                print("Warning: CBF model not loaded. CBF scores will be 0.")
            if not all_item_embeddings:
                print("Warning: Item embeddings not available. CBF scores will be 0.")
            for item_id in item_ids:
                cbf_scores[item_id] = 0.0

        # Normalize and combine scores
        def normalize_and_combine_scores(cf_scores: Dict[str, float], cbf_scores: Dict[str, float]) -> Dict[str, float]:
            if not cf_scores or not cbf_scores:
                return {}
                
            # Convert to tensors
            cf_tensor = torch.tensor(list(cf_scores.values()))
            cbf_tensor = torch.tensor(list(cbf_scores.values()))
            
            # Normalize CF scores from [1, 5] to [0, 1]
            cf_tensor = torch.clamp((cf_tensor - 1.0) / 4.0, 0, 0.98)
            # Normalize CBF scores from [-1,1] to [0,1] and clamp
            cbf_tensor = torch.clamp((cbf_tensor + 1) / 2.0, 0, 0.98)

            # Combine with fixed weights
            combined = (CF_WEIGHT * cf_tensor + CBF_WEIGHT * cbf_tensor)
            
            # Convert back to dictionary
            return {item_id: score.item() for item_id, score in zip(cf_scores.keys(), combined)}

        # Get combined scores
        hybrid_scores = normalize_and_combine_scores(cf_scores, cbf_scores)
        
        # Get diverse recommendations
        def get_diverse_recommendations(scores: Dict[str, float], n: int) -> List[str]:
            selected = []
            remaining = list(scores.items())
            
            while len(selected) < n and remaining:
                # Sort by score
                remaining.sort(key=lambda x: x[1], reverse=True)
                top_items = remaining[:100]  # Consider top 100 items
                
                # Add some randomness for diversity
                if random.random() < diversity_weight:
                    # Pick a random item from top 100
                    selected.append(top_items[random.randint(0, len(top_items)-1)][0])
                else:
                    # Pick the best item
                    selected.append(top_items[0][0])
                
                # Remove selected item
                remaining = [(i, s) for i, s in remaining if i != selected[-1]]
            
            return selected
        
        # Get diverse recommendations
        recommended_items = get_diverse_recommendations(hybrid_scores, n_recommendations)
        
        if not recommended_items:
            return [], [], [], []
        
        # Return items and their scores
        hybrid_scores_list = [hybrid_scores[item_id] for item_id in recommended_items]
        cf_scores_list = [cf_scores[item_id] for item_id in recommended_items]
        cbf_scores_list = [cbf_scores[item_id] for item_id in recommended_items]
        
        return recommended_items, hybrid_scores_list, cf_scores_list, cbf_scores_list
        
    except Exception as e:
        print(f"Error generating hybrid recommendations: {e}")
        traceback.print_exc()
        return [], [], [], []
    