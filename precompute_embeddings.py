#to generate item_embeddings.pt
import torch
import pandas as pd
import time
import logging
import os
from model import get_bert_embeddings, device 

# --- Configuration --- 
INPUT_CSV_PATH = 'cleaned_merged.csv'
ITEM_ID_COLUMN = 'ProductId'
TEXT_COLUMN = 'CleanedSummary'
OUTPUT_EMBEDDINGS_FILE = 'item_embeddings.pt'
BATCH_SIZE = 32 
LOG_EVERY_N_BATCHES = 10 

def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    print(f"Using device: {device}")
    print(f"Loading item data from: {INPUT_CSV_PATH}")

    try:
        item_data_df = pd.read_csv(INPUT_CSV_PATH)
        if ITEM_ID_COLUMN not in item_data_df.columns or TEXT_COLUMN not in item_data_df.columns:
            raise KeyError(f"Missing required columns: '{ITEM_ID_COLUMN}' or '{TEXT_COLUMN}'")
        
        item_text_map = item_data_df.set_index(ITEM_ID_COLUMN)[TEXT_COLUMN].fillna('').to_dict()
        print(f"Loaded text data for {len(item_text_map)} items.")

    except FileNotFoundError:
        logging.error(f"Input CSV file not found at {INPUT_CSV_PATH}")
        return
    except KeyError as e:
        logging.error(f"KeyError during data loading: {e}")
        return
    except Exception as e:
        logging.error(f"An unexpected error occurred during data loading: {e}", exc_info=True)
        return

    item_ids = list(item_text_map.keys())
    item_texts = [item_text_map[item_id] for item_id in item_ids]

    item_embeddings_map = {}
    total_items = len(item_ids)
    total_batches = (total_items + BATCH_SIZE - 1) // BATCH_SIZE
    
    print(f"Starting BERT embedding generation for {total_items} items in {total_batches} batches...")
    start_time = time.time()
    
    # Process in batches without tqdm
    for batch_num, i in enumerate(range(0, total_items, BATCH_SIZE)):
        if batch_num % LOG_EVERY_N_BATCHES == 0 or batch_num == total_batches -1:
            print(f"Processing batch {batch_num + 1}/{total_batches} (Item {i} to {min(i + BATCH_SIZE, total_items)})")

        batch_ids = item_ids[i : i + BATCH_SIZE]
        batch_texts = item_texts[i : i + BATCH_SIZE]
        
        valid_indices = [idx for idx, txt in enumerate(batch_texts) if txt and isinstance(txt, str)]
        valid_batch_ids = [batch_ids[idx] for idx in valid_indices]
        valid_batch_texts = [batch_texts[idx] for idx in valid_indices]

        if not valid_batch_texts:
            logging.warning(f"Skipping batch {batch_num + 1} (starting at item index {i}) due to all texts being empty or invalid.")
            continue
            
        try:
            batch_embeddings = get_bert_embeddings(valid_batch_texts, batch_size=len(valid_batch_texts))
            for item_id, embedding in zip(valid_batch_ids, batch_embeddings):
                item_embeddings_map[item_id] = embedding.cpu()
        except Exception as e:
            logging.error(f"Error processing batch {batch_num + 1} (starting at item index {i}): {e}", exc_info=True)
            # Decide if you want to stop or continue. For now, it continues.

    end_time = time.time()
    duration = end_time - start_time
    print(f"\nFinished generating embeddings in {duration:.2f} seconds.")

    if not item_embeddings_map:
        logging.warning("No embeddings were generated. The output file will be empty or contain an empty map.")

    try:
        print(f"Saving embeddings to {OUTPUT_EMBEDDINGS_FILE}...")
        torch.save(item_embeddings_map, OUTPUT_EMBEDDINGS_FILE)
        print("Embeddings saved successfully.")
        # Verify file size as a basic check
        if os.path.exists(OUTPUT_EMBEDDINGS_FILE):
            file_size_kb = os.path.getsize(OUTPUT_EMBEDDINGS_FILE) / 1024
            if file_size_kb < 10 and total_items > 0: # Threshold in KB, e.g. 10KB
                logging.warning(f"Output file {OUTPUT_EMBEDDINGS_FILE} is very small ({file_size_kb:.2f} KB), which might indicate an issue.")
            else:
                logging.info(f"Output file {OUTPUT_EMBEDDINGS_FILE} size: {file_size_kb:.2f} KB")

    except Exception as e:
        logging.error(f"Error saving embeddings: {e}", exc_info=True)

if __name__ == "__main__":
    main()