import os
import logging
import time
import torch
from src.llm.dataset.config import DatasetConfig
from src.llm.dataset.dataset import CrackLawDataset
from src.llm.dataset.dataloader import create_dataloaders
from src.llm.dataset.statistics import DatasetStatistics
from src.llm.tokenizer.tokenizer import CrackLawTokenizer

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("CrackLaw.LLM.RunDataset")

def main():
    t_start = time.time()
    logger.info("Initializing PyTorch Dataset and DataLoader Pipeline...")
    
    # 1. Initialize Tokenizer (Load pre-trained model)
    tokenizer = CrackLawTokenizer()
    try:
        tokenizer.load()
        logger.info(f"Loaded Tokenizer. Vocab Size: {tokenizer.get_vocab_size()}")
    except FileNotFoundError:
        logger.error("Tokenizer models not found. Please run the tokenizer pipeline first.")
        return
        
    pad_token_id = tokenizer.special_tokens.get_id(tokenizer.config.pad_token)
    
    # 2. Configure Dataset
    config = DatasetConfig(
        max_sequence_length=128, # Set smaller for faster demonstration
        batch_size=4,
        num_workers=0 # Stable for Windows/local
    )
    
    corpus_path = os.path.join("datasets", "corpus", "cracklaw_corpus.jsonl")
    if not os.path.exists(corpus_path):
        logger.error(f"Corpus file not found at {corpus_path}.")
        return
        
    # 3. Create Dataset
    dataset = CrackLawDataset(corpus_path, config, tokenizer)
    
    # 4. Create DataLoaders
    train_loader, val_loader, test_loader = create_dataloaders(dataset, config, pad_token_id)
    
    # 5. Generate and Print Statistics
    stats = DatasetStatistics.generate_report(train_loader, val_loader, test_loader, tokenizer.get_vocab_size())
    DatasetStatistics.print_report(stats)
    
    # 6. Display Example Batch and Tensors
    logger.info("--- EXAMPLE BATCH INSPECTION ---")
    
    for batch in train_loader:
        input_ids = batch["input_ids"]
        labels = batch["labels"]
        attention_mask = batch["attention_mask"]
        
        logger.info(f"Input IDs Shape     : {input_ids.shape} -> (batch_size, seq_len)")
        logger.info(f"Labels Shape        : {labels.shape} -> (batch_size, seq_len)")
        logger.info(f"Attention Mask Shape: {attention_mask.shape} -> (batch_size, seq_len)")
        
        # Display memory usage for the batch
        total_elements = input_ids.numel() + labels.numel() + attention_mask.numel()
        mem_mb = (total_elements * 8) / (1024 * 1024) # Assuming 64-bit int
        logger.info(f"Batch Memory Footprint: {mem_mb:.4f} MB")
        
        logger.info("\n--- SAMPLE DECODING (First Item in Batch) ---")
        # Exclude padding for decoding to make it readable
        first_input = [t.item() for t in input_ids[0] if t.item() != pad_token_id]
        logger.info(f"Decoded Input: {tokenizer.decode(first_input)}")
        
        break # Only show the first batch

    elapsed = time.time() - t_start
    logger.info(f"Dataset pipeline completed successfully in {elapsed:.2f} seconds.")

if __name__ == "__main__":
    main()
