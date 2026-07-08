import os
import sys
import time
import logging
import torch

from src.config import PROJECT_ROOT
from src.llm.transformer.config import TransformerConfig
from src.llm.transformer.model import CrackLawTransformer
from src.llm.transformer.generation import TransformerGenerator
from src.llm.tokenizer.tokenizer import CrackLawTokenizer
from src.llm.tokenizer.config import TokenizerConfig
from src.llm.training.checkpoint_manager import CheckpointManager
from src.llm.training.config import TrainingConfig
from src.llm.transformer.masks import AttentionMasks

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("CrackLaw.Inference")

def main():
    logger.info("Initializing CrackLaw Inference Pipeline...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # 1. Load Tokenizer
    tokenizer_config = TokenizerConfig()
    tokenizer = CrackLawTokenizer(tokenizer_config)
    try:
        tokenizer.load()
        logger.info(f"Loaded Tokenizer. Vocab size: {tokenizer.get_vocab_size()}")
    except FileNotFoundError:
        logger.error("Tokenizer not found! Please run training or tokenizer generation first.")
        return

    # 2. Get latest checkpoint to determine model config
    training_config = TrainingConfig()
    checkpoint_manager = CheckpointManager(training_config)
    
    ckpt_path = checkpoint_manager.get_best_checkpoint()
    if not ckpt_path:
        ckpt_path = checkpoint_manager.get_latest_checkpoint()
        
    if not ckpt_path:
        logger.error("No trained checkpoints found! Please run train.py first.")
        return
        
    logger.info(f"Found checkpoint: {ckpt_path}")
    
    # Extract metadata from checkpoint to reconstruct identical model
    # (In a real system, you'd save the TransformerConfig with the checkpoint.
    # We'll use the verification config we used in train.py for now)
    transformer_config = TransformerConfig(
        vocab_size=tokenizer.get_vocab_size(),
        d_model=128,          
        num_heads=4,
        d_ff=512,
        num_encoder_layers=2,
        num_decoder_layers=2
    )
    
    # 3. Initialize & Load Model
    model = CrackLawTransformer(transformer_config)
    
    logger.info("Loading model weights...")
    try:
        checkpoint_manager.load(ckpt_path, model, device=device)
    except Exception as e:
        logger.error(f"Failed to load checkpoint: {e}")
        return
        
    model.to(device)
    model.eval()
    logger.info("Model loaded successfully.")

    generator = TransformerGenerator(model)
    bos_id = tokenizer.special_tokens.get_id(tokenizer.config.bos_token)
    eos_id = tokenizer.special_tokens.get_id(tokenizer.config.eos_token)
    pad_id = tokenizer.special_tokens.get_id(tokenizer.config.pad_token)

    print("\n" + "="*50)
    print("CrackLawLM Interactive Inference")
    print("Type 'quit' or 'exit' to stop.")
    print("="*50 + "\n")

    while True:
        try:
            prompt = input("Prompt: ")
            if prompt.strip().lower() in ["quit", "exit"]:
                break
            if not prompt.strip():
                continue
                
            # Tokenize
            token_ids = tokenizer.encode(prompt)
            input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
            
            # Create padding mask for source
            padding_mask = (input_ids != pad_id).unsqueeze(1).unsqueeze(2).to(device)

            print("Generating...", end="", flush=True)
            t0 = time.time()
            
            # Generate
            generated_ids = generator.greedy_search(
                src_input_ids=input_ids,
                src_padding_mask=padding_mask,
                max_new_tokens=50, # Keep short for REPL
                start_token_id=bos_id,
                end_token_id=eos_id
            )
            
            t1 = time.time()
            
            # Remove BOS from generated output
            out_ids = generated_ids[0].tolist()
            if out_ids and out_ids[0] == bos_id:
                out_ids = out_ids[1:]
                
            # Decode
            result = tokenizer.decode(out_ids)
            
            # Stats
            time_taken = t1 - t0
            num_tokens = len(out_ids)
            tokens_per_sec = num_tokens / time_taken if time_taken > 0 else 0
            
            print(f"\nResponse: {result}")
            print(f"[{num_tokens} tokens generated in {time_taken:.2f}s ({tokens_per_sec:.2f} tok/s)]\n")
            
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            logger.error(f"Error during generation: {e}")

if __name__ == "__main__":
    main()
