import os
import logging
from src.config import PROJECT_ROOT
from src.llm.transformer.config import TransformerConfig
from src.llm.transformer.model import CrackLawTransformer
from src.llm.tokenizer.tokenizer import CrackLawTokenizer
from src.llm.tokenizer.config import TokenizerConfig
from src.llm.dataset.dataset import CrackLawDataset
from src.llm.dataset.config import DatasetConfig
from src.llm.dataset.dataloader import create_dataloaders
from src.llm.training.config import TrainingConfig
from src.llm.training.trainer import CrackLawTrainer
from src.llm.evaluation.config import EvaluationConfig
from src.llm.evaluation.evaluation_engine import EvaluationEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("CrackLaw.Train")

def main():
    logger.info("Initializing CrackLaw Training Pipeline...")
    
    # 1. Load Tokenizer
    tokenizer_config = TokenizerConfig()
    tokenizer = CrackLawTokenizer(tokenizer_config)
    try:
        tokenizer.load()
        logger.info(f"Loaded Tokenizer. Vocab size: {tokenizer.get_vocab_size()}")
    except FileNotFoundError:
        logger.error("Tokenizer not found! Please run the tokenizer training pipeline first.")
        return

    # 2. Setup Dataset & DataLoaders
    dataset_config = DatasetConfig(
        batch_size=2,          # Small batch for verification
        max_sequence_length=128 # Small sequence for verification
    )
    corpus_path = os.path.join(PROJECT_ROOT, "datasets", "corpus", "cracklaw_corpus.jsonl")
    if not os.path.exists(corpus_path):
        logger.error(f"Corpus file missing at: {corpus_path}")
        return
        
    dataset = CrackLawDataset(corpus_path, dataset_config, tokenizer)
    
    pad_token_id = tokenizer.special_tokens.get_id(tokenizer.config.pad_token)
    train_loader, val_loader, _ = create_dataloaders(dataset, dataset_config, pad_token_id)
    logger.info(f"Created DataLoaders. Train size: {len(train_loader)}, Val size: {len(val_loader)}")

    # 3. Initialize Transformer
    transformer_config = TransformerConfig(
        vocab_size=tokenizer.get_vocab_size(),
        d_model=128,          # Scaled down for quick verification
        num_heads=4,
        d_ff=512,
        num_encoder_layers=2,
        num_decoder_layers=2
    )
    model = CrackLawTransformer(transformer_config)
    logger.info("Initialized CrackLawTransformer.")

    # 4. Setup Training Engine
    training_config = TrainingConfig(
        num_epochs=200,         # Large number to overfit targeted responses
        learning_rate=3e-4,
        optimizer_type="adamw",
        scheduler_type="cosine",
        warmup_steps=10,
        early_stopping_enabled=False
    )
    
    # 5. Setup Evaluation Engine as Callback
    eval_config = EvaluationConfig(
        generation_enabled=True,
        benchmark_enabled=False
    )
    eval_engine = EvaluationEngine(eval_config, tokenizer)
    
    # We pass the eval engine callback to the trainer. Note that we need to pass the device, which trainer manages.
    # To get around this cleanly, we can instantiate the Trainer, get its device, and then add the callback.
    trainer = CrackLawTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        training_config=training_config,
        callbacks=[] # We will add the eval callback after getting the device
    )
    
    eval_callback = eval_engine.as_callback(model, val_loader, trainer.device)
    trainer.callback_manager.add(eval_callback)

    # 6. Train
    logger.info("Starting training loop...")
    trainer.train()
    
    logger.info("Training complete.")

if __name__ == "__main__":
    main()
