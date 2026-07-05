import os
import json
import logging
import time
from src.llm.tokenizer.config import TokenizerConfig
from src.llm.tokenizer.tokenizer import CrackLawTokenizer
from src.llm.tokenizer.statistics import TokenizerStatistics

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("CrackLaw.LLM.RunTokenizer")

def main():
    t_start = time.time()
    logger.info("Initializing Custom BPE Tokenizer Pipeline...")
    
    # 1. Load corpus texts
    corpus_path = os.path.join("datasets", "corpus", "cracklaw_corpus.jsonl")
    if not os.path.exists(corpus_path):
        logger.error(f"Corpus file not found at {corpus_path}. Please run corpus preparation pipeline first.")
        return
        
    texts = []
    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            texts.append(record.get("text", ""))
            
    if not texts:
        logger.error("No text found in corpus.")
        return
        
    # We set vocab size extremely small (e.g. 100) for testing if the corpus is very small.
    # Otherwise set it to 5000.
    target_vocab = min(5000, 50 + len("".join(texts)) // 10)
    config = TokenizerConfig(vocab_size=target_vocab)
    
    tokenizer = CrackLawTokenizer(config)
    
    # 2. Train tokenizer
    tokenizer.train(texts)
    
    # 3. Save tokenizer
    logger.info("Saving tokenizer artifacts...")
    tokenizer.save()
    
    # 4. Generate Statistics
    logger.info("Encoding full corpus to calculate statistics...")
    encoded_texts = [tokenizer.encode(text) for text in texts]
    
    stats = TokenizerStatistics.generate_report(
        tokenizer.vocab, 
        tokenizer.merges, 
        texts, 
        encoded_texts
    )
    TokenizerStatistics.print_report(stats)
    
    # 5. Example encoding/decoding
    example_text = "Section 1. Short title, extent and commencement."
    logger.info("--- EXAMPLE ENCODING / DECODING ---")
    logger.info(f"Original Text : {example_text}")
    
    encoded_example = tokenizer.encode(example_text)
    logger.info(f"Token IDs     : {encoded_example}")
    
    decoded_example = tokenizer.decode(encoded_example)
    logger.info(f"Decoded Text  : {decoded_example}")
    
    elapsed = time.time() - t_start
    logger.info(f"Tokenizer pipeline completed successfully in {elapsed:.2f} seconds.")

if __name__ == "__main__":
    main()
