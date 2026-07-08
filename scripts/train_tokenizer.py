import os
import sys
import json
import logging

sys.path.insert(0, os.path.abspath('.'))
from src.llm.tokenizer.config import TokenizerConfig
from src.llm.tokenizer.tokenizer import CrackLawTokenizer

logging.basicConfig(level=logging.INFO)

def main():
    corpus_path = "datasets/corpus/cracklaw_corpus.jsonl"
    texts = []
    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            if "text" in data:
                texts.append(data["text"])
    
    print(f"Loaded {len(texts)} documents for tokenizer training.")
    
    config = TokenizerConfig()
    tokenizer = CrackLawTokenizer(config)
    print("Training tokenizer...")
    tokenizer.train(texts)
    tokenizer.save()
    print(f"Tokenizer trained and saved. Final vocab size: {tokenizer.get_vocab_size()}")

if __name__ == "__main__":
    main()
