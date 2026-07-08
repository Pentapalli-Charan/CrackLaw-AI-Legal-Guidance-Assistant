import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath('.'))

from src.llm.tokenizer.config import TokenizerConfig
from src.llm.tokenizer.tokenizer import CrackLawTokenizer
from src.retrieval.retrieval_service import RetrievalService
import logging
logging.basicConfig(level=logging.ERROR)

def test_tokenizer():
    print("--- TOKENIZER ---")
    config = TokenizerConfig()
    tok = CrackLawTokenizer(config)
    tok.load()
    print("Vocab size:", tok.get_vocab_size())
    text = "Explain IPC Section 302"
    encoded = tok.encode(text)
    decoded = tok.decode(encoded)
    print("Original:", text)
    print("Decoded:", decoded)
    print("Matches:", text == decoded)
    print("Tokens:", len(encoded))
    unk_id = tok.special_tokens.get_id(tok.config.unk_token)
    unk_count = encoded.count(unk_id)
    print("UNK count:", unk_count)
    if len(text) > 0:
        print("Compression ratio (chars/token):", len(text) / len(encoded) if len(encoded) > 0 else 0)
    print()

def test_retrieval():
    print("--- RETRIEVAL ---")
    rs = RetrievalService()
    rs.initialize()
    queries = ["Hi", "Explain IPC Section 302", "Contract"]
    for q in queries:
        print(f"Query: {q}")
        try:
            results = rs.search(q, top_k=2)
            print(f"Found {len(results)} results")
            for r in results:
                print(f" - score: {r.score}, snippet: {r.document.text[:50]}")
        except Exception as e:
            print(f"Retrieval failed: {e}")
    print()

if __name__ == '__main__':
    try:
        test_tokenizer()
    except Exception as e:
        print(f"Tokenizer error: {e}")
    try:
        test_retrieval()
    except Exception as e:
        print(f"Retrieval error: {e}")
