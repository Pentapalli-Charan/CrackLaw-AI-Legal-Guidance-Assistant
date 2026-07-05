import logging
from typing import Dict, List, Tuple
from collections import Counter
from src.llm.tokenizer.vocabulary import Vocabulary

logger = logging.getLogger("CrackLaw.LLM.TokenizerStatistics")

class TokenizerStatistics:
    """Generates comprehensive statistics about the tokenizer and its performance."""
    
    @staticmethod
    def generate_report(
        vocab: Vocabulary, 
        merges: Dict[Tuple[str, str], int], 
        original_texts: List[str], 
        encoded_texts: List[List[int]]
    ) -> Dict[str, any]:
        
        vocab_size = len(vocab)
        
        # Calculate compression ratio
        total_original_chars = sum(len(text) for text in original_texts)
        total_tokens = sum(len(tokens) for tokens in encoded_texts)
        compression_ratio = total_original_chars / total_tokens if total_tokens > 0 else 0
        
        avg_tokens_per_doc = total_tokens / len(encoded_texts) if encoded_texts else 0
        
        # Most frequent tokens (excluding special tokens or single chars if desired, but we'll include all)
        token_counts = Counter()
        for tokens in encoded_texts:
            for token_id in tokens:
                token_counts[token_id] += 1
                
        most_frequent_ids = token_counts.most_common(10)
        most_frequent = [(vocab.get_token(tid), count) for tid, count in most_frequent_ids]
        
        # Longest token
        longest_token = ""
        for token in vocab.get_mapping().keys():
            if len(token) > len(longest_token) and not vocab.special_tokens.is_special(token):
                longest_token = token
                
        stats = {
            "vocabulary_size": vocab_size,
            "number_of_merges": len(merges),
            "total_documents_processed": len(original_texts),
            "average_tokens_per_document": round(avg_tokens_per_doc, 2),
            "compression_ratio": round(compression_ratio, 2),
            "longest_token_learned": longest_token,
            "most_frequent_tokens": most_frequent
        }
        
        return stats
        
    @staticmethod
    def print_report(stats: Dict[str, any]) -> str:
        report = "====================================================\n"
        report += "               TOKENIZER STATISTICS                 \n"
        report += "====================================================\n"
        for key, value in stats.items():
            friendly_key = key.replace("_", " ").title()
            report += f"{friendly_key}: {value}\n"
        report += "====================================================\n"
        logger.info("\n" + report)
        return report
