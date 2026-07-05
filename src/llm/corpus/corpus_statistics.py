import json
import logging
from typing import List, Dict, Any
from collections import defaultdict
from src.llm.corpus.metadata import CorpusDocument

logger = logging.getLogger("CrackLaw.LLM.CorpusStatistics")

class CorpusStatistics:
    """Calculates and generates reports on the legal corpus."""
    
    def __init__(self):
        pass

    def generate_report(self, documents: List[CorpusDocument]) -> Dict[str, Any]:
        """Calculates statistics for the validated corpus."""
        
        total_docs = len(documents)
        if total_docs == 0:
            return {"error": "Corpus is empty"}
            
        total_words = 0
        acts_count = 0
        judgments_count = 0
        
        largest_doc = {"id": None, "words": -1}
        smallest_doc = {"id": None, "words": float('inf')}
        
        vocab_set = set()
        
        for doc in documents:
            words = doc.word_count
            total_words += words
            
            if doc.doc_type == "laws" or doc.act:
                acts_count += 1
            if doc.doc_type == "judgments" or doc.judgment_title:
                judgments_count += 1
                
            if words > largest_doc["words"]:
                largest_doc = {"id": doc.document_id, "words": words}
                
            if words < smallest_doc["words"]:
                smallest_doc = {"id": doc.document_id, "words": words}
                
            # Naive vocabulary estimation (unique lowercased tokens)
            # We don't want to overcomplicate with full tokenization per requirements
            tokens = doc.text.lower().split()
            vocab_set.update(tokens)
            
        avg_words = total_words / total_docs if total_docs > 0 else 0
        
        stats = {
            "total_documents": total_docs,
            "number_of_acts": acts_count,
            "number_of_judgments": judgments_count,
            "total_words": total_words,
            "average_words_per_document": round(avg_words, 2),
            "vocabulary_estimate": len(vocab_set),
            "largest_document": largest_doc,
            "smallest_document": smallest_doc
        }
        
        return stats
        
    def print_report(self, stats: Dict[str, Any]) -> str:
        """Formats the statistics into a readable string."""
        report = "====================================================\n"
        report += "               CORPUS STATISTICS REPORT               \n"
        report += "====================================================\n"
        for key, value in stats.items():
            friendly_key = key.replace("_", " ").title()
            report += f"{friendly_key}: {value}\n"
        report += "====================================================\n"
        logger.info("\n" + report)
        return report
