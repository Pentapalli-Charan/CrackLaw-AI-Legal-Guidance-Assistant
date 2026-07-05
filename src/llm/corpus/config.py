import os
from dataclasses import dataclass, field
from src.config import PROJECT_ROOT

@dataclass
class CorpusConfig:
    """Configuration for Legal Corpus Preparation pipeline."""
    
    # Paths
    datasets_dir: str = os.path.join(PROJECT_ROOT, "datasets")
    processed_dir: str = os.path.join(PROJECT_ROOT, "datasets", "processed")
    metadata_dir: str = os.path.join(PROJECT_ROOT, "datasets", "metadata")
    chunks_dir: str = os.path.join(PROJECT_ROOT, "datasets", "chunks")
    
    # Output
    corpus_out_dir: str = os.path.join(PROJECT_ROOT, "datasets", "corpus")
    
    # Validation Settings
    min_word_count: int = 10
    max_duplicate_ratio: float = 0.95
    drop_empty: bool = True
    
    # Cleaner Settings
    remove_page_numbers: bool = True
    normalize_whitespace: bool = True
    
    def __post_init__(self):
        os.makedirs(self.corpus_out_dir, exist_ok=True)
