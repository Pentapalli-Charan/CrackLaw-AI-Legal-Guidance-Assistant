import re
import json
import logging
from typing import List, Dict, Any, Union, Tuple
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from src.models.exceptions import PreprocessingError

logger = logging.getLogger("CrackLaw.Models.Preprocessing")

def clean_text(text: str) -> str:
    """Standardizes string representations: strips whitespaces and normalizes case."""
    if not text:
        return ""
    # Normalize whitespaces
    text = re.sub(r"\s+", " ", text)
    # Remove special characters but keep alphanumeric and basic punctuation
    text = re.sub(r"[^\w\s\.,;\?!'\-]", "", text)
    return text.strip().lower()


class VocabularyIndexer:
    """Maps tokenized strings to integer sequences for deep learning models."""

    def __init__(self, max_vocab_size: int = 10000, pad_token: str = "<PAD>", unk_token: str = "<UNK>"):
        self.max_vocab_size = max_vocab_size
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.vocab: Dict[str, int] = {}
        self.inverse_vocab: Dict[int, str] = {}
        self.is_fitted = False

    def fit(self, texts: List[str]) -> "VocabularyIndexer":
        """Builds dictionary index from list of document texts."""
        word_counts: Dict[str, int] = {}
        for text in texts:
            cleaned = clean_text(text)
            tokens = cleaned.split()
            for token in tokens:
                word_counts[token] = word_counts.get(token, 0) + 1

        # Sort by frequency
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        top_words = sorted_words[:self.max_vocab_size - 2]  # Reserve space for pad and unk

        self.vocab = {self.pad_token: 0, self.unk_token: 1}
        for idx, (word, _) in enumerate(top_words, start=2):
            self.vocab[word] = idx

        self.inverse_vocab = {v: k for k, v in self.vocab.items()}
        self.is_fitted = True
        logger.info("Fitted VocabularyIndexer with vocab size: %d", len(self.vocab))
        return self

    def transform(self, texts: List[str], max_len: int = 100) -> np.ndarray:
        """Encodes text strings into padded integer vectors."""
        if not self.is_fitted:
            raise PreprocessingError("VocabularyIndexer must be fitted before transformation.")

        encoded_sequences = []
        for text in texts:
            cleaned = clean_text(text)
            tokens = cleaned.split()
            
            # Map tokens to indices, falling back to UNK
            seq = [self.vocab.get(token, self.vocab[self.unk_token]) for token in tokens]
            
            # Truncate or Pad
            if len(seq) > max_len:
                seq = seq[:max_len]
            else:
                seq = seq + [self.vocab[self.pad_token]] * (max_len - len(seq))
            
            encoded_sequences.append(seq)

        return np.array(encoded_sequences, dtype=np.int32)

    def save(self, filepath: str) -> None:
        """Serializes vocab mapping to a JSON file."""
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump({
                    "vocab": self.vocab,
                    "max_vocab_size": self.max_vocab_size,
                    "pad_token": self.pad_token,
                    "unk_token": self.unk_token
                }, f, indent=2)
            logger.info("Saved vocabulary to %s", filepath)
        except Exception as e:
            raise PreprocessingError(f"Failed to save vocabulary: {e}") from e

    def load(self, filepath: str) -> "VocabularyIndexer":
        """Loads vocabulary state from JSON file."""
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.vocab = data["vocab"]
            self.max_vocab_size = data["max_vocab_size"]
            self.pad_token = data["pad_token"]
            self.unk_token = data["unk_token"]
            self.inverse_vocab = {v: k for k, v in self.vocab.items()}
            self.is_fitted = True
            logger.info("Loaded vocabulary from %s", filepath)
            return self
        except Exception as e:
            raise PreprocessingError(f"Failed to load vocabulary: {e}") from e


class NumericalScaler:
    """Standardizes tabular numerical fields (x - mean) / std."""

    def __init__(self):
        self.means: np.ndarray = np.array([])
        self.stds: np.ndarray = np.array([])
        self.is_fitted = False

    def fit(self, data: np.ndarray) -> "NumericalScaler":
        """Calculates scaling boundaries (mean, standard deviation)."""
        arr = np.asarray(data, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        self.means = np.mean(arr, axis=0)
        self.stds = np.std(arr, axis=0)
        # Avoid division by zero
        self.stds[self.stds == 0.0] = 1.0
        self.is_fitted = True
        logger.info("Fitted NumericalScaler for shape: %s", arr.shape)
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        """Transforms data columns using computed statistics."""
        if not self.is_fitted:
            raise PreprocessingError("NumericalScaler must be fitted before transforming.")
        arr = np.asarray(data, dtype=np.float32)
        original_shape = arr.shape
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        
        scaled = (arr - self.means) / self.stds
        
        if len(original_shape) == 1:
            return scaled.flatten()
        return scaled

    def save(self, filepath: str) -> None:
        """Serializes scaler to a JSON file."""
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump({
                    "means": self.means.tolist(),
                    "stds": self.stds.tolist()
                }, f, indent=2)
            logger.info("Saved scaler params to %s", filepath)
        except Exception as e:
            raise PreprocessingError(f"Failed to save scaler: {e}") from e

    def load(self, filepath: str) -> "NumericalScaler":
        """Loads scaler boundaries from JSON file."""
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.means = np.array(data["means"], dtype=np.float32)
            self.stds = np.array(data["stds"], dtype=np.float32)
            self.is_fitted = True
            logger.info("Loaded scaler params from %s", filepath)
            return self
        except Exception as e:
            raise PreprocessingError(f"Failed to load scaler: {e}") from e
