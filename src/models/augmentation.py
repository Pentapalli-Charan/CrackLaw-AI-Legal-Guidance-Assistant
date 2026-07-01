import random
import logging
from typing import List
import numpy as np

logger = logging.getLogger("CrackLaw.Models.Augmentation")

# Local dictionary of common legal terminology synonyms
LEGAL_SYNONYMS = {
    "contract": "agreement",
    "agreement": "contract",
    "shall": "must",
    "must": "shall",
    "liability": "responsibility",
    "responsibility": "liability",
    "court": "tribunal",
    "tribunal": "court",
    "breach": "violation",
    "violation": "breach",
    "clause": "provision",
    "provision": "clause",
    "compensation": "damages",
    "damages": "compensation",
    "party": "signatory",
    "signatory": "party"
}

def synonym_replacement(text: str, probability: float = 0.15) -> str:
    """Randomly replaces words in text with legal synonyms."""
    if not text:
        return ""
    words = text.split()
    new_words = []
    for word in words:
        # Strip punctuation to look up in synonym dict
        clean_word = word.lower().strip(".,;:!?()\"'")
        if clean_word in LEGAL_SYNONYMS and random.random() < probability:
            syn = LEGAL_SYNONYMS[clean_word]
            # Maintain capitalization style
            if word.istitle():
                syn = syn.title()
            elif word.isupper():
                syn = syn.upper()
            
            # Re-attach stripped punctuation
            lead_punc = word[:len(word) - len(word.lstrip(".,;:!?()\"'"))]
            trail_punc = word[len(word.rstrip(".,;:!?()\"'")):]
            new_words.append(f"{lead_punc}{syn}{trail_punc}")
        else:
            new_words.append(word)
    return " ".join(new_words)


def random_word_deletion(text: str, probability: float = 0.1) -> str:
    """Randomly deletes tokens from a text string with a given probability."""
    if not text:
        return ""
    words = text.split()
    if len(words) <= 2:
        return text  # Avoid empty strings
        
    new_words = [word for word in words if random.random() > probability]
    if not new_words:
        return random.choice(words)
    return " ".join(new_words)


def random_word_swap(text: str, num_swaps: int = 1) -> str:
    """Randomly swaps adjacent words to simulate phrasing noise."""
    if not text:
        return ""
    words = text.split()
    if len(words) < 2:
        return text

    for _ in range(num_swaps):
        idx = random.randint(0, len(words) - 2)
        words[idx], words[idx + 1] = words[idx + 1], words[idx]
    
    return " ".join(words)


def add_gaussian_noise(data: np.ndarray, mean: float = 0.0, std: float = 0.05) -> np.ndarray:
    """Applies random Gaussian noise to numerical/tabular vectors."""
    arr = np.asarray(data, dtype=np.float32)
    noise = np.random.normal(mean, std, arr.shape)
    return arr + noise


class DataAugmenter:
    """Utility wrapper exposing text and numerical augmentation techniques."""

    def __init__(self, text_del_prob: float = 0.05, text_syn_prob: float = 0.1, num_noise_std: float = 0.02):
        self.text_del_prob = text_del_prob
        self.text_syn_prob = text_syn_prob
        self.num_noise_std = num_noise_std

    def augment_text(self, text: str) -> str:
        """Runs combinations of word deletion, synonym swapping, and word swaps on text."""
        augmented = synonym_replacement(text, self.text_syn_prob)
        augmented = random_word_deletion(augmented, self.text_del_prob)
        if random.random() < 0.3:
            augmented = random_word_swap(augmented, num_swaps=1)
        return augmented

    def augment_numerical(self, data: np.ndarray) -> np.ndarray:
        """Applies normal noise distribution to matrices."""
        return add_gaussian_noise(data, std=self.num_noise_std)
