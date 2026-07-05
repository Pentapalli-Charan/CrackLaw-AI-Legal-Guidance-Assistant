from typing import List, Dict, Tuple
from src.llm.tokenizer.config import TokenizerConfig
from src.llm.tokenizer.special_tokens import SpecialTokens
from src.llm.tokenizer.vocabulary import Vocabulary
from src.llm.tokenizer.encoder import Encoder
from src.llm.tokenizer.decoder import Decoder
from src.llm.tokenizer.trainer import Trainer
from src.llm.tokenizer.serialization import Serializer

class CrackLawTokenizer:
    """
    Custom Byte Pair Encoding (BPE) Tokenizer for CrackLaw.
    Provides a unified facade to train, encode, decode, save, and load.
    """
    
    def __init__(self, config: TokenizerConfig = None):
        self.config = config or TokenizerConfig()
        self.special_tokens = SpecialTokens(self.config)
        self.vocab = Vocabulary(self.special_tokens)
        self.merges: Dict[Tuple[str, str], int] = {}
        
        self.encoder = None
        self.decoder = None
        self.serializer = Serializer(self.config)
        
    def _init_components(self):
        self.encoder = Encoder(self.config, self.vocab, self.merges)
        self.decoder = Decoder(self.vocab)
        
    def train(self, texts: List[str]) -> None:
        """Trains the tokenizer on a corpus from scratch."""
        trainer = Trainer(self.config, self.vocab)
        self.merges = trainer.train(texts)
        self._init_components()
        
    def encode(self, text: str) -> List[int]:
        """Encodes a string into a list of token IDs."""
        if self.encoder is None:
            self._init_components()
        return self.encoder.encode(text)
        
    def decode(self, token_ids: List[int]) -> str:
        """Decodes a list of token IDs back into a string."""
        if self.decoder is None:
            self._init_components()
        return self.decoder.decode(token_ids)
        
    def save(self) -> None:
        """Saves the vocabulary and merges to disk."""
        self.serializer.save(self.vocab.get_mapping(), self.merges)
        
    def load(self) -> None:
        """Loads the vocabulary and merges from disk."""
        vocab_mapping, self.merges = self.serializer.load()
        self.vocab.load_mapping(vocab_mapping)
        self._init_components()
        
    def get_vocab_size(self) -> int:
        """Returns the current vocabulary size."""
        return len(self.vocab)
