import os
import logging
import torch

from src.config import Config
from src.ai.exceptions import LLMProviderError
from src.ai.llm_gateway import BaseLLMProvider
from src.llm.transformer.config import TransformerConfig
from src.llm.transformer.model import CrackLawTransformer
from src.llm.transformer.generation import TransformerGenerator
from src.llm.tokenizer.tokenizer import CrackLawTokenizer
from src.llm.tokenizer.config import TokenizerConfig
from src.llm.training.checkpoint_manager import CheckpointManager
from src.llm.training.config import TrainingConfig

logger = logging.getLogger("CrackLaw.AI.Provider.CrackLawLM")

class CrackLawLMProvider(BaseLLMProvider):
    """
    CrackLawLM local inference provider.
    Loads the trained CrackLaw Transformer checkpoint and tokenizer to generate responses locally.
    Uses lazy initialization to avoid loading weights into memory until the first generation request.
    """

    def __init__(self, config: Config, model_name: str):
        super().__init__(config, model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None
        self.generator = None

    def _initialize_model(self):
        """Lazy loads the model and tokenizer from disk."""
        if self.model is not None:
            return

        logger.info("Initializing CrackLawLM local provider...")

        # 1. Load Tokenizer
        tokenizer_config = TokenizerConfig()
        self.tokenizer = CrackLawTokenizer(tokenizer_config)
        try:
            self.tokenizer.load()
            logger.debug(f"Tokenizer loaded with vocab size: {self.tokenizer.get_vocab_size()}")
        except FileNotFoundError as e:
            raise LLMProviderError("CrackLawLM tokenizer not found. Please train it first.") from e

        # 2. Get checkpoint
        training_config = TrainingConfig()
        checkpoint_manager = CheckpointManager(training_config)
        
        ckpt_path = checkpoint_manager.get_best_checkpoint()
        if not ckpt_path:
            ckpt_path = checkpoint_manager.get_latest_checkpoint()
            
        if not ckpt_path:
            raise LLMProviderError("No CrackLawLM checkpoints found. Please train the model first.")

        # 3. Load Model
        # Using the same config as train.py/inference.py for this phase
        transformer_config = TransformerConfig(
            vocab_size=self.tokenizer.get_vocab_size(),
            d_model=128,          
            num_heads=4,
            d_ff=512,
            num_encoder_layers=2,
            num_decoder_layers=2
        )
        
        self.model = CrackLawTransformer(transformer_config)
        
        logger.info(f"Loading CrackLawLM weights from {ckpt_path}...")
        try:
            checkpoint_manager.load(ckpt_path, self.model, device=self.device)
        except Exception as e:
            raise LLMProviderError(f"Failed to load CrackLawLM checkpoint: {e}") from e
            
        self.model.to(self.device)
        self.model.eval()
        self.generator = TransformerGenerator(self.model)
        logger.info("CrackLawLM initialization complete.")

    def generate(self, prompt: str, system_prompt: str, temperature: float, max_tokens: int, top_p: float) -> str:
        """Generates text locally using the CrackLaw Transformer."""
        # Lazy init
        self._initialize_model()
        
        try:
            # Combine system prompt and user prompt
            full_prompt = f"{system_prompt}\n\nUser: {prompt}\nCrackLaw:"
            
            token_ids = self.tokenizer.encode(full_prompt)
            
            pad_id = self.tokenizer.special_tokens.get_id(self.tokenizer.config.pad_token)
            bos_id = self.tokenizer.special_tokens.get_id(self.tokenizer.config.bos_token)
            eos_id = self.tokenizer.special_tokens.get_id(self.tokenizer.config.eos_token)
            
            # Ensure BOS token is present to match training conditions
            if not token_ids or token_ids[0] != bos_id:
                token_ids = [bos_id] + token_ids
                
            # Truncate sequence safely to fit within Positional Encoding limits
            max_seq_len = self.model.config.max_seq_len
            if len(token_ids) > max_seq_len:
                logger.warning(f"Prompt length ({len(token_ids)}) exceeds max_seq_len ({max_seq_len}). Truncating.")
                token_ids = token_ids[:max_seq_len]
                
            input_ids = torch.tensor([token_ids], dtype=torch.long, device=self.device)
            padding_mask = (input_ids != pad_id).unsqueeze(1).unsqueeze(2).to(self.device)
            
            # Use the local generator
            # Currently uses greedy search, ignoring temperature and top_p for now as per TransformerGenerator
            generated_ids = self.generator.greedy_search(
                src_input_ids=input_ids,
                src_padding_mask=padding_mask,
                max_new_tokens=max_tokens,
                start_token_id=bos_id,
                end_token_id=eos_id
            )
            
            out_ids = generated_ids[0].tolist()
            if out_ids and out_ids[0] == bos_id:
                out_ids = out_ids[1:]
                
            return self.tokenizer.decode(out_ids)
            
        except Exception as e:
            logger.error(f"CrackLawLM generation error: {e}")
            raise LLMProviderError(f"CrackLawLM local generation failed: {e}") from e
