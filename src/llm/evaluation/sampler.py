"""
CrackLawLM Prompt Sampler
===========================
Manages evaluation prompts for text generation quality assessment.
Provides predefined legal prompts and encodes them for the model.
"""

import logging
from typing import List, Dict, Optional

import torch

logger = logging.getLogger("CrackLaw.LLM.Evaluation.Sampler")


class PromptSampler:
    """
    Manages and prepares prompts for text generation evaluation.

    Stores predefined legal prompts, encodes them into token IDs,
    and provides device-ready tensors for the generator.
    """

    def __init__(self, prompts: List[str]):
        self.prompts = prompts

    def prepare_prompt(
        self,
        prompt: str,
        tokenizer,
        device: torch.device,
    ) -> Dict[str, torch.Tensor]:
        """
        Encodes a text prompt into model-ready tensors.

        Args:
            prompt:    The text prompt string.
            tokenizer: The CrackLawTokenizer instance.
            device:    Target torch device.

        Returns:
            Dictionary with src_input_ids and src_padding_mask tensors.
        """
        token_ids = tokenizer.encode(prompt)

        # Add BOS token at the start
        bos_id = tokenizer.special_tokens.get_id(tokenizer.config.bos_token)
        token_ids = [bos_id] + token_ids

        src_input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
        src_padding_mask = torch.ones(
            1, 1, 1, len(token_ids), dtype=torch.bool, device=device
        )

        return {
            "src_input_ids": src_input_ids,
            "src_padding_mask": src_padding_mask,
            "prompt_text": prompt,
            "prompt_token_ids": token_ids,
        }

    def get_all_prompts(self) -> List[str]:
        """Returns all stored prompts."""
        return list(self.prompts)
