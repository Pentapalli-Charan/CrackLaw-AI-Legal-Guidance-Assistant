"""
CrackLawLM Text Generation Evaluator
=======================================
Generates sample legal text from predefined prompts after each checkpoint.
Stores outputs and enables checkpoint-to-checkpoint comparison.
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional

import torch
import torch.nn as nn

from src.llm.transformer.generation import TransformerGenerator
from src.llm.evaluation.config import EvaluationConfig
from src.llm.evaluation.sampler import PromptSampler

logger = logging.getLogger("CrackLaw.LLM.Evaluation.TextGeneration")


class TextGenerationEvaluator:
    """
    Generates sample text from the model using predefined legal prompts.

    After each checkpoint/evaluation, generates text and stores outputs
    for tracking model quality progression over training.
    """

    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.sampler = PromptSampler(config.generation_prompts)
        self.generation_history: List[Dict[str, Any]] = []
        self.output_file = os.path.join(config.output_dir, "generations.jsonl")

    @torch.no_grad()
    def generate_samples(
        self,
        model: nn.Module,
        tokenizer,
        device: torch.device,
        epoch: int = 0,
        global_step: int = 0,
    ) -> List[Dict[str, str]]:
        """
        Generates text for all configured prompts.

        Args:
            model:       The CrackLawTransformer model.
            tokenizer:   The CrackLawTokenizer.
            device:      Torch device.
            epoch:       Current epoch (for tracking).
            global_step: Current step (for tracking).

        Returns:
            List of dictionaries with prompt, generated_text, and metadata.
        """
        if not self.config.generation_enabled:
            return []

        model.eval()
        generator = TransformerGenerator(model)

        bos_id = tokenizer.special_tokens.get_id(tokenizer.config.bos_token)
        eos_id = tokenizer.special_tokens.get_id(tokenizer.config.eos_token)

        results = []
        prompts = self.sampler.get_all_prompts()
        num_samples = min(self.config.num_generation_samples, len(prompts))

        for i in range(num_samples):
            prompt = prompts[i]

            try:
                prepared = self.sampler.prepare_prompt(prompt, tokenizer, device)

                generated_ids = generator.greedy_search(
                    src_input_ids=prepared["src_input_ids"],
                    src_padding_mask=prepared["src_padding_mask"],
                    max_new_tokens=self.config.max_generation_tokens,
                    start_token_id=bos_id,
                    end_token_id=eos_id,
                )

                # Decode generated tokens
                gen_token_list = generated_ids.squeeze(0).tolist()
                generated_text = tokenizer.decode(gen_token_list)

                result = {
                    "prompt": prompt,
                    "generated_text": generated_text,
                    "num_tokens_generated": len(gen_token_list),
                    "epoch": epoch,
                    "global_step": global_step,
                }
                results.append(result)

                logger.info(
                    f"  Generation [{i+1}/{num_samples}]: "
                    f"\"{prompt[:40]}...\" → "
                    f"\"{generated_text[:60]}...\""
                )

            except Exception as e:
                logger.warning(f"Generation failed for prompt '{prompt[:30]}': {e}")
                results.append({
                    "prompt": prompt,
                    "generated_text": f"[ERROR: {str(e)}]",
                    "num_tokens_generated": 0,
                    "epoch": epoch,
                    "global_step": global_step,
                })

        # Store in history and persist
        self.generation_history.append({
            "epoch": epoch,
            "global_step": global_step,
            "samples": results,
        })
        self._persist(results, epoch, global_step)

        return results

    def compare_checkpoints(self) -> List[Dict[str, Any]]:
        """
        Compares generated outputs across checkpoints.

        Returns:
            List of comparisons, each showing how the same prompt's
            output evolved across epochs.
        """
        if len(self.generation_history) < 2:
            return []

        comparisons = []
        prompts = self.sampler.get_all_prompts()

        for prompt_idx, prompt in enumerate(prompts):
            evolution = {
                "prompt": prompt,
                "generations_over_time": [],
            }
            for entry in self.generation_history:
                samples = entry.get("samples", [])
                if prompt_idx < len(samples):
                    evolution["generations_over_time"].append({
                        "epoch": entry["epoch"],
                        "text": samples[prompt_idx].get("generated_text", ""),
                        "num_tokens": samples[prompt_idx].get("num_tokens_generated", 0),
                    })
            if evolution["generations_over_time"]:
                comparisons.append(evolution)

        return comparisons

    def _persist(self, results: List[Dict], epoch: int, global_step: int):
        """Appends generation results to the JSONL output file."""
        record = {
            "epoch": epoch,
            "global_step": global_step,
            "samples": results,
        }
        with open(self.output_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
