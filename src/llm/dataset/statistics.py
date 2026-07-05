import logging
from typing import Dict, Any
from torch.utils.data import DataLoader

logger = logging.getLogger("CrackLaw.LLM.DatasetStatistics")

class DatasetStatistics:
    """Generates statistics about the compiled PyTorch dataset and loaders."""
    
    @staticmethod
    def generate_report(
        train_loader: DataLoader, 
        val_loader: DataLoader, 
        test_loader: DataLoader,
        vocab_size: int
    ) -> Dict[str, Any]:
        
        train_samples = len(train_loader.dataset)
        val_samples = len(val_loader.dataset)
        test_samples = len(test_loader.dataset)
        total_samples = train_samples + val_samples + test_samples
        
        # Calculate padding ratio and sequence lengths using a small sample of the train loader
        total_tokens = 0
        total_padding = 0
        max_seq_len = 0
        
        # We only iterate over a few batches to estimate if dataset is huge
        max_batches_to_check = min(10, len(train_loader))
        
        for i, batch in enumerate(train_loader):
            if i >= max_batches_to_check:
                break
                
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            
            seq_len = input_ids.shape[1]
            if seq_len > max_seq_len:
                max_seq_len = seq_len
                
            # Count elements
            batch_total = input_ids.numel()
            batch_valid = attention_mask.sum().item()
            batch_pad = batch_total - batch_valid
            
            total_tokens += batch_total
            total_padding += batch_pad
            
        pad_ratio = (total_padding / total_tokens) if total_tokens > 0 else 0
            
        stats = {
            "total_dataset_size": total_samples,
            "training_samples": train_samples,
            "validation_samples": val_samples,
            "test_samples": test_samples,
            "observed_max_sequence_length": max_seq_len,
            "estimated_padding_ratio": round(pad_ratio, 4),
            "vocabulary_size": vocab_size
        }
        return stats

    @staticmethod
    def print_report(stats: Dict[str, Any]) -> str:
        report = "====================================================\n"
        report += "               DATASET STATISTICS                   \n"
        report += "====================================================\n"
        for key, value in stats.items():
            friendly_key = key.replace("_", " ").title()
            report += f"{friendly_key}: {value}\n"
        report += "====================================================\n"
        logger.info("\n" + report)
        return report
