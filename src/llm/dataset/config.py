from dataclasses import dataclass

@dataclass
class DatasetConfig:
    """Configuration for PyTorch Dataset and DataLoader."""
    
    # Sequence Length
    max_sequence_length: int = 512
    
    # DataLoader Settings
    batch_size: int = 8
    num_workers: int = 0  # 0 for local/Windows stability, increase for production Linux
    pin_memory: bool = True
    
    # Splitting
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    
    # Padding
    pad_to_max_length: bool = False # If false, dynamic padding (to batch max) is used in the collator
    
    def __post_init__(self):
        # Ensure ratios sum to 1.0 (with slight float tolerance)
        total = self.train_ratio + self.val_ratio + self.test_ratio
        assert abs(total - 1.0) < 1e-5, f"Split ratios must sum to 1.0, got {total}"
