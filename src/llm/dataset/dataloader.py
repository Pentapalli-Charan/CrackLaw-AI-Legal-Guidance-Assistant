from torch.utils.data import DataLoader, random_split
from typing import Tuple
from src.llm.dataset.config import DatasetConfig
from src.llm.dataset.dataset import CrackLawDataset
from src.llm.dataset.collator import DataCollator

def create_dataloaders(
    dataset: CrackLawDataset, 
    config: DatasetConfig, 
    pad_token_id: int
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Splits the dataset and creates PyTorch DataLoaders for train, validation, and test.
    """
    total_len = len(dataset)
    
    # Calculate lengths
    train_len = int(total_len * config.train_ratio)
    val_len = int(total_len * config.val_ratio)
    test_len = total_len - train_len - val_len
    
    # Split
    train_set, val_set, test_set = random_split(dataset, [train_len, val_len, test_len])
    
    # Collator handles dynamic padding
    collator = DataCollator(pad_token_id=pad_token_id)
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_set,
        batch_size=config.batch_size,
        shuffle=True, # Shuffle train data
        collate_fn=collator,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory
    )
    
    val_loader = DataLoader(
        val_set,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory
    )
    
    test_loader = DataLoader(
        test_set,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory
    )
    
    return train_loader, val_loader, test_loader
