import logging
from typing import List, Dict, Any, Tuple, Union, Optional
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset as TorchDataset, DataLoader as TorchDataLoader
from src.models.exceptions import PreprocessingError

logger = logging.getLogger("CrackLaw.Models.DatasetLoader")

class SimpleDataset(TorchDataset):
    """Simple PyTorch Dataset wrapper for tensors."""

    def __init__(self, x_data: np.ndarray, y_data: np.ndarray):
        self.x = torch.tensor(x_data, dtype=torch.float32 if x_data.dtype in [np.float32, np.float64] else torch.long)
        self.y = torch.tensor(y_data, dtype=torch.float32 if y_data.dtype in [np.float32, np.float64] else torch.long)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y[idx]


class ModelDatasetLoader:
    """Orchestrates loading, random split partitioning, and batching of raw dataset payloads."""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state

    def split_dataset(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        val_size: float = 0.15,
        test_size: float = 0.15,
        stratify: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Splits features and labels into train, val, and test matrices."""
        if len(features) != len(labels):
            raise PreprocessingError(f"Features size ({len(features)}) and labels size ({len(labels)}) mismatch.")

        # If stratify is requested, check if class frequencies allow it
        stratify_labels = labels if stratify else None

        # Split off test set first
        total_test_ratio = test_size / (1.0)
        x_train_val, x_test, y_train_val, y_test = train_test_split(
            features, labels,
            test_size=total_test_ratio,
            random_state=self.random_state,
            stratify=stratify_labels
        )

        # Split remaining train_val into train and val
        val_ratio_of_train_val = val_size / (1.0 - test_size)
        stratify_val = y_train_val if stratify else None
        
        x_train, x_val, y_train, y_val = train_test_split(
            x_train_val, y_train_val,
            test_size=val_ratio_of_train_val,
            random_state=self.random_state,
            stratify=stratify_val
        )

        logger.info(
            "Split dataset. Train shape: %s, Val shape: %s, Test shape: %s",
            x_train.shape, x_val.shape, x_test.shape
        )
        return x_train, y_train, x_val, y_val, x_test, y_test

    def get_pytorch_dataloaders(
        self,
        x_train: np.ndarray, y_train: np.ndarray,
        x_val: np.ndarray, y_val: np.ndarray,
        x_test: np.ndarray, y_test: np.ndarray,
        batch_size: int = 32,
        shuffle_train: bool = True
    ) -> Tuple[TorchDataLoader, TorchDataLoader, TorchDataLoader]:
        """Wraps split NumPy matrices inside active PyTorch DataLoader iterators."""
        try:
            train_ds = SimpleDataset(x_train, y_train)
            val_ds = SimpleDataset(x_val, y_val)
            test_ds = SimpleDataset(x_test, y_test)

            train_loader = TorchDataLoader(train_ds, batch_size=batch_size, shuffle=shuffle_train)
            val_loader = TorchDataLoader(val_ds, batch_size=batch_size, shuffle=False)
            test_loader = TorchDataLoader(test_ds, batch_size=batch_size, shuffle=False)

            return train_loader, val_loader, test_loader
        except Exception as e:
            raise PreprocessingError(f"Failed to generate PyTorch DataLoader: {e}") from e

    def get_numpy_batches(
        self,
        x_data: np.ndarray,
        y_data: np.ndarray,
        batch_size: int = 32,
        shuffle: bool = True
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Splits matrices into list of mini-batch tuple arrays for framework-independent loops."""
        indices = np.arange(len(x_data))
        if shuffle:
            np.random.seed(self.random_state)
            np.random.shuffle(indices)

        x_shuffled = x_data[indices]
        y_shuffled = y_data[indices]

        batches = []
        for i in range(0, len(x_data), batch_size):
            batches.append((
                x_shuffled[i:i + batch_size],
                y_shuffled[i:i + batch_size]
            ))
        return batches
