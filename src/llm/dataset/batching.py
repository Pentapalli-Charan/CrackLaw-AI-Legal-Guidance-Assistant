from torch.utils.data import Sampler, Dataset
from typing import Iterator, List
import random

class LengthGroupedSampler(Sampler):
    """
    Advanced batching: Samples indices in a way that groups sequences of similar lengths.
    This minimizes the amount of padding required in dynamic padding, saving memory and compute.
    """
    def __init__(self, dataset: Dataset, batch_size: int, shuffle: bool = True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        
    def __iter__(self) -> Iterator[List[int]]:
        # In a real length-grouped sampler, we would measure lengths of all items
        # and sort/bucket them. For now, we implement a simple random batch sampler
        # as a placeholder for advanced bucket batching.
        indices = list(range(len(self.dataset)))
        if self.shuffle:
            random.shuffle(indices)
            
        # Yield batches of indices
        for i in range(0, len(indices), self.batch_size):
            yield indices[i:i + self.batch_size]
            
    def __len__(self) -> int:
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size
