from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class SyntheticTokenDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, samples: int, sequence_length: int, vocab_size: int, seed: int) -> None:
        self.samples = samples
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.seed = seed

    def __len__(self) -> int:
        return self.samples

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        generator = torch.Generator().manual_seed(self.seed + index)
        tokens = torch.randint(
            0, self.vocab_size, (self.sequence_length + 1,), generator=generator
        )
        return tokens[:-1], tokens[1:]


class BinaryTokenDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, path: str | Path, sequence_length: int, dtype: str = "uint32") -> None:
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(self.path)
        self.sequence_length = sequence_length
        self.tokens = np.memmap(self.path, mode="r", dtype=np.dtype(dtype))
        self.sample_width = sequence_length + 1
        self.sample_count = len(self.tokens) // self.sample_width
        if self.sample_count == 0:
            raise ValueError(f"{self.path} does not contain one complete training sample")

    def __len__(self) -> int:
        return self.sample_count

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = index * self.sample_width
        array = np.asarray(self.tokens[start : start + self.sample_width], dtype=np.int64).copy()
        tokens = torch.from_numpy(array)
        return tokens[:-1], tokens[1:]

