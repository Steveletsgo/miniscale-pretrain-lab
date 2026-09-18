from pathlib import Path

import numpy as np
import torch

from miniscale.data import BinaryTokenDataset, SyntheticTokenDataset


def test_synthetic_dataset_is_deterministic() -> None:
    dataset = SyntheticTokenDataset(samples=4, sequence_length=8, vocab_size=32, seed=7)
    first = dataset[0]
    second = dataset[0]
    assert torch.equal(first[0], second[0])
    assert torch.equal(first[1], second[1])
    assert torch.equal(first[0][1:], first[1][:-1])


def test_binary_dataset(tmp_path: Path) -> None:
    path = tmp_path / "tokens.bin"
    np.arange(18, dtype=np.uint32).tofile(path)
    dataset = BinaryTokenDataset(path, sequence_length=8)
    assert len(dataset) == 2
    inputs, labels = dataset[1]
    assert inputs.tolist() == list(range(9, 17))
    assert labels.tolist() == list(range(10, 18))

