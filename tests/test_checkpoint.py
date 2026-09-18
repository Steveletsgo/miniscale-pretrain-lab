from pathlib import Path

import torch

from miniscale.checkpoint import load_checkpoint, save_checkpoint
from miniscale.model import DecoderConfig, DecoderLM


def test_checkpoint_round_trip(tmp_path: Path) -> None:
    config = DecoderConfig(
        vocab_size=32,
        num_layers=1,
        hidden_size=32,
        num_attention_heads=4,
        num_kv_heads=2,
        intermediate_size=64,
        max_sequence_length=8,
    )
    model = DecoderLM(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    path = tmp_path / "checkpoint.pt"
    original = {name: value.detach().clone() for name, value in model.state_dict().items()}
    save_checkpoint(path, model, optimizer, scheduler, 7, {"name": "test"})
    for parameter in model.parameters():
        parameter.data.zero_()
    step = load_checkpoint(path, model, optimizer, scheduler, torch.device("cpu"))
    assert step == 7
    for name, value in model.state_dict().items():
        assert torch.equal(value, original[name])


def test_checkpoint_round_trip_with_cuda_map_location(tmp_path: Path) -> None:
    if not torch.cuda.is_available():
        return

    device = torch.device("cuda")
    model = torch.nn.Linear(4, 2).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    path = tmp_path / "checkpoint-cuda.pt"

    save_checkpoint(path, model, optimizer, scheduler, 11, {"name": "cuda-test"})
    step = load_checkpoint(path, model, optimizer, scheduler, device)

    assert step == 11
