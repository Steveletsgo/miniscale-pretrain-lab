from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

from .model import DecoderConfig


@dataclass
class DataConfig:
    train_path: str | None = None
    validation_path: str | None = None
    dtype: str = "uint32"
    synthetic_train_samples: int = 0
    synthetic_validation_samples: int = 0
    num_workers: int = 0


@dataclass
class TrainConfig:
    seed: int = 1337
    strategy: str = "single"
    precision: str = "fp32"
    micro_batch_size: int = 4
    gradient_accumulation_steps: int = 1
    max_steps: int = 100
    eval_interval: int = 100
    eval_batches: int = 10
    checkpoint_interval: int = 100
    log_interval: int = 10
    learning_rate: float = 3e-4
    min_learning_rate: float = 3e-5
    warmup_steps: int = 10
    weight_decay: float = 0.1
    max_grad_norm: float = 1.0


@dataclass
class OutputConfig:
    run_dir: str = "runs/default"
    resume_from: str | None = None


@dataclass
class ExperimentConfig:
    model: DecoderConfig
    data: DataConfig
    train: TrainConfig
    output: OutputConfig

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_config(path: str | Path) -> ExperimentConfig:
    with Path(path).open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)
    config = ExperimentConfig(
        model=DecoderConfig(**raw["model"]),
        data=DataConfig(**raw["data"]),
        train=TrainConfig(**raw["train"]),
        output=OutputConfig(**raw["output"]),
    )
    validate_config(config)
    return config


def validate_config(config: ExperimentConfig) -> None:
    model = config.model
    train = config.train
    if model.hidden_size % model.num_attention_heads != 0:
        raise ValueError("hidden_size must be divisible by num_attention_heads")
    if model.num_attention_heads % model.num_kv_heads != 0:
        raise ValueError("num_attention_heads must be divisible by num_kv_heads")
    if train.strategy not in {"single", "ddp"}:
        raise ValueError("strategy must be 'single' or 'ddp' in milestone 1")
    if train.precision not in {"fp32", "fp16", "bf16"}:
        raise ValueError("precision must be fp32, fp16 or bf16")
    if train.max_steps <= 0 or train.gradient_accumulation_steps <= 0:
        raise ValueError("max_steps and gradient_accumulation_steps must be positive")

