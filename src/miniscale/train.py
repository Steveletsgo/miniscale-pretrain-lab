from __future__ import annotations

import argparse
import json
import math
import random
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

from .checkpoint import load_checkpoint, save_checkpoint
from .config import ExperimentConfig, load_config
from .data import BinaryTokenDataset, SyntheticTokenDataset
from .distributed import DistributedContext, finalize, initialize
from .model import DecoderLM


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir")
    parser.add_argument("--resume")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_dataset(config: ExperimentConfig, split: str):
    path = config.data.train_path if split == "train" else config.data.validation_path
    samples = (
        config.data.synthetic_train_samples
        if split == "train"
        else config.data.synthetic_validation_samples
    )
    if path:
        return BinaryTokenDataset(path, config.model.max_sequence_length, config.data.dtype)
    if samples <= 0:
        raise ValueError(f"no {split} dataset configured")
    seed_offset = 0 if split == "train" else 1_000_000
    return SyntheticTokenDataset(
        samples,
        config.model.max_sequence_length,
        config.model.vocab_size,
        config.train.seed + seed_offset,
    )


def build_loader(config: ExperimentConfig, context: DistributedContext, split: str):
    dataset = build_dataset(config, split)
    sampler = None
    if context.distributed:
        sampler = DistributedSampler(
            dataset,
            num_replicas=context.world_size,
            rank=context.rank,
            shuffle=split == "train",
            seed=config.train.seed,
            drop_last=split == "train",
        )
    return DataLoader(
        dataset,
        batch_size=config.train.micro_batch_size,
        shuffle=sampler is None and split == "train",
        sampler=sampler,
        num_workers=config.data.num_workers,
        pin_memory=context.device.type == "cuda",
        drop_last=split == "train",
    )


def build_optimizer(model: torch.nn.Module, config: ExperimentConfig):
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        (decay if parameter.ndim >= 2 and "norm" not in name else no_decay).append(parameter)
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": config.train.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=config.train.learning_rate,
        betas=(0.9, 0.95),
    )


def build_scheduler(optimizer, config: ExperimentConfig):
    warmup = max(config.train.warmup_steps, 1)
    total = max(config.train.max_steps, warmup + 1)
    minimum_ratio = config.train.min_learning_rate / config.train.learning_rate

    def multiplier(step: int) -> float:
        if step < warmup:
            return max(step, 1) / warmup
        progress = min((step - warmup) / (total - warmup), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return minimum_ratio + (1.0 - minimum_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


def autocast_context(device: torch.device, precision: str):
    if device.type != "cuda" or precision == "fp32":
        return nullcontext()
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


@torch.no_grad()
def evaluate(model, loader, config: ExperimentConfig, context: DistributedContext) -> float:
    model.eval()
    losses = []
    for batch_index, (input_ids, labels) in enumerate(loader):
        if batch_index >= config.train.eval_batches:
            break
        input_ids = input_ids.to(context.device, non_blocking=True)
        labels = labels.to(context.device, non_blocking=True)
        with autocast_context(context.device, config.train.precision):
            loss = model(input_ids, labels)["loss"]
        losses.append(loss.detach())
    if not losses:
        raise RuntimeError("validation loader produced no batches")
    mean_loss = torch.stack(losses).mean()
    if context.distributed:
        dist.all_reduce(mean_loss, op=dist.ReduceOp.SUM)
        mean_loss /= context.world_size
    model.train()
    return float(mean_loss.item())


def infinite_batches(loader):
    epoch = 0
    while True:
        if isinstance(loader.sampler, DistributedSampler):
            loader.sampler.set_epoch(epoch)
        yield from loader
        epoch += 1


def append_metric(path: Path, record: dict) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")


def train(config: ExperimentConfig) -> None:
    context = initialize(config.train.strategy)
    seed_everything(config.train.seed + context.rank)
    run_dir = Path(config.output.run_dir)
    if context.is_main:
        run_dir.mkdir(parents=True, exist_ok=True)
        with (run_dir / "config.json").open("w", encoding="utf-8") as stream:
            json.dump(config.as_dict(), stream, indent=2)

    model = DecoderLM(config.model).to(context.device)
    if context.distributed:
        model = DistributedDataParallel(
            model,
            device_ids=[context.local_rank] if context.device.type == "cuda" else None,
        )
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config)
    train_loader = build_loader(config, context, "train")
    validation_loader = build_loader(config, context, "validation")
    batch_iterator = infinite_batches(train_loader)
    scaler = torch.amp.GradScaler(
        "cuda", enabled=context.device.type == "cuda" and config.train.precision == "fp16"
    )

    start_step = 0
    if config.output.resume_from:
        start_step = load_checkpoint(
            config.output.resume_from, model, optimizer, scheduler, context.device
        )

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if context.is_main:
        print(f"parameters={parameter_count:,} device={context.device} world_size={context.world_size}")

    model.train()
    metrics_path = run_dir / "metrics.jsonl"
    optimizer.zero_grad(set_to_none=True)
    for step in range(start_step + 1, config.train.max_steps + 1):
        step_started = time.perf_counter()
        accumulated_loss = torch.zeros((), device=context.device)
        tokens = 0
        for accumulation_index in range(config.train.gradient_accumulation_steps):
            input_ids, labels = next(batch_iterator)
            input_ids = input_ids.to(context.device, non_blocking=True)
            labels = labels.to(context.device, non_blocking=True)
            tokens += input_ids.numel() * context.world_size
            should_sync = accumulation_index == config.train.gradient_accumulation_steps - 1
            sync_context = nullcontext()
            if context.distributed and not should_sync:
                sync_context = model.no_sync()
            with sync_context:
                with autocast_context(context.device, config.train.precision):
                    loss = model(input_ids, labels)["loss"]
                    scaled_loss = loss / config.train.gradient_accumulation_steps
                scaler.scale(scaled_loss).backward()
                accumulated_loss += loss.detach()

        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.train.max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        if context.device.type == "cuda":
            torch.cuda.synchronize(context.device)
        elapsed = time.perf_counter() - step_started
        mean_loss = accumulated_loss / config.train.gradient_accumulation_steps
        if context.distributed:
            dist.all_reduce(mean_loss, op=dist.ReduceOp.SUM)
            mean_loss /= context.world_size

        if context.is_main and step % config.train.log_interval == 0:
            record = {
                "step": step,
                "train_loss": float(mean_loss.item()),
                "learning_rate": float(scheduler.get_last_lr()[0]),
                "grad_norm": float(grad_norm),
                "step_time_seconds": elapsed,
                "tokens_per_second": tokens / elapsed,
                "world_size": context.world_size,
            }
            append_metric(metrics_path, record)
            print(json.dumps(record))

        if step % config.train.eval_interval == 0:
            validation_loss = evaluate(model, validation_loader, config, context)
            if context.is_main:
                append_metric(
                    metrics_path,
                    {"step": step, "validation_loss": validation_loss, "type": "evaluation"},
                )

        if context.is_main and step % config.train.checkpoint_interval == 0:
            save_checkpoint(
                run_dir / f"step-{step:07d}.pt",
                model,
                optimizer,
                scheduler,
                step,
                config.as_dict(),
            )

    if context.is_main:
        save_checkpoint(
            run_dir / "final.pt",
            model,
            optimizer,
            scheduler,
            config.train.max_steps,
            config.as_dict(),
        )
    finalize(context)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.run_dir:
        config.output.run_dir = args.run_dir
    if args.resume:
        config.output.resume_from = args.resume
    train(config)


if __name__ == "__main__":
    main()
