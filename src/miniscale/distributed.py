from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import torch.distributed as dist


@dataclass
class DistributedContext:
    rank: int
    local_rank: int
    world_size: int
    distributed: bool
    device: torch.device

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def initialize(strategy: str) -> DistributedContext:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = strategy == "ddp" and world_size > 1
    if distributed:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)
        rank = dist.get_rank()
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
        else:
            device = torch.device("cpu")
    else:
        rank = 0
        local_rank = 0
        world_size = 1
        device = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")
    return DistributedContext(rank, local_rank, world_size, distributed, device)


def finalize(context: DistributedContext) -> None:
    if context.distributed:
        dist.barrier()
        dist.destroy_process_group()

