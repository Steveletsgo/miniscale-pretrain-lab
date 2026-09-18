# Experiment plan

## Questions

1. How do DDP, FSDP full sharding and ZeRO-2/3 trade throughput for memory?
2. What fixed-global-batch scaling efficiency is achieved on 1, 2 and 4 identical GPUs?
3. How much memory does activation checkpointing save, and what recomputation cost does it add?
4. Does sequence packing increase effective tokens/s without changing validation quality?
5. Can model, optimizer, scheduler, RNG and data progress be restored after interruption?

## Workloads

### Workload A: from-scratch model

- Model: approximately 117.8M parameters
- Objective: causal language modeling
- Sequence length: 1024
- Target training tokens: approximately 524M
- Purpose: verify the complete data and pretraining pipeline

### Workload B: continual pretraining

- Model: Qwen2.5-1.5B base
- Objective: causal language modeling on a versioned open corpus subset
- Sequence length: 2048
- Target training tokens: approximately 65M
- Purpose: benchmark a mainstream architecture under realistic model-state memory pressure

## Fixed-global-batch scaling matrix

| GPUs | Micro batch/GPU | Accumulation | Global batch |
|---:|---:|---:|---:|
| 1 | 4 | 8 | 32 |
| 2 | 4 | 4 | 32 |
| 4 | 4 | 2 | 32 |

Each run uses 50 warm-up steps followed by 500 measured steps and is repeated three times.

## Required metrics

- train and validation loss;
- processed tokens and tokens/s;
- step-time median and P95;
- allocated and reserved peak GPU memory;
- GPU utilization and data-loader wait time;
- checkpoint save/load latency;
- scaling efficiency relative to one GPU;
- recovery correctness after process termination.

## Claims policy

Microbenchmark speedups must not be reported as end-to-end training speedups. Single-node
experiments must not be described as production-scale or multi-node pretraining. All published
numbers must point to a versioned JSON result and the Git commit that produced it.

