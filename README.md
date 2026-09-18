# MiniScale Pretrain Lab

A reproducible laboratory for learning the mechanics of language-model pretraining before
moving to rented multi-GPU infrastructure.

The repository deliberately distinguishes three activities:

- **From-scratch pretraining:** a randomly initialized decoder-only Transformer.
- **Continual pretraining:** continued next-token training of an existing base model.
- **Supervised fine-tuning:** instruction/response training, which is not called pretraining here.

## Current milestone

Milestone 1 provides:

- a decoder-only Transformer with RMSNorm, RoPE, grouped-query attention and SwiGLU;
- deterministic synthetic data for CPU/GPU smoke tests;
- memory-mapped token data for real corpus training;
- single-process and `torchrun` DDP execution;
- BF16/FP16 autocast, gradient accumulation, clipping and cosine scheduling;
- checkpoints containing model, optimizer, scheduler, step and RNG state;
- machine-readable JSONL metrics and reproducible YAML configurations.

FSDP, DeepSpeed ZeRO, Qwen continual pretraining and 1/2/4-GPU scaling results are later
milestones. They will only be merged after validation on the target CUDA environment.

## Quick start

Create an environment using the PyTorch build appropriate for the machine's CUDA version,
then install this repository:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[dev]"
```

Run tests and the CPU smoke experiment:

```bash
pytest -q
python -m miniscale.train --config configs/smoke_cpu.yaml
```

On the validated 8GB RTX 4060 Laptop development machine, run the short 117.8M-parameter
BF16 experiment with:

```bash
python -m miniscale.train --config configs/local_4060_120m.yaml
```

The measured systems baseline, including its mid-run throughput change and limitations, is
documented in [`benchmarks/local-rtx4060-120m.md`](benchmarks/local-rtx4060-120m.md).

The first real-corpus learning curve is documented in
[`benchmarks/local-rtx4060-fineweb-120m.md`](benchmarks/local-rtx4060-fineweb-120m.md).

## WSL CUDA operator development

The validated local operator-development environment lives in the `Ubuntu-AI` WSL
distribution on `D:`. It includes CUDA 12.8, PyTorch 2.7.0+cu128, Triton 3.3.0 and the native
C++/CUDA build toolchain. Enter it from PowerShell and activate it with:

```powershell
wsl -d Ubuntu-AI
```

```bash
cd /mnt/d/AI-Projects/miniscale-pretrain-lab
source scripts/activate_wsl_env.sh
source /home/ai/.venvs/miniscale/bin/activate
```

Run the three compilation/execution smoke tests:

```bash
mkdir -p /tmp/miniscale-cuda
nvcc -O3 -std=c++17 -arch=sm_89 benchmarks/cuda/vector_add.cu \
  -o /tmp/miniscale-cuda/vector_add
/tmp/miniscale-cuda/vector_add
python benchmarks/cuda/torch_extension_smoke.py
python benchmarks/triton/vector_add.py
```

See [`docs/wsl-cuda-environment.md`](docs/wsl-cuda-environment.md) for the exact versions,
measured smoke benchmark and current WSL profiling limitations.

Resume from a checkpoint while overriding the final step and output directory:

```bash
python -m miniscale.train \
  --config configs/local_4060_120m.yaml \
  --resume runs/local_4060_120m/final.pt \
  --max-steps 22 \
  --run-dir runs/local_4060_resume
```

Run on four GPUs with DDP:

```bash
torchrun --standalone --nproc-per-node=4 \
  -m miniscale.train --config configs/pretrain_120m.yaml
```

## Data preparation

Stream a bounded FineWeb-Edu slice. This command reads the upstream dataset lazily and stops
after 5,000 accepted documents or 40 million characters, whichever is reached first:

```bash
python scripts/download_fineweb_edu.py --output-dir data/raw/fineweb_edu
```

FineWeb-Edu is published by Hugging Face under the ODC-By license; see the
[official dataset card](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu).

Train a 32K BPE tokenizer from newline-delimited JSON:

```bash
python scripts/train_tokenizer.py \
  --input data/raw/train.jsonl \
  --output data/tokenizer/tokenizer.json \
  --text-field text \
  --vocab-size 32000
```

Convert the corpus to a memory-mapped `uint32` token stream:

```bash
python scripts/prepare_data.py \
  --input data/raw/train.jsonl \
  --tokenizer data/tokenizer/tokenizer.json \
  --output data/processed/train.bin \
  --text-field text
```

Repeat `prepare_data.py` for `validation.jsonl`, then run the local real-corpus experiment:

```bash
python -m miniscale.train --config configs/local_4060_fineweb_120m.yaml
```

Data, checkpoints and credentials are intentionally excluded from Git.

## Reproducibility contract

Every published experiment must report:

1. model and data configuration;
2. exact Git commit;
3. GPU model/count, driver, CUDA and PyTorch versions;
4. global batch size and processed tokens;
5. validation loss, tokens/s, peak memory and step-time distribution;
6. whether checkpoint and failure-recovery tests passed;
7. limitations, including the fact that single-node experiments are not equivalent to
   production-scale multi-node pretraining.

## Roadmap

- [x] CPU smoke test and baseline model
- [x] Local RTX 4060 Laptop BF16 systems baseline
- [x] DDP-compatible training loop
- [x] deterministic checkpoint/resume
- [ ] Validate BF16 training on one cloud GPU
- [ ] Add and validate FSDP full-shard checkpointing
- [ ] Add DeepSpeed ZeRO-2/3 configurations
- [ ] Run fixed-global-batch 1/2/4-GPU scaling experiments
- [ ] Run Qwen2.5-1.5B continual pretraining experiment
- [ ] Add profiler traces and failure-injection report
- [ ] Add optional frozen-vision-encoder multimodal alignment experiment
