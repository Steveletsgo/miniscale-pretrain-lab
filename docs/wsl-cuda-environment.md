# WSL CUDA operator-development environment

This document records the local environment used for CUDA, Triton and PyTorch extension
experiments. It is a development and correctness-validation environment, not evidence of
multi-node or production-scale training experience.

## Storage layout

- WSL distribution: `Ubuntu-AI` (Ubuntu 24.04 LTS)
- Distribution storage: `D:\WSL\Ubuntu-AI\ext4.vhdx`
- Repository: `D:\AI-Projects\miniscale-pretrain-lab`
- Linux virtual environment: `/home/ai/.venvs/miniscale`
- Linux model/tool caches: `/home/ai/.cache`

Both the WSL virtual disk and the repository live on `D:`. The Windows system drive is not
used for Linux packages, virtual environments, compiler caches or model caches.

## Validated versions

| Component | Version / device |
|---|---|
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU, 8,188 MiB, compute capability 8.9 |
| Windows NVIDIA driver | 591.74 |
| CUDA Toolkit | 12.8 (`nvcc` 12.8.93) |
| PyTorch | 2.7.0+cu128 |
| Triton | 3.3.0 |
| NCCL | 2.26.2 |
| GCC / G++ | 13.3.0 |
| CMake | 3.28.3 |
| Ninja | 1.11.1 |
| Nsight Systems | 2024.6.2 |
| Nsight Compute | 2025.1.1 |

The CUDA toolkit was installed from NVIDIA's WSL Ubuntu repository using the
`cuda-toolkit-12-8` package. No Linux display driver package was installed: WSL uses the
Windows host driver through GPU paravirtualization.

## Enter and activate the environment

Open PowerShell and enter the distribution:

```powershell
wsl -d Ubuntu-AI
```

Then activate the compiler paths, caches and Python environment:

```bash
cd /mnt/d/AI-Projects/miniscale-pretrain-lab
source scripts/activate_wsl_env.sh
source /home/ai/.venvs/miniscale/bin/activate
```

Confirm the GPU and toolchain:

```bash
nvidia-smi
nvcc --version
python -c "import torch, triton; print(torch.__version__, triton.__version__); print(torch.cuda.get_device_name())"
```

## Validation commands

### Native CUDA

```bash
mkdir -p /tmp/miniscale-cuda
nvcc -O3 -std=c++17 -arch=sm_89 \
  benchmarks/cuda/vector_add.cu \
  -o /tmp/miniscale-cuda/vector_add
/tmp/miniscale-cuda/vector_add
```

Validated result: 1,048,576 elements and zero maximum error.

### PyTorch C++/CUDA extension

```bash
python benchmarks/cuda/torch_extension_smoke.py
```

This compiles a C++ binding and an SM 8.9 CUDA kernel with Ninja, links the extension
against PyTorch CUDA, loads it in Python and checks its output against `torch.add`.
Validated result: 1,048,576 elements and zero maximum error.

### Triton JIT kernel

```bash
python benchmarks/triton/vector_add.py
```

One local environment-validation run on 16,777,216 `float32` elements reported:

| Implementation | Median time | Effective bandwidth |
|---|---:|---:|
| Triton vector add | 1.7015 ms | 118.32 GB/s |
| PyTorch eager add | 2.3326 ms | 86.31 GB/s |

These are single-machine smoke-benchmark numbers. They confirm that Triton JIT compilation
and execution work; they are not a stable performance claim and should not be compared across
machines without fixed power, clock, warm-up and repetition controls.

### Repository regression

```bash
pytest -q --basetemp /tmp/miniscale-pytest
ruff check src tests scripts benchmarks
```

Validated result on 2026-09-18: 7 tests passed and all Ruff checks passed on both Windows and
WSL.

## Profiling and debugging limitations under WSL

- Nsight Systems records CUDA API calls, including kernel launches and synchronization, but
  this machine did not expose CUDA GPU kernel timeline data to the report.
- System-wide CPU sampling is disabled by the WSL kernel's performance-counter policy. Process
  tree CPU tracing is available.
- Compute Sanitizer currently fails to initialize the WDDM debugger interface. NVIDIA's
  `EnableDebuggerInterface.bat` would need to be run from an elevated Windows shell before
  claiming sanitizer validation.

Therefore, kernel correctness is currently validated with explicit reference comparisons and
synchronization. No claim is made that Compute Sanitizer or a complete GPU timeline has passed.

## Next experiment

The next bounded milestone is a fused Triton RMSNorm or SwiGLU kernel with:

1. correctness tests against the PyTorch reference in FP32 and BF16;
2. shapes representative of this repository's Transformer;
3. warm-up and repeated latency measurements;
4. effective-bandwidth reporting and profiler evidence where WSL permits it;
5. an honest comparison against PyTorch eager and `torch.compile`.
