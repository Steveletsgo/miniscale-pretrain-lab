# Local RTX 4060 Laptop systems baseline

This is a systems-validation run, not a model-quality result. The token stream is
deterministic synthetic random data, so loss convergence is neither expected nor claimed.

## Reproducibility

| Field | Value |
| --- | --- |
| Date | 2026-09-18 |
| Source commit | `e28e7b595311bdfc944f62d42437203a49d307e7` |
| Configuration | `configs/local_4060_120m.yaml` |
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU, 8,188 MiB |
| Driver | 591.74 |
| PyTorch / CUDA runtime | 2.7.0+cu128 / 12.8 |
| Precision | BF16 autocast |
| Parameters | 117,787,392 |
| Sequence length | 256 |
| Micro batch / accumulation | 2 / 4 |
| Tokens per optimizer step | 2,048 |
| Optimizer steps | 20 |

Command:

```bash
python -m miniscale.train \
  --config configs/local_4060_120m.yaml \
  --run-dir runs/local_4060_120m_e28e7b5
```

## Measurements

The first step is excluded from steady-state summaries because it includes CUDA warm-up.

| Segment | Mean tokens/s | Median tokens/s | Range | Mean step time |
| --- | ---: | ---: | ---: | ---: |
| Steps 2-20 | 5,374.84 | 3,540.39 | 3,292.58-8,387.84 | 0.4517 s |
| Steps 2-9 | 8,022.92 | 8,063.99 | 7,360.57-8,387.84 | 0.2556 s |
| Steps 11-20 | 3,439.82 | 3,440.46 | 3,292.58-3,661.17 | 0.5958 s |

- Peak allocated CUDA memory: 2.4604 GiB.
- Peak reserved CUDA memory: 2.6875 GiB.
- Validation loss: 10.5327 at step 10 and 10.5309 at step 20.
- Final full-state checkpoint: approximately 1,348 MiB.
- Checkpoint round-trip and CUDA `map_location` recovery are covered by the test suite.

## Interpretation and limitations

Throughput changed sharply after step 10 while reported CUDA memory stayed flat. The run did
not sample GPU temperature, clocks or board power during training, so it cannot distinguish
thermal or power throttling from another transient system effect. This result is therefore
reported in two phases rather than summarized by only the faster interval. A controlled rerun
should log temperature, SM clock, power draw and GPU utilization at a fixed cadence.

The random-token loss is only a numerical-stability signal. It is close to the expected scale
for a 32K-way next-token prediction problem and must not be presented as evidence of language
learning. Model-quality experiments require a real corpus, a trained tokenizer, held-out data
and token-budget-matched comparisons.
