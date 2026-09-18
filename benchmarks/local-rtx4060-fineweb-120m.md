# Local RTX 4060 FineWeb-Edu learning baseline

This run validates the complete real-data path: bounded streaming download, train/validation
split, tokenizer training, binary token preparation, BF16 next-token training, evaluation and
full-state checkpointing. It is a short learning-curve experiment, not a converged pretrained
model.

## Reproducibility

| Field | Value |
| --- | --- |
| Date | 2026-09-18 |
| Source commit | `122613b768422a052803b45905ed47953c77a513` |
| Training configuration | `configs/local_4060_fineweb_120m.yaml` |
| Dataset | HuggingFaceFW/fineweb-edu, `sample-10BT`, revision `v1.4.0` |
| Dataset license | ODC-By |
| Accepted documents | 4,750 train / 250 validation |
| Prepared tokens | 4,966,769 train / 229,723 validation |
| Tokenizer | Locally trained 32K Byte-Level BPE |
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU, 8,188 MiB |
| Driver | 591.74 |
| PyTorch / CUDA runtime | 2.7.0+cu128 / 12.8 |
| Parameters | 117,787,392 |
| Precision | BF16 autocast |
| Sequence length | 256 |
| Micro batch / accumulation | 2 / 4 |
| Tokens per optimizer step | 2,048 |
| Optimizer steps / processed tokens | 100 / 204,800 |

The source documents, tokenizer, binary streams and checkpoints remain under Git-ignored
`data/` and `runs/` directories. They can be regenerated using the commands in the README.

## Learning results

| Metric | Result |
| --- | ---: |
| Training loss, step 1 | 10.5533 |
| Training loss, step 100 | 7.4841 |
| Mean training loss, steps 1-10 | 9.8945 |
| Mean training loss, steps 91-100 | 7.5228 |
| Validation loss, step 25 | 7.8717 |
| Validation loss, step 50 | 7.8048 |
| Validation loss, step 75 | 7.6615 |
| Validation loss, step 100 | 7.5760 |
| Validation perplexity, step 100 | 1,950.87 |

The monotonically improving checkpoint evaluations show that the early loss reduction also
appears on held-out documents. The high final perplexity is expected after only 204.8K training
tokens and should not be compared with fully trained language models.

## Systems results

| Metric | Result |
| --- | ---: |
| Mean tokens/s, steps 2-100 | 3,706.19 |
| Median tokens/s, steps 2-100 | 3,319.94 |
| Peak allocated CUDA memory | 2.4604 GiB |
| Peak reserved CUDA memory | 2.6875 GiB |
| Full-state checkpoint size | approximately 1,348 MiB |

As in the synthetic baseline, throughput fell after the initial phase without a corresponding
memory increase. This machine needs a follow-up run with concurrent GPU temperature, SM clock,
power and utilization logging before attributing the change to thermal or power throttling.

## What this experiment does and does not establish

It establishes that the repository can reproducibly execute a real-corpus pretraining loop and
measure held-out loss on a consumer GPU. It does not establish multi-GPU scaling, convergence,
benchmark quality, optimal hyperparameters or production reliability. Those claims require
token-budget-matched longer runs and controlled 1/2/4-GPU experiments.
