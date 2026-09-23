# Two-GPU microbatch probe — 2026-09-22

- **Largest practical tested size: 64,000 windows/GPU**, with two accumulation passes per update.
- **Largest passing size: 80,000**; 88,000 and 96,000 failed with CUDA OOM. The exact boundary between 80,000 and 88,000 was not searched.
- **Keep 1,024 for now if speed is the objective:** larger microbatches did not improve measured throughput. Official config was not changed.
- Effective optimizer batch stays **128,000/GPU, 256,000 globally**. Changing microbatch size also changes the local variance-penalty statistics.

## Measurements

Two RTX A6000s, GPUs 0 and 1; CUDA reports 44.55 GiB usable per device. Every passing row below ran three complete optimizer updates. Memory is the maximum across updates and ranks. Time averages updates 2–3 using the slower rank; the first update is warmup.

| Microbatch/GPU | Peak allocated GiB | Peak reserved GiB | Seconds/update | Global windows/s |
|---:|---:|---:|---:|---:|
| 1,024 | 0.70 | 0.91 | 5.14 | 49,856 |
| 32,000 | 16.06 | 20.27 | 5.47 | 46,825 |
| 64,000 | 31.94 | 36.13 | 5.41 | 47,298 |
| 80,000 | 38.91 | 43.78 | 5.57 | 45,987 |
| 88,000 / 96,000 | OOM | OOM | — | — |

- Table uses repeated existing toy windows. The probe calls production `_training_update`: FP32 student, frozen resident teacher, target encoder, DDP, losses, backward, AdamW, clipping and EMA. No simulator collection or official training is launched.
- Additional **64,000 verification on the stopped run's completed rank-1 collection** passed three full updates, using distinct rank index ranges. Peak allocated/reserved memory matched the toy result. Device usage after updates was approximately **36.68 GiB**, leaving **7.87 GiB (17.7%)**. Later updates took approximately **5.56 seconds**.
- At 80,000, device usage reached **44.34 GiB**, leaving only **0.22 GiB**. This is unsuitable for a full run with additional allocations.
- These are short update benchmarks, including data gathering and transfers, not full-epoch throughput or learning-quality measurements. Repeated indices become cached; full-buffer random reads, simulator residency and evaluation are not exercised.
- The stopped official run had an empty metrics log and no rank-0 train manifest. Its reported 11% usage may have been during collection; teacher inference has a separate `collection.inference_batch_size=2048` setting.

## Reproduce the capacity check

Run from the repository root. Each candidate uses a fresh process; an OOM candidate exits unsuccessfully. Results contain per-rank JSON. The manifest is opened read only.

```bash
source .venv/bin/activate
CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=1 torchrun \
  --standalone --nnodes=1 --nproc_per_node=2 \
  -m project.jepa_distill.profiling.probe_microbatch \
  --manifest experiments/jepa_distill/datasets/run_2026_0922_2052/rank_001/train/round_0000/manifest.json \
  --microbatch-size 64000 --full-batch \
  --output-dir experiments/jepa_distill/microbatch_probe/recheck64000
```

Raw results and config/source snapshots: `experiments/jepa_distill/microbatch_probe/`. `full1024`, `full32000`, `mb64000`, `mb80000` supply the comparison; `real64000` supplies the real-data verification. `mb1024` is an initial screening result with a smaller effective batch and is excluded from the table.

## Optional full-run override

The user starts official training. Use a **new run ID** for a fresh run:

```bash
CUDA_VISIBLE_DEVICES=0,1 project/jepa_distill/scripts/train_2gpu.sh \
  --run-id YOUR_NEW_RUN --set training.microbatch_size=64000
```

## Update bottleneck profile

`--profile` records CPU/CUDA events for the last update after warmup. Results for a full 128,000-window/GPU update at microbatch 1,024 are in `experiments/jepa_distill/microbatch_probe/profile1024/`.

| Rank-0 event | Measured time | Interpretation |
|---|---:|---|
| CPU `index_copy_` | 1.31 s | Pooled dataset copies gathered windows into another CPU batch |
| CPU CUDA memcpy API | 0.67 s | Transfer submission, staging and possible waiting |
| CPU stream synchronization | 0.59 s; 2,813 calls | Blocking transfers, scalar checks and metric extraction |
| GPU `addmm` + `mm` | 1.06 s | Dense forward/backward matrix multiplication |
| GPU layer norm forward/backward | 0.33 s | Normalization work |

- CPU and GPU times overlap and profiler overhead affects timing; do not sum these into an update duration. CUDA host-to-device copies themselves took 0.28 s; NCCL all-reduce events took 0.11 s on rank 0.
- The measured update has substantial batch assembly/transfer/synchronization costs alongside neural-network compute. GPU memory capacity is not the demonstrated speed bottleneck.
- First optimization candidates: reduce pooled CPU batch copies, prefetch/overlap transfers, and consolidate scalar synchronization while preserving failure checks. Also, training currently transfers all five observations although the encoder uses only the first and last.
- This cached toy profile does not establish full-buffer random disk-read cost. No production optimization was applied.
