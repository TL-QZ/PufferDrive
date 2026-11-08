# Historical PPO/SAC run evidence

Generated from local W&B binary histories and saved run metadata. The analyzer used the installed W&B SDK datastore parser in read-only mode; no W&B API, network request, or training job was started.

## Run summary

| Run | Global received agent steps | Logged rows / cycles | Kept fraction: cycle mean / final | Filtering samples | Eval rows: observed / expected | Hardware recorded |
|---|---:|---:|---:|---:|---:|---|
| PPO baseline, 2026-11-06 | 4,014,080,000 | 245 / 245 | 25.89% / 24.61% | 777,810,592 all-rank kept | 123 / 123 | NVIDIA RTX A6000 |
| PPO baseline, 2026-08-26 | 10,010,624,000 | 611 / 611 | 23.95% / 39.19% | 1,843,680,894 all-rank kept | 7 / 7 | NVIDIA RTX A6000 |
| SAC round 2 TES, 2026-10-29 trial 98 | 4,000,032,000 | 1,615 / 1,615 | not logged | not applicable | 808 / 808 | NVIDIA RTX A5000 |

## Filtering history and sample accounting

- **PPO baseline, 2026-11-06:** 245 of 245 training rows include `losses/kept_fraction`. The arithmetic mean across cycles is 25.8883%, the rank-0 valid-sample-weighted mean is 26.2315%, and the final-cycle value is 24.6122%. The full curve and per-cycle estimates are in `historical_summary.json`.
  Reconstructing sample counts from the logged `masked_fraction` and `kept_fraction` gives 1,482,588,314 valid and 388,905,296 kept transitions on rank 0; the DDP minimum-count trim means the all-rank kept count is 777,810,592. The global valid denominator is unavailable because rank-1 masked counts were not logged.
- **PPO baseline, 2026-08-26:** 611 of 611 training rows include `losses/kept_fraction`. The arithmetic mean across cycles is 23.9512%, the rank-0 valid-sample-weighted mean is 24.3351%, and the final-cycle value is 39.1922%. The full curve and per-cycle estimates are in `historical_summary.json`.
  Reconstructing sample counts from the logged `masked_fraction` and `kept_fraction` gives 3,788,111,990 valid and 921,840,447 kept transitions on rank 0; the DDP minimum-count trim means the all-rank kept count is 1,843,680,894. The global valid denominator is unavailable because rank-1 masked counts were not logged.
- **SAC round 2 TES, 2026-10-29 trial 98:** PPO advantage filtering does not apply, so no kept fraction is logged. SAC directly records 19,845,120,000 global replay-sampled transition exposures, 3,307,875,030 global valid inserts, a final replay size of 2,048,000, and 77,520 learner updates per rank.

## Epoch counters and evaluation counts

PPO logs `epoch` as `dist_sum(self.epoch)` in `pufferlib/pufferl.py:866`, while one call to `PuffeRL.train()` increments the local epoch once. Both PPO runs have two-rank epoch deltas, so divide the displayed epoch by two to recover cycles. Evaluation scheduling checks each rank-local epoch. The completion evaluator runs once more if the final cycle is off an interval boundary.

- **PPO baseline, 2026-11-06:** 245 cycles, interval 2, expected 123 evaluations; W&B records 123.
- **PPO baseline, 2026-08-26:** 611 cycles, interval 100, expected 7 evaluations; W&B records 7.
- **SAC round 2 TES, 2026-10-29 trial 98:** 1,615 cycles, interval 2, expected 808 evaluations; W&B records 808.

## Learner exposure and profiled time

`performance/train` is the primary total training-phase estimate. It measures the outer `train()` region; `performance/learn` is a narrower nested gradient/update region and is shown separately below.

| Run | Global received transitions | Global learner transition exposures | Exposures / received | `performance/train` estimate (rank 0) | Estimated µs / received | Estimated µs / exposure |
|---|---:|---:|---:|---:|---:|---:|
| PPO baseline, 2026-11-06 | 4,014,080,000 | 2,333,431,776 | 0.581 | 5,989.7 s | 1.492 | 2.567 |
| PPO baseline, 2026-08-26 | 10,010,624,000 | 5,531,042,682 | 0.553 | 13,944.9 s | 1.393 | 2.521 |
| SAC round 2 TES, 2026-10-29 trial 98 | 4,000,032,000 | 19,845,120,000 | 4.961 | 162,189.7 s | 40.547 | 8.173 |

| Run | `performance/learn` estimate (rank 0) | Estimated µs / received | Estimated µs / exposure |
|---|---:|---:|---:|
| PPO baseline, 2026-11-06 | 3,959.8 s | 0.986 | 1.697 |
| PPO baseline, 2026-08-26 | 9,318.6 s | 0.931 | 1.685 |
| SAC round 2 TES, 2026-10-29 trial 98 | 57,322.9 s | 14.331 | 2.889 |
- SAC replay exposure is 8.50x the inferred exposure of PPO baseline, 2026-11-06 (SAC 19,845,120,000; PPO 2,333,431,776).
- SAC has 27.08x the cumulative `performance/train` estimate of PPO baseline, 2026-11-06; divided by exposure, that is 3.18x. Hardware and workloads differ, so this is descriptive only.
- SAC replay exposure is 3.59x the inferred exposure of PPO baseline, 2026-08-26 (SAC 19,845,120,000; PPO 5,531,042,682).
- SAC has 11.63x the cumulative `performance/train` estimate of PPO baseline, 2026-08-26; divided by exposure, that is 3.24x. Hardware and workloads differ, so this is descriptive only.

Both profile values are cumulative rank-0 estimates from each saved run summary. `performance/train` encloses the full trainer `train()` call, including the nested train-misc bookkeeping; `performance/learn` is the inner gradient/update region. PPO exposure is inferred as the globally retained sample count times `train.update_epochs`; SAC exposure comes from the direct global replay counter. The per-row ratios normalize the outer `performance/train` estimate by global received rows and global learner exposures.

These normalized times remain descriptive estimates. The profiler samples every fifth cycle, multiplies each sampled interval by five, and synchronizes CUDA at profiling boundaries. PPO used NVIDIA RTX A6000 hardware while SAC used RTX A5000 hardware, and their trainer paths differ. The ratios are not same-hardware speed measurements or controlled estimates of an algorithm-only effect.

## Provenance and limits

- Each primary `.wandb` file is identified by repository-relative path, byte size, and SHA-256 in the JSON. Config and final summary values are copied from its local W&B protobuf records.
- The 2026-08-26 PPO run has a second local W&B file dated 2026-09-11 with the same run ID. It has one supplemental history record and no filter metric; it is recorded separately and not merged into the complete 611-cycle history.
- PPO filter history is rank-0 telemetry. The logged kept count is trimmed to the DDP global minimum and therefore is common across ranks, but total valid samples across ranks cannot be recovered from rank-0-only masked metrics. Fractions are stored as floating-point values; counts are reconstructed and rounded to the nearest transition.
- W&B hardware metadata reports host GPU inventory (eight A6000s for PPO, eight A5000s for SAC), not allocated GPU count. The SAC trial has separate rank records showing visible IDs 2 and 3; physical IDs for the PPO runs were not recorded in the local W&B files.
- `performance/*` values describe sampled trainer profiler regions. They do not isolate benchmark evaluation work in `run_training_evaluation`; evaluation counts are reported separately and no evaluation duration is inferred here.
- PPO and SAC used different GPU models and different training configurations. These artifacts establish workload and evaluation counts, not a same-hardware speed comparison or a direct wall-time estimate.
- W&B `summary` values are end-of-run values. The cycle mean and per-cycle counts above use the full stored history, so they are distinct from the final summary fraction.

## Reproduction

From the repository root, with the project virtual environment active:

```bash
source .venv/bin/activate
python -B project/PPO_SAC_Profiling/analyze_history.py
```

The script writes `historical_summary.json` and `historical_summary.md` next to itself. `-B` suppresses bytecode artifacts.
