# Two-GPU 1B-step PQL optimization experiment

Preflight (2026-11-09): a two-A6000 NCCL run reached 10,048 received rows for a
10,000-row budget, stopped both ranks, retained identical rank model hashes,
and saved three periodic learner snapshots plus final artifacts. The final actor
matched the learner snapshot and passed the evaluation architecture loader.
This validates stopping/artifacts at small geometry, not full-scale throughput.

This experiment uses the same saved SAC/TES trial-98 source as the preceding
profiling investigation, copied into `source_sac.yaml`. It is not a reproduction
of every official PQL tuning setting.

| Setting | Stock baseline | Optimized run |
|---|---|---|
| GPUs | Same two A6000s, sequential runs | Same |
| Environments / agent slots per rank | 20 / 3,200 per environment | Same |
| Replay rows per rank / refresh | 2,048,000 / 50% | Same |
| Logical / microbatch per rank | 128,000 / 32,000 | Same; matches PPO |
| Epochs per cycle | 3 | 3 |
| Return estimator | PQL, horizon 128, lambda 0.95 | Same |
| Targets / actor Q | Recomputed during training | Fixed once per cycle |
| Replay storage | Pinned CPU, original configuration | GPU |
| Filtering | None | Target-advantage EWMA, beta .25, scale .01 |
| Learning-rate schedule budget | 1B received rows | 1B received rows |
| Stopping limit | 1 warmup + 3 measured cycles | 1B global received rows |

The replay remains SAC-sized. PPO's full rollout is 8,192,000 rows per rank;
matching that is a different experiment and may exceed GPU replay capacity.
SAC gamma .999, tau .005, actor/critic LR .0005, alpha LR .0003, adaptive target
entropy settings, reward/environment configuration, and bfloat16 precision are
preserved. PQL uses raw one-step replay. Evaluation and remote logging are off.

Warmup is real training and counts toward the 1B budget; it is excluded only
from steady-state timing summaries. Received steps are globally aggregated
simulator slots, not inserted replay rows or optimizer exposures. Collection
can overshoot the limit by a receive group. The optimized cycle limit is a
10,000-cycle safety bound, not a substitute for checking `budget_reached`.

Periodic and final learner artifacts are saved outside timed phases. They omit
replay and simulator state and are not exact-resume checkpoints. The final actor
artifact is available for later evaluation. No policy-quality claim is implied
by a throughput result.

```bash
CUDA_VISIBLE_DEVICES=4,5 bash project/PPO_SAC_Profiling/large_run_1b/launch_2gpu.sh \
  project/PPO_SAC_Profiling/results/pql_1b_2gpu_UNIQUE_NAME
```

The launcher runs the short baseline then the optimized run, stops on failure,
and writes a comparison when both finish. It copies configurations and an exact
profiling source archive into the output directory. Outputs are ignored by Git.
Inspect `active_case.txt`, each case's `progress.json`, and its console log for
status. Compare the first three measured optimized cycles with the baseline for
an early-training comparison; compare the full optimized run separately, because
the policy, retained fraction, and collection cost change during training.

Live comparison (only completed cycles are read):

```bash
source .venv/bin/activate
python project/PPO_SAC_Profiling/large_run_1b/compare_progress.py PATH_TO_RUN
```
