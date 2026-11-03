# Frozen SAC critic calibration

This folder measures final Round2 TES-SAC trial 98 critics against sampled
discounted returns, grouped by distance to the next waypoint. See [PLAN.md](PLAN.md)
for the original analysis contract and [REPORT.md](REPORT.md) for results when
collection and analysis finish.

## Files

| File | Purpose |
|---|---|
| `collect.py` | Load the matching final actor, twin critics, target critics, and alpha; collect frozen categorical-policy trajectories |
| `returns.py` | Compute reward and soft returns with explicit terminal, timeout, and missing-continuation boundaries |
| `diagnostic.py`, `native/` | Private read-only native metadata export; shared simulator sources and extension remain unchanged |
| `analyze.py` | Distance bins, scene-cluster uncertainty, reliability, confounders, and censored-tail diagnostics |
| `runs/`, `analysis/`, `tests/` | Recorded audit/data files, resulting figures/tables, and local correctness checks |

## Reproduce collection

From the repository root, activate `.venv`. The following is one independent
training-matched rollout, using only physical GPU 6. Choose a new output path
for each run; existing directories are rejected.

```bash
source .venv/bin/activate
CUDA_VISIBLE_DEVICES=6 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
python -m project.soft_actor_critic_q_calibration.collect \
  --output project/soft_actor_critic_q_calibration/runs/main_gpu6 \
  --env-seed 61011 --action-seed 71011 --steps 2560 --record-stride 8
```

The matching GPU 7 run uses `CUDA_VISIBLE_DEVICES=7`, output `main_gpu7`,
environment seed 61012, and action seed 71012. These are independent initializations,
not repeated futures from identical states. Both preserve the saved 3200-agent
Drive configuration, 0.3-second timestep, reward randomization, and map selection.

Return arithmetic uses every valid transition. Scalar records are saved every
eight ticks only after returns have been calculated; this does not change the
return horizon. `transitions.npz` contains no pickled objects or bulk observations.
Native initialization seeds remain uint64, avoiding float32 rounding.

## Reproduce analysis and checks

```bash
PYTHONDONTWRITEBYTECODE=1 python -m project.soft_actor_critic_q_calibration.analyze \
  --runs 'project/soft_actor_critic_q_calibration/runs/main_gpu*' \
  --output-dir project/soft_actor_critic_q_calibration/analysis
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q \
  project/soft_actor_critic_q_calibration/tests \
  -o cache_dir=project/soft_actor_critic_q_calibration/.pytest_cache
```

Exclude `pilot_gpu6` from the main analysis: its 64-tick collection cap leaves
almost all returns censored. Main-run audit files record checkpoint/config
hashes, code hashes, seeds, devices, exact saved environment configuration,
observed boundaries, and before/after checks that model state stayed frozen.

## Interpretation

The primary reference is soft return, excluding entropy of the initial action
and including subsequent sampled-action entropy at the checkpoint's fixed alpha.
Reward-only return is reported separately. Complete returns are available only
for suffixes reaching true terminals; that subset is selected by its outcome.
Timeout/gap/cap suffixes remain censored. Appending a target-critic tail is a
secondary diagnostic, not independent Monte Carlo truth.

The road-network distance is the raw lane-graph matrix entry, not remaining
distance along the original randomly generated route. Euclidean distance is a
separate comparison. Native goal state and returned observations can disagree
on goal-regeneration rows; those rows are flagged and sensitivity is reported.
This experiment tests the sampled policy under its training configuration, not
the saved mean-action CARLA benchmark score.
