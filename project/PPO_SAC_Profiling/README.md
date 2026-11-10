# PPO / SAC profiling

This folder contains bounded real-Drive benchmarks, historical workload analysis,
and isolated SAC replay optimizations. Production learner files are unchanged.

Start with [REPORT.md](REPORT.md) for findings and limitations, then
[results/matched_summary.md](results/matched_summary.md) for the controlled matrix.
The follow-up [fixed-target experiment](FIXED_TARGET_REPORT.md) tests PPO-like
target reuse in SAC. It preserves the cycle-start Bellman formula but changes
the fresh-target update sequence; these variants are opt-in algorithmic experiments.
The [actor-Q follow-up](FIXED_ACTOR_Q_REPORT.md) also freezes cycle-start online
Q scores for actor training, with current actor probabilities and alpha.

The [PQL/filtering extension](PQL_FILTER_REPORT.md) adds ordered PQL target
caching and opt-in advantage-based row selection for throughput experiments.
Its [implementation contract](PQL_FILTER_PLAN.md) separates target correctness
from the deliberate changes to training workload. Raw `results/` outputs are
local and ignored by Git; tracked reports summarize completed measurements.

## Reproduce a bounded run

Activate the repository environment, select idle GPU(s), and use a new output
directory. The harness refuses to reuse an existing directory.

```bash
source .venv/bin/activate
CUDA_VISIBLE_DEVICES=4 python project/PPO_SAC_Profiling/profile_training.py \
  --profile-config project/PPO_SAC_Profiling/configs/sac_single_gpu.yaml \
  --output-dir project/PPO_SAC_Profiling/results/my_sac_probe
```

The checked-in presets are sizing/smoke recipes. The matched measurement matrix
uses full model architecture and production 128,000/32,000 optimizer/microbatch
sizes, with reduced replay/rollout size for controlled comparisons:

```bash
CUDA_VISIBLE_DEVICES=4 python project/PPO_SAC_Profiling/run_matrix.py \
  --output-dir project/PPO_SAC_Profiling/results/my_matrix \
  --cases ppo_filter ppo_all sac_stock sac_obs_only sac_cached sac_gpu
```

`run_matrix.py --production --ranks 2` selects 20 environments with 3,200 agents
per rank, PPO horizon 128, and SAC replay capacity 2,048,000 per rank. Select two
idle GPUs through `CUDA_VISIBLE_DEVICES`. Unfiltered distributed PPO is rejected
because the production masking path can produce different backward counts across
ranks; the single-GPU unfiltered control is supported.

The matrix also accepts `--ppo-weights PATH` and `--sac-weights PATH`, passing the
appropriate model-only initialization to each case. `--trace-cycle N` captures
that cycle separately for every selected case and excludes it from throughput.

For a trained-policy initialization, use `--weights-from <local-file>`: PPO accepts
its final model state dict, SAC accepts its learner-only trainer checkpoint. This
copies model weights only, not optimizer, replay, RNG, counters, alpha or schedule
state. It is a fresh bounded profiling experiment, not a resumed training run.

## Measurements and safety boundaries

- `--warmup` cycles are excluded; `--cycles` counts subsequent cycles. Compilation
  and initialization are not included in warmed throughput. Cold timings remain
  in the raw cycle files.
- CUDA is synchronized at collection and training boundaries. The stock internal
  profiler is off by default; `--stock-profiler on` enables its sampled internal
  synchronization. Include at least five total cycles to observe a sampled cycle.
- `--trace-cycle N` captures one zero-based cycle with CPU/CUDA operator data and
  a Chrome trace. Trace cycles are automatically excluded from throughput summaries.
- No remote logging, benchmark evaluation, model checkpoints or automatic resume
  occur. The original environment and network architecture are retained except
  explicitly recorded resource settings. Runs stop after a bounded cycle count.
- Small fresh-policy probes do not establish late-training filtering, convergence,
  policy quality, or a verified 4B wall time. Shared-host CPU/memory contention can
  still affect measurements even when the selected GPU is idle.

## Analysis and tests

```bash
python project/PPO_SAC_Profiling/analyze_history.py
python project/PPO_SAC_Profiling/analyze_runs.py \
  project/PPO_SAC_Profiling/results/my_matrix \
  --output project/PPO_SAC_Profiling/results/my_matrix_summary
python project/PPO_SAC_Profiling/analyze_traces.py \
  project/PPO_SAC_Profiling/results/operator_traces \
  --output project/PPO_SAC_Profiling/results/trace_summary
python -m pytest -q project/PPO_SAC_Profiling
```

Use `SAC_PROFILE_CUDA_TEST=1 CUDA_VISIBLE_DEVICES=4` with pytest to include the
small pinned-memory CUDA parity checks. Run those separately from timed GPU
benchmarks. The benchmark candidate implementations live in `sac_variants.py`;
the fixed-target variants support both n-step SAC and PQL. The gather-only
`obs_only` and `cached` variants still reject PQL. These are experimental copies
of the update loop, not production defaults.

For PPO-style adaptive filtering with GPU replay, use
`--profile-config project/PPO_SAC_Profiling/configs/pql_ewma_gpu.yaml`.
It keeps scores at or above 1% of the moving maximum, with EWMA beta 0.25;
see [the filtering report](PQL_FILTER_REPORT.md) for score definitions and
zero-survivor behavior.

Select `--cases sac_cached sac_fixed_cached` to compare CPU replay with and
without fixed-cycle targets, or `--cases sac_obs_only_gpu sac_fixed_gpu` for GPU
replay. Fixed cases include target precomputation in learner timing and audit
target staleness after timing. Use `analyze_fixed_targets.py INPUT --output PREFIX`
for actual bootstrap-work counts and audit results. Default matrices exclude
fixed-target cases.

Use `--cases sac_fixed_both_gpu` for GPU replay with both critic targets and
actor Q scores cached for a cycle. Compare against `ppo_filter`,
`sac_obs_only_gpu` and `sac_fixed_gpu` with the same geometry and initialization.
`--replay-capacity 2048000` tests full SAC replay size while retaining the smaller
simulator setup; it does not change PPO's rollout size.

See `PLAN.md` for acceptance criteria, `SOURCE_AUDIT.md` for code findings, and
`historical_summary.md` for full-history workload and timing reconstruction.
