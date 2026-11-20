# TES filtering profile

This wrapper reuses `project/PPO_SAC_Profiling/profile_training.py` and patches
only the in-memory profiling config and trainer instance. It does not change the
production learner or experiment configs.

Run each case on the same hardware with an identical source config, checkpoint,
seed, batch sizes, cycle counts, and output directory shape. Use a fresh output
directory for every invocation:

```bash
source .venv/bin/activate
torchrun --standalone --nproc-per-node=2 \
  project/soft_actor_critic_experience_filtering/profiling/profile_tes_filter.py \
  --case filter \
  --source-config /path/to/resolved_filtered_tes/config.yaml \
  --weights-from /path/to/final/trainer_state.pt \
  --output-dir /tmp/tes-filter-profile-filter \
  --algorithm sac --device cuda --cycles 2 --warmup 1 \
  --num-envs 20 --agents 3200 --replay-capacity 2048000 \
  --logical-batch 128000 --microbatch 64000 --update-epochs 1 \
  --compile on --replay-location gpu --total-agent-steps 4000000000
```

Repeat with `--case baseline` and `--case all_keep`, changing only `--case` and
`--output-dir`. The wrapper forwards the normal harness CLI. `baseline` sets
`train.experience_filter: null`; `filter` uses the source filter config as-is;
`all_keep` sets only `threshold_scale: 0.0`, which retains all finite
nonnegative TD-error scores. The source config must contain an enabled TES
filter recipe. `--source-config` and `--weights-from` are required.

All cases load policy, online critic, target critic, and `log_alpha` from the
same learner-only checkpoint. Optimizers, schedulers, adaptive target-entropy
controller state, counters, RNG, and replay remain fresh. `--restore-filter-state`
is optional for `filter` only; it restores the checkpoint's filter EMA and
counters. It is incompatible with `all_keep`, whose changed threshold scale
does not match the checkpoint filter config.

For late-run cost probes, add `--restore-update-state` to ALL cases. This
restores Adam moments, learning rates, LR schedules and the entropy controller.
Controller observations receive the checkpoint's received-step offset so its
forced-decay clock remains monotone; the harness's received-step counters still
start at zero. This is a profiling initialization, not an exact resume. The
old replay and simulator state are unavailable. Do not expect historical
retention to be reproduced by a freshly collected replay. A weights-only probe
can distort retention substantially through fresh Adam updates at the initial LR.

Each rank writes `tes_filter_cycle_NNN_rank_NNN.json` with actual filter
candidate/retained counts and fractions, train logs, direct `_critic_loss_sum`
and `_policy_loss_rows` exposures, and collection/training wall durations.
The learner's candidate scoring and target construction still visit every
candidate. Thus these data measure the controlled learner profile, not a
speedup inferred from retention alone.

Use a separate `--detailed` pass. It adds synchronized wall spans for target
construction, candidate scoring, filter selection, filtered critic/actor
training, baseline critic/policy loss calls, and optimizer steps. JSON labels
inclusive and exclusive spans; nested inclusive values overlap and must not be
summed. CUDA synchronization wait and method-call counts are reported as
instrumentation overhead indicators. Detailed timings are not directly
comparable to plain timings, and backward is not timed separately.

Sidecars are buffered until the benchmark finishes so their file writes are
outside the timed phases. The wrapper adapts the harness's exposure counters
for the production filter; the original harness only understands its older
profiling-specific filter. This wrapper requires one epoch and shared
`actor_critic` filtering.

Harness-reported cycle timing includes collection and optimization separately.
The initial checkpoint is a weights warm start rather than full learner resume,
so this profile does not reproduce the checkpoint's replay, optimizer, or
adaptive target-entropy trajectory.

`analyze_history.py` compares completed TES trial directories, validates their
two-cycle evaluation cadence and contiguous logs, and saves phase estimates,
actual wall times, received-step windows and retained exposures. The
`performance/eval` key means collection; scheduled benchmark evaluation and its
checkpoint are outside that timer. Timers sample every fifth cycle and scale by
five. The residual wall time must not be called a directly measured evaluator
duration.

`summarize_probes.py ROOT` validates finite final parameters and exact agreement
between per-rank actor/critic row counters and the main harness retained counts
for `baseline_restored`, `filter_restored`, `all_keep_restored` and
`filter_detailed`. It writes `probe_comparison.json` under ROOT. Detailed method
timings are inclusive and overlap (for example targets contain compute_targets).
