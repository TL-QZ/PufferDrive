# nuPlan SDC fine-tuning at dt=0.3

Use `train/launch_nuplan_sdc_finetune_2gpu.sh 0` for source seed 0; seeds 1 and
2 use the same workflow. First run the bounded sizing command below on two
allocated GPUs. **GPU sizing is unverified:** on September 10, 2026, NVIDIA-SMI
could not communicate with the driver and PyTorch reported zero CUDA devices.
The [hardware preflight](../profiling/2026-09-10_cpu_preflight/summary.json) records
this limitation. No concurrency or microbatch has been selected from GPU measurements.

## Timing and isolation

Fine-tuning sets `resample_replay_to_dt=true`, `dt=0.3`, `scenario_length=66`,
and `resample_frequency=66`. Loading a 201-state, 0.1-second recording retains
indices **0, 3, ..., 198**: 67 states and 66 transitions over **19.8 seconds**.
There is no interpolation or shorter final transition. Source binaries remain
unchanged. Agent positions, headings, velocities, dimensions, validity, and
nonempty traffic-light histories use the same indices. Velocities remain m/s.
Geometry can be shared; temporal arrays remain private to each C environment.

The flag defaults to false in Hydra, Python, and C. Enabled loading requires
replay mode, `init_step=0`, `init_step_spread=false`, finite positive timesteps,
an integer `dt/log_dt` ratio (absolute tolerance `1e-5`), consistent logged
lengths, and enough states for the requested episode. Invalid creation reports
the scenario path and field. Enabled ground-truth export returns the initial
state plus each transition: `(agents, 1, 67)` for this experiment. For shorter
requested episodes it exports the corresponding initial state plus transitions.
The disabled path retains its existing export behavior.

All evaluation routes explicitly override the checkpoint with `dt=0.1` and
`resample_replay_to_dt=false`, including periodic evaluation, standalone
evaluation, and failure replay. nuPlan evaluation remains 200 transitions over
20 seconds. CARLA training settings, checkpoint architecture, controllers,
goals, reward formulas, and PPO are preserved.

New run output:

```text
experiments/baseline_run_sync_2026-08-24/nuplan_sdc_finetune_dt03/
  baseline_run_sync_2026-08-24_nuplan_sdc_finetune_dt03_TIMESTAMP_seedN/
```

Previous `nuplan_sdc_finetune` runs remain separate. The updated train, eval,
and render scripts all use the new root and prefix.

## Starting resource configuration — static estimates

The starting YAML uses 32 SDCs per Python environment, 20 environments and
20 workers per rank, with 10 environments per inference batch.

| Per-GPU quantity | CARLA launcher | Fine-tune starting value |
|---|---:|---:|
| Concurrent controlled agents | 64,000 | 640 SDCs |
| Observations per inference call | 32,000 | 320 |
| Rollout transitions | 8,192,000 | 640,000 |
| Gradient microbatch | 32,000 | 32,000 |
| Logical optimizer batch | 128,000 | 128,000 |

The logical batch uses four-way accumulation. Advantage filtering means actual
retained transitions and optimizer updates vary; the profiler records both.
The 1,000-step horizon divides the microbatch exactly and spans multiple reset
episodes. It adds neither recurrent training nor observation history.
The collector marks truncations as done before GAE. Existing value bootstrapping
uses the previous value as a proxy because the simulator resets internally;
this heuristic is unchanged.

The budget is **1B global transitions**, or 500M per rank with two GPUs.
Each outer epoch collects 1.28M globally. About 782 epochs collect 1.00096B,
including normal final-rollout overshoot. Evaluation every 20 epochs and
checkpointing every four correspond to 25.6M and 5.12M global transitions.
For concurrency 16/32/64, evaluation intervals are 40/20/10 and checkpoint
intervals are 8/4/2.

Learning rate `1e-4`, annealing, advantage filtering, value bootstrapping, and
weights-only initialization are retained. `env.num_maps=169715` exposes every
`.bin` scenario currently in `data/nuplan_train` to random-with-replacement
sampling. This makes the full pool eligible; it does not enforce a complete
ordered pass before a scenario can repeat.
Training uses three PPO epochs, `gamma=0.999`, `gae_lambda=0.95`, compilation,
BF16 AMP, and no CPU buffer offload. These are starting settings, not measured
GPU utilization or throughput claims.

## Bounded sizing — at most 45 minutes

From the repository root, activate the venv and name the source run explicitly:

```bash
source .venv/bin/activate
CUDA_VISIBLE_DEVICES=0,1 python project/baseline_run_sync_2026-08-24/train/profile_nuplan_dt03.py \
  --source-run experiments/baseline_run_sync_2026-08-24/baseline_run_sync_2026-08-24_2026-08-26_16-23-18_seed0 \
  --output-dir /tmp/nuplan_dt03_sizing
```

Choose a new output directory each time. The profiler uses the real checkpoint,
environment, collector, and PPO path. Each candidate runs in a fresh two-rank
process group for one warm-up and two measured rollout/update cycles. It writes
rank measurements, logs, candidate results, and a session summary; it disables
W&B, evaluations, trainer-state discovery, and checkpoint writes. It never starts
the full fine-tune. Ten seconds of the 45-minute cap are reserved for teardown
and report writing. Incomplete candidates cannot qualify.

Concurrency is tested in order 16, 32, 64, keeping 20 workers per rank. Larger
candidates are skipped when conservative scaling of measured GPU reserved
memory or process-tree PSS predicts insufficient headroom. Qualification requires
at least 15% GPU headroom and 20% host memory available, no swap growth, finite
losses, positive updates on both ranks, and synchronized retained counts and
update counts. Host monitoring samples every second; reports include per-rank
process-tree RSS/PSS and the combined process-tree peaks. GPU reports include
peak allocated/reserved bytes, device free bytes, collection/update time, and
global end-to-end transitions/second across the two measured cycles.

The fastest qualifying concurrency is selected, preferring the smaller one
within 5% of the fastest. A 64,000 microbatch with two-way accumulation is then
tested and adopted only for at least 5% improvement within the same limits.
A 16,000 microbatch with eight-way accumulation is tested only after an actual
32,000-microbatch CUDA OOM at the smallest concurrency. Selection writes
`selected_resources.yaml`; the launcher reads it only when explicitly supplied.
There is no runtime resizing.

## Launch, resume, evaluate

1. Preview the selected configuration (normally under 10 seconds):

   ```bash
   RESOURCE_CONFIG=/tmp/nuplan_dt03_sizing/selected_resources.yaml DRY_RUN=1 \
     project/baseline_run_sync_2026-08-24/train/launch_nuplan_sdc_finetune_2gpu.sh 0
   ```

2. Launch by removing `DRY_RUN=1`. Omit `RESOURCE_CONFIG` only to use the unmeasured
   starting YAML. The launcher stages only `final_model.pt` and `config.yaml`;
   source CARLA trainer state is never staged.

3. Resume by setting `RUN_NAME` to the existing dt03 run name and using the same
   resource file. Its own `trainer_state.pt` is discovered. The guard rejects
   mismatched timing, resources, optimization configuration, recurrence, or run
   directory before launch. A source checkpoint mismatch is also rejected.

4. Evaluate a completed run with
   `CUDA_VISIBLE_DEVICES=0 project/baseline_run_sync_2026-08-24/eval/evaluate_nuplan_sdc_finetune.sh 0`.
   Replay selected failures with
   `CUDA_VISIBLE_DEVICES=0 project/baseline_run_sync_2026-08-24/render/render_nuplan_sdc_finetune_failures.sh 0 nuplan_single`.
   Both support `DRY_RUN=1` and explicitly use the 0.1-second benchmark configuration.

## Verification evidence — September 10, 2026

- Before changing C, fixed-seed/fixed-action CPU outputs were captured externally
  for CARLA at 0.3 seconds and replay at 0.1 seconds. After rebuilding, observations,
  rewards, terminal flags, truncations, and resets matched **exactly**. Omitted and
  false flags also matched exactly. Existing goldens were not regenerated.
- Focused C tests verify every retained temporal field, lights, validity,
  unchanged velocity units, 67-state boundaries, shared geometry with private
  trajectories, and rejected partial loads. AddressSanitizer, UndefinedBehaviorSanitizer,
  and leak detection passed. The existing C unit and smoke suites passed.
- Python tests verify malformed inputs, repeated rejection, cleanup when a later
  environment fails, map-switch failure, repeated map switches, export dimensions,
  Hydra/DDP batch arithmetic, resume guards, all three evaluation merge routes,
  selection rules, and train/eval/render launcher dry runs.
  The combined focused run passed **52 tests**, including the existing configuration
  and randomized-initialization checks, in 35.21 seconds.
- A real seed-0 CARLA checkpoint completed a bounded **CPU** integration test:
  264 collected transitions across multiple episodes, a successful PPO update
  with finite losses and a fresh optimizer, and trainer-state save/restore. This
  used two serial environments, horizon/microbatch 132, one PPO epoch, FP32, and
  no compilation. It is not GPU throughput or DDP runtime evidence.
- `python setup.py build_ext --inplace --force` succeeded. There were no C source
  compiler warnings; GCC emitted its serial-LTO scheduling warning. Two-GPU
  memory sizing, compiled BF16 execution, throughput selection, and DDP runtime
  synchronization remain **unverified because CUDA is unavailable**.

Re-run focused checks:

```bash
source .venv/bin/activate
python -m pytest tests/unit_tests/test_replay_resampling.py project/baseline_run_sync_2026-08-24/train/test_dt03_workflow.py -q
make -C tests/drive unit smoke
```

The pre-change comparison requires the existing external capture via
`REPLAY_BASELINE=/tmp/pufferdrive_replay_before_dt03.npz`; it skips when absent.
The CPU checkpoint test requires `DT03_TEST_SOURCE_RUN` pointing to a source run.
