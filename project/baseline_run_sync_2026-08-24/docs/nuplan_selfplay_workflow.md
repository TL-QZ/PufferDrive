# Train nuPlan vehicle self-play from scratch

From the repository root, preview seed 0 (normally under 15 seconds):

```bash
DRY_RUN=1 project/baseline_run_sync_2026-08-24/train/launch_nuplan_selfplay_2gpu.sh 0
```

Start training by removing `DRY_RUN=1`. The launcher activates `.venv`, defaults
to GPUs **1 and 2**, and accepts seeds `0`, `1`, and `2`. Override the devices with
`CUDA_VISIBLE_DEVICES=4,5`. GPU availability can change; the profiler's measurements
describe its recorded session, not an allocation reservation.

## Experiment contract

The policy starts with random weights and a fresh optimizer. Training uses
**10 billion global agent transitions**, split across two DDP ranks, with normal
final-rollout overshoot. This counts transitions from all controlled vehicles,
not simulator steps or scenarios. There is no CARLA checkpoint dependency.

All **169,715** binaries in `data/nuplan_train` are eligible for the existing
random sampling with replacement. This does not guarantee a complete pass before
repeats. The launcher rejects a dataset-count mismatch instead of silently using
only part of a changed dataset.

The environment uses `simulation_mode=replay` to initialize recorded scenes and
advance logged background traffic. **Both SDC and eligible non-SDC vehicles use
the same learned policy.** Non-vehicles replay. Existing eligibility rules require
validity at initialization and a route; expert-marked vehicles retain replay.
The 300-entity scene cap and final-scene buffer packing remain unchanged simulator
behaviors. Recorded agents appearing later are not all spawned by vehicle mode.

Training uses `dt=0.3` and synchronized logged-frame resampling. A 201-state,
0.1-second recording retains states `0,3,...,198`: **66 transitions / 19.8 seconds**.
Scenes resample every 66 steps. Source binaries remain unchanged.

PPO uses horizon **128**, learning rate **0.0005** with annealing, three update
epochs, `gamma=0.999`, `gae_lambda=0.95`, BF16 AMP, compilation, and GPU rollout
buffers. The policy is feed-forward: horizon 128 does not add observation history.
The logical minibatch is 128,000 transitions per rank. Existing reward/goal
settings, observation layout, architecture, advantage filtering, and value
bootstrapping are preserved. The existing bootstrap heuristic is unchanged.

Periodic evaluation runs `carla_fast,nuplan_single,nuplan_multi`, with capacity
300 agents. The benchmark catalog explicitly restores **dt=0.1** and disables
replay resampling; nuPlan evaluation uses 200 transitions / 20 seconds. These are
the existing comparison benchmarks, not matched-dt=0.3 measurements.

## Evaluate a completed run

Run the final checkpoint on one GPU (replace `1` with your chosen GPU):

```bash
CUDA_VISIBLE_DEVICES=1 project/baseline_run_sync_2026-08-24/eval/evaluate_nuplan_selfplay.sh 0
```

The script finds exactly one self-play run for seed `0`, `1`, or `2`, requires
`final_model.pt` and `config.yaml`, and evaluates `carla,nuplan_single,nuplan_multi`.
It uses the existing benchmark catalog, 300-agent capacity, mean action
selection, `env.action_type=continuous`, `dt=0.1`, and
`resample_replay_to_dt=false`. The policy remains discrete: mean selection
averages its action-table controls. The explicit continuous-environment override
replaces the saved training environment's discrete action space after checkpoint
merging; otherwise `sample_logits` raises a `ValueError`. Rendering and observation
capture are disabled. W&B login verification occurs only when starting evaluation.
Add `DRY_RUN=1` to preview the command without evaluation or W&B access.

Results go under the selected run's
`eval/<benchmark>_final_model_mean_metrics/<timestamp>/`, including
`episode_metrics.csv` and `evaluation_summary.json`.

Initial September 13 validation passed shell syntax, both full-suite and nuPlan-only dry
runs, checkpoint/catalog configuration resolution, and six invalid-input or
missing-run checks. Those static checks missed the discrete-environment/mean-action
incompatibility. A subsequent actual launcher run reproduced the `ValueError`
in `pufferlib/pytorch.py::sample_logits` before the first CARLA scenario completed.

After adding `env.action_type=continuous`, a bounded GPU test with the real final
checkpoint completed two scenarios each for CARLA, nuPlan SDC-only, and nuPlan
multi-vehicle. CARLA used 16 steps for this smoke test; nuPlan retained 200 steps.
The test used one worker, disabled W&B reporting, and removed temporary results.
This verifies the corrected action path, not completion of the full benchmark
suite. A regression test exercises mean actions after checkpoint/config merging:

```bash
source .venv/bin/activate
python -m pytest -q project/baseline_run_sync_2026-08-24/eval/test_selfplay_evaluation.py
```

To evaluate only nuPlan:

```bash
CUDA_VISIBLE_DEVICES=1 BENCHMARKS=nuplan_single,nuplan_multi \
  project/baseline_run_sync_2026-08-24/eval/evaluate_nuplan_selfplay.sh 0
```

### Deferred issue: carla_fast training-time evaluation

**Open — reported September 13, 2026.** The user reported that `carla_fast`
evaluation during self-play training appeared to fail. The completed seed-0 run
is `baseline_run_sync_2026-08-24_nuplan_selfplay_dt03_2026-09-11_20-52-53_seed0`.
Diagnosis and repair are explicitly deferred. Later, inspect the training-time
error and resolved benchmark configuration, reproduce it, and fix its cause.
No evaluation-engine or benchmark changes are included in this launcher addition.
The final script selects full `carla`, not `carla_fast`; that does not establish
that the underlying issue cannot affect standalone CARLA evaluation.

## Render selected evaluation failures

Preview seed 0's nuPlan multi-vehicle failures, capped at 10 replays:

```bash
CUDA_VISIBLE_DEVICES=1 DRY_RUN=1 \
  project/baseline_run_sync_2026-08-24/render/render_nuplan_selfplay_failures.sh \
  0 nuplan_multi all_infractions 10
```

Remove `DRY_RUN=1` to render. The arguments are seed (`0|1|2`), benchmark
(`carla|nuplan_single|nuplan_multi`), optional failure mode, and optional render
limit. Modes are `all_infractions` (default), `collision`, `at_fault_collision`,
`offroad`, and `red_light`. The default limit is `null` (all matching failures);
set the fourth argument or `MAX_RENDERED_FAILURES` to a positive integer to cap it.

The launcher requires exactly one self-play run and uses its latest timestamped
standalone final-evaluation `episode_metrics.csv`, matching the existing render
launchers. It prints the chosen CSV and copies it into a new
`failure_analysis/<mode>/<timestamp>/` directory before replaying selected failures.
The original evaluation CSV is preserved. Open the printed HTML index at
`failure_analysis/<mode>/<timestamp>/failures/rendered_replays/index.html`.
Outputs are interactive HTML and replay data, not video files.

Rendering uses the evaluator's mean-action settings: the discrete policy sends
continuous controls, `dt=0.1`, and replay resampling is disabled. Observation
capture is enabled for the replay panels; W&B is disabled. A dry run creates no
analysis directory and does not construct an environment or use the GPU.

## Resources and profiling

The September 11, 2026 probe used two RTX A6000s (44.55 GiB visible memory each).
The selected defaults are **512 agents per environment, 20 workers/environments
per rank, 10 environments per inference batch, and a 32,000 microbatch**.
This gives **10,240 concurrent controlled-agent slots per GPU**, 5,120 observations
per inference call, and 1,310,720 rollout transitions per rank. The observation
width is 923 and the policy has 7,615,245 parameters.

| Agents/environment | Workers/GPU | Microbatch/GPU | Global transitions/s | Peak reserved GiB/GPU |
|---:|---:|---:|---:|---:|
| **512** | **20** | **32,000** | **233,094** | **16.61** |
| 1,024 | 20 | 32,000 | 243,167 | 21.50 |
| 2,048 | 20 | 32,000 | 239,464 | 30.38 |
| 512 | 20 | 64,000 | 240,729 | 28.55 |
| 1,024 | 10 | 32,000 | 190,649 | 16.65 |
| 256 | 40 | 32,000 | 244,090 | 16.65 |

All six candidates qualified. The selected setting uses the least peak reserved
GPU memory among candidates within 5% of the fastest throughput. Increasing the
buffer footprint did not produce a meaningful speedup. The 3,200-agent candidate
was skipped because measured memory growth predicted insufficient headroom.
Memory-growth prediction uses two measured sizes when available, plus 10% slack,
so fixed model memory is not simply multiplied with the rollout size.

The global rollout is **2,621,440 transitions**. Checkpoints every 10 epochs occur
every **26.2144M** transitions; evaluation every 39 epochs occurs every
**102.23616M**. The 10B budget completes after 3,815 rollouts, collecting
10,000,793,600 transitions including the final overshoot. PPO retains only a
subset of collected transitions after masking and advantage filtering; the
report records retained counts and actual optimizer updates on both ranks.

See the [retained sizing report](../profiling/2026-09-11_nuplan_selfplay/summary.json)
for per-rank timing, memory, losses, synchronization checks, and final confirmation.
The independent confirmation qualified at **219,583 global transitions/s** with
**16.61 GiB peak reserved memory per GPU** and **13.75 GiB combined process-tree
PSS**. Eight post-warm-up utilization samples averaged **56.8% / 69.1%** on GPUs
1 / 2. Collection and optimization both contribute to elapsed time; these
measurements do not imply continuously saturated GPUs. Short-run throughput
varied between confirmation and sizing, approximately **220k–233k transitions/s**
for the selected settings. The probes took **841 seconds** in total, including
the initial report-serialization failure; the session stayed below 45 minutes.

These bounded, seed-42 measurements establish short-run fit and throughput;
they do not establish long-run learning quality or sustained utilization over
the 10B run. Profiling excludes periodic evaluation and checkpoint overhead.

To repeat sizing without starting a production run:

```bash
source .venv/bin/activate
CUDA_VISIBLE_DEVICES=1,2 python project/baseline_run_sync_2026-08-24/train/profile_nuplan_selfplay.py \
  --output-dir /tmp/nuplan_selfplay_sizing_new
```

Use a new output directory. The cap is 45 minutes, with at most 8 minutes per
candidate. Each candidate runs one warm-up and two measured rollout/update cycles.
The probe disables W&B, evaluations, and checkpoints. It removes its temporary
run directories and compiler caches, and terminates owned process groups on
failure or interruption. The output contains only a JSON report and, if a
candidate qualifies, `selected_resources.yaml`. A rerun does not automatically
change launch defaults; transfer its selected resource keys into the experiment
YAML and revalidate before creating a new run.

Qualification requires positive synchronized optimizer updates, finite losses,
at least 15% GPU headroom, 20% host-memory headroom, and no swap growth. Candidates
that fail, time out, or lack three complete cycles cannot qualify. Selection
prefers lower peak reserved GPU memory within 5% of the fastest throughput.
The final confirmation reports GPU utilization sampled after warm-up separately
from utilization including startup. Throughput excludes the warm-up cycle.

## Output and resume

Outputs and the W&B group use `baseline_run_sync_2026-08-24_nuplan_selfplay_dt03`.
Run directories are isolated under:

```text
experiments/baseline_run_sync_2026-08-24/nuplan_selfplay_dt03/
  baseline_run_sync_2026-08-24_nuplan_selfplay_dt03_TIMESTAMP_seedN/
```

Resume by passing the exact existing run name and original seed:

```bash
RUN_NAME=baseline_run_sync_2026-08-24_nuplan_selfplay_dt03_TIMESTAMP_seed0 \
  project/baseline_run_sync_2026-08-24/train/launch_nuplan_selfplay_2gpu.sh 0
```

The launcher checks saved resolved settings, including policy, environment,
resources, seed, budget, evaluation, and output directory, before loading that
run's `trainer_state.pt`. A nonempty run directory without trainer state is
rejected; choose a new run name. A dry run does not create a run directory.

Run the focused CPU validation with:

```bash
source .venv/bin/activate
python -m pytest -q project/baseline_run_sync_2026-08-24/train/test_selfplay_workflow.py
```

Validation passed: 10 CPU tests, all three seed dry runs, seven invalid-launch
checks, and the CLI's own-state resume selection/nonempty-directory rejection.
Resume validation here covers configuration and state-path selection; a real
checkpoint save/reload training cycle was not run. No C source changed or rebuild
was needed. GPUs returned to 4 MiB used and 0% utilization after cleanup, with no
probe workers remaining. That September 11 preparation did not start production
training; the user subsequently completed training, and seed 0's final checkpoint
was found when adding the evaluation launcher on September 13.

Adjacent issue left untouched: the existing `train/profile_nuplan_dt03.py` writes
the same uncast NumPy transition counter that caused the initial JSON failure
in the copied probe. The new self-play profiler converts that counter to `int`.
