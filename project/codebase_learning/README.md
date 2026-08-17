# PufferDrive Codebase Learning Report

Last updated: 2026-08-02.

This report is written for a new reader who knows only this much: PufferDrive is doing self-play reinforcement learning
for driving, and some datasets/maps are converted into binary files. It explains what the repo contains, how the pieces
fit together, what your current experiment is running, how to read training/eval output, and how to develop safely.

## Executive Summary

PufferDrive is a multi-agent autonomous-driving RL stack. It has three main layers:

1. **C simulator:** `pufferlib/ocean/drive/drive.h`, `binding.c`, and related headers implement the fast driving
   environment. This is where reset, stepping, dynamics, collision checks, rewards, observations, metrics, maps, and
   rendering live.
2. **Python environment wrapper:** `pufferlib/ocean/drive/drive.py` exposes the C simulator as a PufferLib/Gym-like
   vectorized environment and translates YAML config into C init arguments.
3. **PyTorch training loop:** `pufferlib/pufferl.py` implements PPO-style RL, distributed training via `torchrun`,
   rollout buffers, advantage computation, dashboard logging, checkpointing, eval orchestration, and CLI entry points.

The default `puffer_drive` experiment is **not imitation learning** and is **not replaying human trajectories**. It is
synthetic self-play on CARLA OpenDRIVE-style binary maps:

```text
env.simulation_mode = gigaflow
env.map_dir         = pufferlib/resources/drive/binaries/carla
env.num_maps        = 8
env.goal_source     = map
```

Agents are procedurally spawned on drivable lanes, receive randomized goals and reward coefficients, then all policy-
controlled vehicles share one neural network. The policy learns from its own closed-loop consequences.

Your current run:

```bash
torchrun --standalone --nnodes=1 --nproc-per-node=8 -m pufferlib.pufferl train puffer_drive \
  train.minibatch_size=65536 train.max_minibatch_size=32768
```

It is an 8-GPU DDP run. Each rank creates `vec.num_envs=20` Drive env instances, and each Drive env instance has
`env.num_agents=1024`, so the full job has about `163,840` resident policy agents and collects `20,971,520` agent
transitions per PPO update across all ranks.

## Mental Model

Think of one training iteration as:

```text
YAML config
  -> pufferlib.pufferl.load_config()
  -> pufferlib.vector.Multiprocessing
  -> 20 Python Drive env instances per rank
  -> each Drive instance owns multiple C env slots
  -> C simulator writes flat observations/rewards/dones into shared arrays
  -> PyTorch policy samples actions and values
  -> C simulator applies actions, moves agents, computes rewards/metrics
  -> PPO trains on the collected rollout buffer
  -> checkpoint/eval/logging happen from rank 0
```

The hot loop is not Python object-heavy. C owns the per-agent arrays and writes directly into preallocated NumPy/shared
memory buffers. Python mostly orchestrates.

## Repository Map

High-level directories:

| Path | Purpose |
|---|---|
| `pufferlib/ocean/drive/` | Driving simulator implementation: C core, Python wrapper, render code, bindings. |
| `pufferlib/ocean/torch.py` | PufferDrive neural policy architecture. |
| `pufferlib/pufferl.py` | Main CLI and PPO training/eval/mining loop. |
| `pufferlib/vector.py` | Serial, multiprocessing, and Ray vectorization backends. |
| `pufferlib/config/puffer_drive.yaml` | The main experiment config. Most training behavior starts here. |
| `pufferlib/ocean/benchmark/` | Unified evaluation system and metrics/rendering workflows. |
| `pufferlib/resources/drive/binaries/` | Bundled binary map/scenario resources. Current training uses `carla/`. |
| `data_utils/` | Dataset manifest and fetch/conversion helpers. |
| `docs/` | Operational docs for cluster training, data, nuPlan, and evaluation. |
| `notebooks/` | Jupytext notebooks for observations, rewards, metrics, training, inference, architecture. |
| `tests/` | Unit, C, smoke, and notebook tests. |
| `experiments/` | Training output directories: frozen config, model checkpoints, trainer state. |
| `weights/` | Existing pretrained/checkpoint files outside current experiment runs. |
| `scripts/` | Cluster, eval, nightly, setup, rendering, and helper scripts. |

Core driving files:

| File | What to know |
|---|---|
| `drive.h` | The main simulator. Despite being a header, it contains most C implementation. |
| `binding.c` | Python C extension: creates envs, exposes constants/functions, converts logs/state to Python. |
| `drive.py` | Python `Drive` class: validates config, computes obs size, creates C envs, handles resampling/reset. |
| `constants.h` | Numeric constants and enum-like mode values shared by C/Python. |
| `datatypes.h` | C structs: `Agent`, `RoadMapElement`, `TrafficControlElement`, `LaneGraph`. |
| `map_data.h` | Binary map/scenario loading and map-cache helpers. |
| `idm.h` | Rule-based IDM controller for background/alternate traffic behavior. |
| `visualize.c`, `render.h`, `egl_headless.h` | Rendering paths for local/GPU/headless output. |
| `drivenet.h` | C inference helper for exported policy weights used by the visualizer. |

## External Context And Related Work

The upstream branch you linked is:

- <https://github.com/Emerge-Lab/PufferDrive/tree/3.0>

The GitHub README describes PufferDrive as a MARL autonomous-driving RL environment with a C GigaFlow/Waymo-replay
simulation engine and a Python/PyTorch PPO training loop.

The closest paper context is **Robust Autonomy Emerges from Self-Play**:

- arXiv: <https://arxiv.org/abs/2502.03349>
- PMLR proceedings page: <https://proceedings.mlr.press/v267/cusumano-towner25a.html>

That paper argues that robust driving can emerge from large-scale self-play without training on human demonstrations.
It introduces GigaFlow as a high-throughput batched simulator and reports training at very large scale on an 8-GPU
node. PufferDrive is not necessarily a byte-for-byte reproduction of the paper's internal code, but the design intent is
clearly aligned: high-throughput multi-agent self-play on vectorized driving maps.

Two related 2026 threads worth knowing:

- **Beyond Self-Play and Scale: A Behavior Benchmark for Generalization in Autonomous Driving**:
  <https://arxiv.org/abs/2605.10034>. This is relevant because it discusses PufferDrive/GigaFlow-style RL policies and
  the need to evaluate them against diverse behavior models, not only their self-play training distribution.
- **Scaling Self-Play for End-to-End Driving**: <https://arxiv.org/abs/2606.19641>. This extends the self-play idea toward
  pixel/end-to-end policies rather than vector observations.

## Configuration System

The main config file is:

```text
pufferlib/config/puffer_drive.yaml
```

It is loaded by:

```text
pufferlib/pufferl.py::load_config()
```

The CLI uses Hydra-style overrides:

```bash
puffer train puffer_drive train.learning_rate=0.001 env.num_agents=512
```

Old dashed flags like `--train.learning-rate=...` are rejected. Use dotted `key=value` overrides.

The environment section is schema-validated in:

```text
pufferlib/config_schema.py
```

That file defines enums such as `SimulationMode`, `ActionType`, `DynamicsModel`, `InfractionBehavior`, `ControlMode`,
`Controller`, `GoalSource`, and `GoalRegen`. This matters because typoed values should fail early, before C init.

### Important Config Sections

| Section | Meaning |
|---|---|
| top-level | CLI/logging/run settings: `env_name`, `policy_name`, `wandb`, `tb`, `neptune`, `load_model_path`. |
| `vec` | Python vectorization: backend, env count, worker count, batch size, seed. |
| `env` | Simulator settings: maps, agents, dynamics, reward, observations, resets. |
| `policy` | Neural network dimensions and activations. |
| `train` | PPO, optimizer, rollout, precision, checkpoint settings. |
| `eval` | Named evaluators and validation schedules. |
| `mine` | Failure-mining workflow config. |
| `controlled_exp` / `sweep` | Hyperparameter sweep definitions. |

### Current Default Training Config

The default train mode is synthetic GigaFlow:

```yaml
env:
  simulation_mode: gigaflow
  num_agents: 1024
  min_agents_per_env: 1
  max_agents_per_env: 120
  action_type: discrete
  dynamics_model: jerk
  dt: 0.3
  scenario_length: 2560
  termination_mode: 1
  map_dir: pufferlib/resources/drive/binaries/carla
  num_maps: 8
  goal_source: map
  reward_conditioning: true
  reward_randomization: true
```

The default vectorization is:

```yaml
vec:
  backend: Multiprocessing
  num_envs: 20
  num_workers: auto
  batch_size: auto
  zero_copy: true
  seed: 42
```

For `num_envs=20`, `num_workers=auto` becomes 20. `batch_size=auto` becomes `num_envs // 2 = 10`.

The default PPO config is:

```yaml
train:
  device: cuda
  optimizer: adamw
  amp: true
  precision: bfloat16
  total_timesteps: 500000000000
  learning_rate: 0.0005
  gamma: 0.999
  gae_lambda: 0.95
  update_epochs: 3
  clip_coef: 0.2
  vf_coef: 0.5
  ent_coef: 0.01
  batch_size: auto
  minibatch_size: 65536
  max_minibatch_size: 65536
  bptt_horizon: 128
  checkpoint_interval: 50
```

Under TorchRun, `load_config()` divides `train.total_timesteps` by `WORLD_SIZE` in each rank's saved config. For your
8-rank run, the frozen `config.yaml` shows `62,500,000,000` rank-local steps, but the intended job-level total remains
`500,000,000,000`.

## Data, Maps, And Scenario Modes

There are two conceptual modes:

### `gigaflow`

This is procedural self-play. It uses map geometry, not logged trajectories, to spawn agents and create goals. In your
run:

```text
pufferlib/resources/drive/binaries/carla/
  opendrive__Town01.bin
  opendrive__Town02.bin
  opendrive__Town03.bin
  opendrive__Town04.bin
  opendrive__Town05.bin
  opendrive__Town06.bin
  opendrive__Town07.bin
  opendrive__Town10HD.bin
```

In GigaFlow mode:

- C samples a random number of agents per internal C env between `min_agents_per_env` and `max_agents_per_env`.
- Each agent is spawned on a drivable lane from the map grid.
- Agent size is randomized.
- Goals are generated from map lanes (`goal_source=map`) or routes (`goal_source=route`).
- Traffic-light state schedules can be generated procedurally.
- The policy controls vehicles according to `control_mode` and controller settings.

Your current training run uses `goal_source=map`, so the agent is not following a logged human route. It receives goal
points sampled from the map/lane graph.

### `replay`

Replay mode consumes binary scenario files built from datasets such as nuPlan/WOMD. In replay mode, bins can include:

- logged agent trajectories,
- map geometry,
- traffic controls,
- scenario identity.

The docs describe the nuPlan pipeline as:

```text
raw nuPlan -> py123d -> arrow -> 123Drive -> PufferDrive .bin
```

Preconverted datasets are registered in:

```text
data_utils/datasets.yaml
```

Fetch default mini sets with:

```bash
python data_utils/fetch_data.py
```

Replay training example from the docs:

```bash
puffer train puffer_drive \
  env.map_dir=data/nuplan_mini_train \
  env.num_maps=250 \
  env.simulation_mode=replay \
  env.control_mode=control_sdc_only \
  env.scenario_length=200
```

That is materially different from your current experiment. It would train/evaluate on logged scenarios and often control
only the SDC/ego agent while background agents replay or use other controllers.

## Python Environment Wrapper

The Python env class is:

```text
pufferlib/ocean/drive/drive.py::Drive
```

Its main responsibilities:

1. Accept YAML/CLI config as Python keyword arguments.
2. Validate user-facing strings like `simulation_mode`, `control_mode`, `goal_source`.
3. Compute observation size from C constants and config.
4. Discover `.bin` map files under `map_dir`.
5. Call `binding.shared()` to split agents across internal C env slots.
6. Call `binding.env_init()` once per internal C env slot.
7. Call `binding.vectorize()` and `binding.vec_reset()` to create the C-side vectorized env bundle.
8. On every step, pass actions into C and return observations/rewards/dones/truncations/info.

Observation size is computed in `drive.py` from:

```text
ego_features
reward_conditioning coefficients
goal features
partner slots
lane slots
boundary slots
traffic control slots
valid-count fields
```

For your current config:

```text
ego:                    10
reward coefficients:    17
goals:                   3 goals * 3 = 9
partners:               16 * 9 = 144
lanes kept:             int(70 * (1 - 0.3)) = 49 slots * 9 = 441
boundaries kept:        int(50 * (1 - 0.4)) = 30 slots * 9 = 270
traffic controls:        4 * 7 = 28
valid counts:            4
total observation dim:  923
```

The action space for your current config is:

```text
action_type: discrete
dynamics_model: jerk
single_action_space: MultiDiscrete([12])
```

Those 12 actions are `4 longitudinal jerk choices * 3 lateral jerk choices`.

## C Simulator Core

Most simulator behavior is implemented in:

```text
pufferlib/ocean/drive/drive.h
```

Key structs:

| Struct | Meaning |
|---|---|
| `Agent` | One traffic participant: type, logged trajectory, sim pose, velocity, route, goals, reward coefficients, status. |
| `RoadMapElement` | Lane/road-line/edge geometry and lane graph connectivity. |
| `TrafficControlElement` | Traffic light/stop/yield data and controlled lanes. |
| `LaneGraph` | Lane-to-lane distance table used for routing/GPS-like goal distance. |
| `Drive` | Whole C env: agents, map, buffers, config, rewards, logs, render state. |
| `Log` | Aggregated training/eval metrics and reward-component sums. |

### Reset Flow

The reset logic is in:

```text
drive.h::c_reset()
drive.h::set_active_agents()
drive.h::spawn_agent()
```

In GigaFlow mode:

1. Agents are spawned at random valid map positions.
2. Vehicle dimensions are sampled.
3. Spawn collision/offroad/stop-line checks reject invalid starts.
4. Goals are generated from map or route.
5. Reward coefficients are generated, either fixed or randomized.
6. Initial metrics and observations are computed.

In replay mode:

1. Agents are loaded from logged trajectories.
2. `init_step` selects the logged timestep to initialize from.
3. Controlled/background agents are chosen according to `control_mode`.
4. Non-controlled agents can replay logged motion or use other controllers.

### Step Flow

The hot step function is:

```text
drive.h::c_step()
```

One step does:

1. Clear reward/done/truncation buffers.
2. Update masks for stopped/removed/erratic agents.
3. Increment timestep.
4. Move background replay/IDM agents.
5. Move active policy agents via dynamics.
6. Update stopped-duration counters.
7. Compute metrics and rewards for active agents.
8. Mark terminals for stopped/removed agents.
9. Reset env if scenario length or early-reset condition fires.
10. Compute next observations.
11. Update goals for agents that reached current targets.

This means observations returned after a normal step are post-action, post-reward, next-state observations.

### Dynamics

The current default uses jerk dynamics:

```text
env.dynamics_model = jerk
env.action_type = discrete
```

Relevant constants in `constants.h`:

```c
JERK_LONG[4] = {-15.0f, -4.0f, 0.0f, 4.0f}
JERK_LAT[3]  = {-4.0f, 0.0f, 4.0f}
MAX_SPEED    = 40.0f
```

The jerk model integrates longitudinal and lateral acceleration, converts lateral acceleration into curvature/steering,
applies steering-rate and steering-position limits, then updates pose with bicycle-style motion. This is more physically
smooth than directly choosing acceleration and steering.

### Controllers

Per-agent controller modes:

| Controller | Meaning |
|---|---|
| `policy` | Neural policy action controls the agent. |
| `static` | Agent does not move. |
| `replay` | Agent follows logged trajectory. Mainly replay mode. |
| `idm` | Rule-based Intelligent Driver Model behavior. |

Your current run has:

```yaml
sdc_controller: policy
non_sdc_controller: policy
non_vehicle_controller: auto
control_mode: control_vehicles
```

So all controllable vehicles are policy-controlled. Non-vehicle behavior follows `auto`, but GigaFlow spawning currently
creates vehicles.

### Infraction Behavior

For collision, offroad, and traffic lights:

```yaml
collision_behavior: stop
offroad_behavior: stop
traffic_light_behavior: stop
```

The C constants define:

```text
ignore = do not stop/remove for this infraction
stop   = freeze the agent
remove = remove the agent from simulation
```

With `termination_mode=1`, an env resets early when too many active agents become inactive:

```yaml
inactive_agent_threshold: 0.4
```

That means if more than 40% of active agents are stopped/removed, the C env truncates and resets.

## Observations

Observations are flat float vectors in `[-1, 1]` nominally. They are written by:

```text
drive.h::compute_observations()
drive.h::write_ego_obs()
drive.h::write_reward_target_obs()
drive.h::write_partner_obs()
drive.h::write_road_obs()
drive.h::write_traffic_control_obs()
```

The layout is:

```text
[ego]
[reward coefficients if reward_conditioning=true]
[goal points]
[partner agent rows]
[lane segment rows]
[boundary segment rows]
[traffic control rows]
[valid row counts]
```

### Ego Features

`EGO_FEATURES = 10`:

1. signed speed / max speed
2. width / normalization
3. length / normalization
4. steering angle / steering limit
5. longitudinal acceleration / limit
6. lateral acceleration / limit
7. lane-center distance / lane-distance normalization
8. lane heading alignment
9. current lane speed limit / max speed
10. seconds stopped / max stopped seconds

### Reward/Goal Context

If `reward_conditioning=true`, the observation includes 17 reward-conditioning coefficients. Your current run enables
this. Because `reward_randomization=true`, different agents can see different reward/style coefficients.

Then the observation includes `num_goals * 3` goal fields. Current config:

```text
num_goals = 3
goal features = ego-frame x, ego-frame y, relative z
```

### Partner Rows

Each partner row has `PARTNER_FEATURES = 9`:

```text
relative x, relative y, relative z,
length, width,
relative heading x, relative heading y,
signed speed,
seconds stopped
```

The env keeps the nearest partners within `obs_range_partner_m`.

### Road Rows

Lanes and boundaries both use `9` features per row:

```text
relative segment midpoint x/y/z,
segment half length,
lane width / normalization,
relative segment direction x/y,
goal lane distance absolute,
goal lane distance relative
```

For boundaries, the last two goal-distance fields are currently zero-filled.

Your current config uses road dropout:

```yaml
obs_slots_lane_n: 70
obs_dropout_lane: 0.3       # 49 kept
obs_slots_boundary_n: 50
obs_dropout_boundary: 0.4   # 30 kept
obs_lane_stride: 2
obs_boundary_stride: 1
```

Road dropout reduces model input size. Stride reduces road polyline density when loading/processing observation rows.

### Traffic Control Rows

Traffic-control rows have 7 raw features:

```text
stop-line endpoint 1 x/y,
stop-line endpoint 2 x/y,
relative z,
traffic control type,
traffic control state
```

The neural model one-hot encodes type and state before passing them through the traffic-control encoder.

### Valid Counts

The final 4 values are counts:

```text
lane_count, boundary_count, partner_count, traffic_control_count
```

These let the neural net know how many slots are real versus zero padding. In current config,
`policy.mask_padded_features=false`, so the model max-pools all rows, including zero-padded rows. Tests cover the
masked-pooling path, but it is not active in your current run.

## Reward Function

Rewards are computed in:

```text
drive.h::compute_rewards()
```

The current reward terms include:

| Component | Training effect |
|---|---|
| collision | Negative penalty if collision metric fires, speed-dependent extra penalty. |
| offroad | Negative penalty if agent goes offroad. |
| red light / stop line | Negative penalty on traffic-control violation. |
| goal | Positive reward for reaching goals, with speed check on final waypoint. |
| lane align | Rewards/penalizes heading and velocity relative to lane direction. |
| lane center | Penalizes distance from lane center, with a small center bonus shape. |
| comfort | Penalizes excessive acceleration/jerk/comfort violations. |
| velocity | Small progress-aligned velocity reward. |
| timestep | Tiny penalty while moving/accelerating. |
| reverse | Penalty for negative signed speed. |
| overspeed | Penalty for exceeding speed limit. |
| ADE | Optional replay imitation-ish displacement term; `0.0` in your run. |

Your current config enables:

```yaml
reward_conditioning: true
reward_randomization: true
```

This is important. Each agent gets reward coefficients sampled from configured ranges. Those coefficients are also
included in the observation, so the same neural network can learn a family of behaviors/styles. This matches the
GigaFlow idea of one parameterized policy producing varied behavior.

Erratic-agent features:

```yaml
partner_blindness_prob: 0.02
partner_blindness_trigger_prob: 0.03
phantom_braking_prob: 0.02
phantom_braking_trigger_prob: 0.03
phantom_braking_duration: 10
```

Flagged erratic agents are masked out of PPO training via `env->masks[i] = 0`, but they still exist in the simulation
and affect other agents. This is a robustness trick: the policy sees unusual agents without learning directly from their
corrupted behavior.

## Metrics And Logs

Training logs are assembled in:

```text
drive.h::add_log()
binding.c::my_log()
pufferl.py::mean_and_log()
pufferl.py::print_dashboard()
```

There are three classes of metrics:

1. **Training loop metrics:** SPS, steps, epoch, losses, performance timing.
2. **Environment metrics:** returns, collision/offroad/goal/infraction rates.
3. **Evaluation metrics:** puffer score components, coverage, episode CSVs, renders.

### Dashboard Summary

The console dashboard shows:

| Field | Meaning |
|---|---|
| `Env` | Environment name, here `puffer_drive`. |
| `Params` | Trainable parameter count. |
| `Steps` | Agent steps, summed across DDP ranks. |
| `SPS` | Agent steps per second, summed across DDP ranks. |
| `Epoch` | PPO update count. In DDP logs this display comes from rank 0, while logged `epoch` is summed in code. |
| `Uptime` | Wall-clock time since training process started. |
| `Remaining` | ETA from total timesteps and current SPS. |

### Performance Timing

Dashboard performance rows:

| Row | Meaning |
|---|---|
| `Evaluate` | Time spent collecting rollout data from env and policy. This is not validation eval; it is rollout collection. |
| `Forward` under Evaluate | Policy forward-pass time during rollout collection. |
| `Env` | Waiting for / stepping the simulator. |
| `Copy` | CPU/GPU tensor copy time for observations/rewards/dones. |
| `Train` | PPO optimization time after rollout collection. |
| `Learn` | Backprop/optimizer time. |
| `Train Forward` | Forward passes during PPO minibatch training. |

If `Env` dominates, simulator/vectorization is the bottleneck. If `Learn` dominates, the learner/model/microbatch is the
bottleneck. If `Copy` dominates, data movement or CPU offload may be the issue.

### Loss Metrics

Common PPO losses:

| Metric | Meaning | What to watch |
|---|---|---|
| `policy_loss` | PPO clipped actor objective. | Sign/magnitude alone is not a score; watch stability. |
| `value_loss` | Value-function regression loss. | Very high values can mean reward scale/critic instability. |
| `entropy` | Action-distribution entropy. | Lower means policy becomes more deterministic. Too low too early can mean collapse. |
| `old_approx_kl`, `approx_kl` | Policy update size diagnostics. | Spikes can signal overly large learning rate/minibatch issues. |
| `clipfrac` | Fraction of samples clipped by PPO ratio. | High values mean PPO clipping is active often. |
| `explained_variance` | Critic fit quality for returns. | Higher is better; near/under 0 means weak value predictions. |
| `masked_fraction` | Fraction of transitions masked out. | Erratic/stopped/removed agents increase this. |
| `kept_fraction` | Fraction kept after advantage filtering. | Very low means training only on high-advantage tail. |
| `filtered_fraction` | Complement of kept fraction. | Useful for detecting over-aggressive filtering. |
| `filter_threshold`, `ema_max` | Advantage filter threshold state. | Debug sampling/filter behavior. |

Your run uses non-RNN transition PPO, so `_train_ppo_transition()` is active. It filters transitions by absolute
advantage before PPO updates, then synchronizes kept counts across DDP ranks.

### Environment Metrics

Environment metrics are logged under `environment/<name>` in WandB/TensorBoard and shown as user stats on the console.
Your current run has `wandb=false`, `tb=false`, and `neptune=false`, so most output is console plus checkpoints unless
you redirected stdout.

Key env metrics:

| Metric | Meaning |
|---|---|
| `episode_return` | Sum of rewards per episode/agent, averaged through log aggregation. |
| `episode_length` | Episode length in simulator steps. Early resets lower it. |
| `collision_rate` | Fraction/count of agents with a collision in completed episodes. |
| `offroad_rate` | Fraction/count of agents that went offroad. |
| `red_light_violation_rate` | Traffic-control violation rate. |
| `num_goals_reached` | Average/aggregate goals reached by agents. |
| `score` | Training score: agent reached full goal set without being stopped/removed. |
| `dnf_rate` | Did-not-finish rate: no infractions but did not reach goal. |
| `lane_center_rate` | Fraction of timesteps near lane center. |
| `velocity_progress_sum` | Accumulated forward progress signal. |
| `avg_speed_per_agent` | Average speed over timesteps. |
| `avg_distance_per_infraction` | Total distance divided by infraction count. Higher is better. |
| `reward_components/<x>` | Cumulative contribution from each reward term. |

Important interpretation: in training, `compute_eval_metrics=false`, so the heavier puffer-score metrics are disabled
inside the training env.

Validation sets `eval_mode=1`, which changes evaluation behavior such as deterministic traffic-light handling and
scenario bookkeeping. That is not the same switch as `compute_eval_metrics`. The current `EvalManager` starts from the
training config and then applies evaluator env overrides. Since `validation_defaults.env` does not explicitly set
`compute_eval_metrics=true`, the training value can persist in inline validation unless you add that override. Treat
`puffer_score` fields as available only when the frozen eval config for the run has `compute_eval_metrics: true`.

### Puffer Score

The puffer score is computed when `compute_eval_metrics=true`. In C:

```text
puffer_score = multiplier * weighted_average
```

The multiplier is a product of gates:

```text
no_at_fault * no_offroad * no_red_light * making_progress * driving_direction_score
```

The weighted average combines:

```text
TTC score         weight 5
progress ratio    weight 5
speed compliance  weight 4
multi-lane score  weight 3
comfort score     weight 2
```

This means one serious gate failure can zero the puffer score even if smoothness/progress submetrics look decent. When
debugging an eval configured with `compute_eval_metrics=true`, inspect both `puffer_score` and its components.

### Checkpoint Metrics

Checkpointing happens every:

```yaml
train.checkpoint_interval: 50
```

Files:

```text
experiments/<run_id>/config.yaml
experiments/<run_id>/models/model_puffer_drive_000050.pt
experiments/<run_id>/models/model_puffer_drive_000100.pt
experiments/<run_id>/trainer_state.pt
experiments/<run_id>/best_models/best_trainer_state_000050.pt
```

`model_*.pt` stores policy weights. `trainer_state.pt` stores policy, optimizer, scheduler, step counters, RNG state,
best score, and selected model name. Use `trainer_state.pt` for true resume; use `model_*.pt` for loading weights.

## Training Algorithm

The main class is:

```text
pufferlib/pufferl.py::PuffeRL
```

The main loop is:

```text
train()
  -> pufferl.evaluate()  # rollout collection
  -> pufferl.train()     # PPO update
```

`evaluate()` is unfortunately named from a learning-loop perspective. In training it means “collect rollout data,” not
“run validation.” Validation is handled by `EvalManager`.

### Rollout Buffer

At startup, PuffeRL creates tensors:

```text
observations[segments, horizon, obs_dim]
actions[segments, horizon, action_dim]
values[segments, horizon]
logprobs[segments, horizon]
rewards[segments, horizon]
terminals[segments, horizon]
truncations[segments, horizon]
masks[segments, horizon]
```

For your current per-rank config:

```text
total_agents per rank = vecenv.num_agents = 20 * 1024 = 20,480
horizon = bptt_horizon = 128
batch_size auto = 20,480 * 128 = 2,621,440 transitions/update/rank
```

Across 8 ranks:

```text
20,971,520 transitions/update
```

### Advantage Computation

Advantages are computed by:

```text
pufferl.py::compute_puff_advantage()
pufferlib/extensions/cuda/pufferlib.cu
```

If CUDA extension support is available, it uses the compiled Torch extension. Otherwise it falls back to CPU. The config
has V-trace-style clipping fields:

```yaml
vtrace_rho_clip: 1
vtrace_c_clip: 1
```

For non-RNN transition PPO, the code currently calls `_compute_advantages(..., 1.0, 1.0)`.

### PPO Update

PPO loss is in:

```text
pufferl.py::_ppo_loss()
```

It computes:

- action log-prob ratio,
- clipped policy loss,
- value loss,
- entropy bonus,
- KL and clip diagnostics.

Your run uses:

```text
logical minibatch: 65,536
max microbatch: 32,768
gradient accumulation: 2
```

That means the learner forwards/backprops 32,768 transitions at a time, divides loss by 2, and steps the optimizer after
two microbatches. This is why it fits on 48 GB A6000 GPUs.

## Neural Network Architecture

The policy class is:

```text
pufferlib/ocean/torch.py::Drive
```

The backbone class is:

```text
pufferlib/ocean/torch.py::DriveBackbone
```

Architecture:

1. Slice flat observation into semantic groups.
2. Encode each group with a small MLP encoder:
   - ego,
   - partners,
   - lanes,
   - boundaries,
   - traffic controls,
   - context.
3. For slot groups, max-pool over slots.
4. Concatenate all encoded features.
5. Pass through a backbone MLP.
6. Actor head outputs action logits.
7. Critic head outputs scalar value.

Current config:

```yaml
policy:
  ego_input_size: 128
  partner_input_size: 256
  lane_input_size: 256
  boundary_input_size: 256
  traffic_control_input_size: 128
  context_input_size: 128
  backbone_hidden_size: 1024
  backbone_num_layers: 3
  actor_hidden_size: 256
  actor_num_layers: 1
  critic_hidden_size: 256
  critic_num_layers: 1
  shared_network: false
```

Because `shared_network=false`, actor and critic have separate backbones. This increases memory/compute but gives the
critic its own representation.

For discrete jerk actions, the actor outputs 12 logits. `pufferlib.pytorch.sample_logits()` samples actions during
rollout and computes log-probs for PPO.

## Vectorization And Distributed Training

There are two layers of parallelism:

1. **Within one process/rank:** `pufferlib.vector.Multiprocessing` creates multiple worker processes and shared memory
   arrays.
2. **Across GPUs:** `torchrun` creates one Python process per GPU and wraps the policy in `DistributedDataParallel`.

### Vector Backend

Default:

```yaml
vec.backend: Multiprocessing
vec.num_envs: 20
vec.num_workers: auto
vec.batch_size: auto
vec.zero_copy: true
```

`Multiprocessing` creates a driver env only to inspect shapes, then creates worker processes. Each worker owns its env,
but observations/actions/rewards are shared via `multiprocessing.RawArray`.

`zero_copy=true` means the learner reads batches directly from shared memory slices where possible.

### TorchRun

When `LOCAL_RANK` exists:

1. `load_config()` divides `total_timesteps` by `WORLD_SIZE`.
2. `train()` sets the CUDA device to local rank.
3. `torch.distributed.init_process_group(backend="nccl")` is called.
4. The policy is wrapped in `DistributedDataParallel`.
5. Only rank 0 owns external loggers and saves checkpoints/evals.

Gradients are synchronized during DDP backprop. Each rank collects its own rollout data.

## Evaluation System

Evaluation docs live at:

```text
docs/evaluation.md
```

Code lives at:

```text
pufferlib/ocean/benchmark/manager.py
pufferlib/ocean/benchmark/evaluators/
```

Concepts:

| Concept | Meaning |
|---|---|
| `Evaluator` | One named eval suite from `eval.<name>` config. |
| `EvalManager` | Discovers eval sections and runs enabled/due evaluators. |
| `multi_scenario` | Sweeps a scenario set in one batched rollout. |
| `wosac` | Waymo Open Sim Agents Challenge metrics path. |

Default enabled validation:

```yaml
eval.validation_gigaflow:
  enabled: 'true'
  interval: 250
  type: multi_scenario
  mode: inline
  render: 'false'
```

It inherits `validation_defaults`, then overrides env to:

```yaml
simulation_mode: gigaflow
map_dir: pufferlib/resources/drive/binaries/carla
num_maps: 8
num_agents: 1024
min_agents_per_env: 40
max_agents_per_env: 40
scenario_length: 500
resample_frequency: 500
goal_source: route
reward_randomization: false
termination_mode: 0
traffic_light_behavior: ignore
```

So training and validation are not identical:

| Aspect | Training | Validation GigaFlow |
|---|---|---|
| goal source | `map` | `route` |
| reward randomization | true | false |
| compute eval metrics | false | inherits false unless explicitly overridden |
| scenario length | 2560 | 500 |
| min/max agents per internal C env | 1/120 | 40/40 |
| traffic lights | stop | ignore |
| termination | early reset allowed | fixed length |

If you want validation to emit puffer-score fields, add this to the evaluator env overrides:

```bash
puffer train puffer_drive eval.validation_gigaflow.env.compute_eval_metrics=true
```

Run standalone eval:

```bash
puffer eval puffer_drive --evaluator validation_gigaflow \
  load_model_path=experiments/puffer_drive_1785693123/models/model_puffer_drive_000100.pt
```

Render observations:

```bash
puffer eval puffer_drive --eval_simulation gigaflow \
  load_model_path=experiments/puffer_drive_1785693123/models/model_puffer_drive_000100.pt \
  --num_scenarios 10 --render 1 --render-backend obs_html
```

Render backends:

| Backend | Output | Use |
|---|---|---|
| `egl` | MP4 videos | Shareable visual clips. |
| `triage_html` | HTML replay with metrics | Episode failure triage. |
| `obs_html` | HTML scene plus neural observation | Inspect what policy sees. |

## Failure Mining

Failure mining is a post-training workflow in:

```text
pufferlib/pufferl.py::mine_failures()
pufferlib/mining_viz.py
```

It rolls out a policy, records per-episode compact replays for episodes whose `episode_return` is below a threshold,
renders failures to HTML, and writes an index.

Example:

```bash
puffer mine_failures puffer_drive \
  load_model_path=experiments/puffer_drive_1785693123/models/model_puffer_drive_000100.pt \
  mine.output_dir=./failure_mining/puffer_drive_1785693123 \
  mine.num_episodes=200 \
  mine.score_threshold=-10.0
```

Output:

```text
episodes.csv
replays/episode_NNNNNN.replay.zlib
renders/episode_NNNNNN.html
renders/index.html
```

This is useful after a long run because aggregate metrics hide individual weird scenes.

## Your Current Experiment

Detailed run notes are in:

```text
project/first_run/status.md
```

Key facts:

```text
experiment dir: experiments/puffer_drive_1785693123
latest on-disk checkpoint checked: epoch 100
rank-local global_step: 262,144,000
8-rank summed agent steps: 2,097,152,000
```

Current run command from the process table:

```bash
torchrun --standalone --nnodes=1 --nproc-per-node=8 -m pufferlib.pufferl train puffer_drive \
  train.minibatch_size=65536 train.max_minibatch_size=32768
```

What it is training:

- one shared policy for all controlled vehicles,
- synthetic closed-loop GigaFlow traffic,
- 8 CARLA binary maps,
- map-sampled goals,
- randomized reward/style conditioning,
- jerk dynamics,
- stop-on-collision/offroad/traffic-control behavior,
- PPO with bf16 AMP.

What it is not doing:

- not training on nuPlan replay,
- not supervised imitation from human trajectories,
- not evaluating every checkpoint yet until epoch 250,
- not writing WandB/TensorBoard logs because those are disabled.

## How To Develop

### Environment Setup

From repo root:

```bash
source .venv/bin/activate
```

After editing any C files under `pufferlib/ocean/drive/`:

```bash
python setup.py build_ext --inplace --force
```

This is mandatory. Python imports the compiled extension:

```text
pufferlib/ocean/drive/binding.cpython-*.so
```

If you edit C but do not rebuild, your changes are not running.

### Common Commands

Small CPU smoke train:

```bash
puffer train puffer_drive train.device=cpu vec.backend=Serial env.num_agents=64 \
  train.total_timesteps=20480 train.batch_size=10240 train.minibatch_size=2560 train.bptt_horizon=16
```

Default train:

```bash
puffer train puffer_drive
```

8-GPU train:

```bash
torchrun --standalone --nnodes=1 --nproc-per-node=8 -m pufferlib.pufferl train puffer_drive \
  train.minibatch_size=65536 train.max_minibatch_size=32768
```

Rebuild C:

```bash
python setup.py build_ext --inplace --force
```

Run tests:

```bash
make test
make test-unit
make test-c
make test-smoke
make test-notebooks
```

Debug C with sanitizers:

```bash
DEBUG=1 python setup.py build_ext --inplace --force
CUDA_VISIBLE_DEVICES=None LD_PRELOAD=$(gcc -print-file-name=libasan.so) \
  python -m pufferlib.pufferl train puffer_drive train.device=cpu vec.backend=Serial
```

### Where To Change Things

| Goal | Edit |
|---|---|
| Change default experiment scale/rewards/maps | `pufferlib/config/puffer_drive.yaml` |
| Add/change env config schema | `pufferlib/config_schema.py` |
| Change simulation step behavior | `pufferlib/ocean/drive/drive.h::c_step()` and helpers |
| Change reward terms | `drive.h::compute_rewards()` |
| Change metrics/logs | `drive.h::add_log()`, `binding.c::my_log()` |
| Change observation layout | `drive.py` obs-size logic, `drive.h::compute_observations()`, `pufferlib/ocean/torch.py` slicing |
| Change dynamics | `drive.h::move_dynamics()` |
| Change spawn/goals | `drive.h::spawn_agent()`, route/goal helpers |
| Change neural architecture | `pufferlib/ocean/torch.py` |
| Change PPO/training loop | `pufferlib/pufferl.py` |
| Change eval behavior | `pufferlib/ocean/benchmark/evaluators/`, `manager.py`, config `eval.*` |
| Add dataset fetch entry | `data_utils/datasets.yaml` |
| Add cluster recipe | `scripts/cluster_configs/`, `scripts/submit_cluster.py` |

### Observation Changes Are Cross-Cutting

If you change observation layout, update all three layers together:

1. C writer in `drive.h`.
2. Python dimension calculation in `drive.py`.
3. Torch slicing/model input handling in `pufferlib/ocean/torch.py`.

Then run:

```bash
python setup.py build_ext --inplace --force
pytest tests/unit_tests/test_drive_backbone.py tests/unit_tests/test_drive_config.py
make test-c
```

Observation bugs often show up as shape mismatches, silent zero padding, or policies training on shifted features.

### Reward/Metric Changes Need Interpretation

If you change rewards:

- expect `episode_return` scale to change,
- critic `value_loss` scale may change,
- PPO stability may require learning-rate or advantage-filter changes,
- old checkpoints may not be directly comparable,
- validation metrics are more stable than raw training return if reward randomization is enabled.

If you change metrics:

- update `binding.c::my_log()` so Python can see the new metric,
- update eval aggregation if it should appear in eval reports,
- add tests around edge cases if it is used for checkpoint selection.

### C Hot Path Rules

The project style guide in `AGENTS.md` is strict for C:

- no malloc/free in `c_step`, observation generation, collision, reward hot paths,
- bounded loops,
- named constants for magic values,
- one function mutates one subsystem,
- zero compiler warnings,
- validate untrusted external inputs before init/reset/step,
- fail fast on invalid maps/configs.

Follow this. The simulator runs enormous step counts, so tiny hot-path overhead matters.

## Testing Strategy

Test layers:

| Test area | Location | Purpose |
|---|---|---|
| C unit tests | `tests/drive/*.c` | Dynamics, geometry, goals, IDM, map cache, metrics, observations/rewards. |
| Python unit tests | `tests/unit_tests/` | Config, model slicing, eval manager, geometry utilities. |
| Smoke train/eval | `tests/smoke_tests/` | End-to-end deterministic CPU training/eval golden checks. |
| Notebook tests | `tests/notebooks/` | Ensure learning/inspection notebooks execute. |

Good default before committing simulator changes:

```bash
python setup.py build_ext --inplace --force
make test-unit
make test-c
```

For training-loop or end-to-end behavior:

```bash
make test-smoke
```

For observation/reward understanding:

```bash
make test-notebooks
```

## Learning Path Through The Code

Recommended reading order:

1. `pufferlib/config/puffer_drive.yaml`
   - Understand the experiment before reading code.
2. `project/first_run/status.md`
   - Understand your actual run scale and maps.
3. `pufferlib/ocean/drive/drive.py`
   - See how config becomes env objects, obs size, map files, and C init.
4. `pufferlib/ocean/drive/constants.h`
   - Learn modes, feature counts, reward indices, metric indices.
5. `pufferlib/ocean/drive/datatypes.h`
   - Learn the simulator data model.
6. `pufferlib/ocean/drive/drive.h::c_step()`
   - Learn the step lifecycle.
7. `drive.h::compute_observations()` and writers
   - Learn what the policy sees.
8. `drive.h::compute_rewards()`
   - Learn what the policy optimizes.
9. `pufferlib/ocean/torch.py`
   - Learn how observation slots become neural features.
10. `pufferlib/pufferl.py::PuffeRL`
    - Learn rollout collection, PPO, logging, and checkpointing.
11. `docs/evaluation.md`
    - Learn how to run and interpret validation.
12. `notebooks/01_observations.py` through `06_architecture.py`
    - Use notebooks to inspect behavior interactively.

## Practical Debugging Recipes

### “My config override did nothing”

Check:

- Did you use Hydra `key=value` syntax?
- Is the key actually present in `puffer_drive.yaml`?
- Did the frozen `experiments/<run>/config.yaml` include your override?
- If changing C behavior, did you rebuild?

### “Training OOMs”

Try in this order:

1. Lower `train.max_minibatch_size`.
2. Keep `train.minibatch_size` fixed if you want same logical minibatch and use gradient accumulation.
3. Reduce policy size.
4. Reduce observation slots/dropout less aggressively.
5. Reduce `vec.num_envs` or `env.num_agents`.

Your current successful A6000 recipe uses:

```text
train.minibatch_size=65536
train.max_minibatch_size=32768
```

### “SPS is low”

Look at dashboard timing:

- High `Env`: simulator/vectorization bottleneck.
- High `Forward`: policy inference bottleneck.
- High `Learn`: backprop/model/minibatch bottleneck.
- High `Copy`: data movement bottleneck.

Possible levers:

- `vec.num_envs`, `vec.batch_size`, `vec.num_workers`,
- policy dimensions,
- observation slot counts,
- `use_map_cache`,
- `train.max_minibatch_size`,
- CPU load and worker oversubscription.

### “Evaluation looks worse than training”

Expected possibilities:

- validation uses `goal_source=route`, while training uses `goal_source=map`;
- validation disables reward randomization;
- validation uses fixed 40-agent env slots;
- validation ignores traffic lights;
- validation has shorter `scenario_length=500`;
- training `score` and eval `puffer_score` are not the same metric.

Always compare the exact config sections.

### “Checkpoint does not load”

Check:

- For standalone eval, use `load_model_path=<path/to/model_*.pt>`.
- The code tries to merge architecture-critical settings from sibling `config.yaml`.
- If you changed observation layout without preserving checkpoint config, old models may not match new obs shape.
- DDP checkpoints may have `module.` prefixes; `clean_policy_state_dict()` strips common prefixes.

## Glossary

| Term | Meaning |
|---|---|
| Agent step | One controlled agent taking one simulator step. Multi-agent envs generate many per env tick. |
| PPO update / epoch | One cycle of rollout collection plus PPO optimization. |
| GigaFlow | Procedural high-throughput multi-agent self-play mode. |
| Replay | Mode using logged trajectories/scenario bins from datasets. |
| SDC | Self-driving car / ego target agent in replay-style scenarios. |
| IDM | Intelligent Driver Model, a rule-based car-following controller. |
| `terminal` | Agent-level done, e.g. stopped/removed. |
| `truncation` | Env-level reset boundary, e.g. scenario length or early reset. |
| Mask | Whether a transition should train PPO. Erratic/stopped/removed agents are masked out. |
| Reward conditioning | Policy observes reward/style coefficients. |
| Reward randomization | Per-agent reward/style coefficients are randomized each episode. |
| `score` | Training success-style metric: goals reached without stopped/removed. |
| `puffer_score` | Eval-oriented composite score with safety/progress gates and weighted submetrics. |

## Source Trail

Local files read for this report:

```text
README.md
docs/evaluation.md
docs/data_storage.md
docs/nuplan_data.md
2502.03349v1.pdf
pufferlib/config/puffer_drive.yaml
pufferlib/config_schema.py
pufferlib/pufferl.py
pufferlib/vector.py
pufferlib/ocean/environment.py
pufferlib/ocean/torch.py
pufferlib/ocean/drive/drive.py
pufferlib/ocean/drive/drive.h
pufferlib/ocean/drive/binding.c
pufferlib/ocean/drive/constants.h
pufferlib/ocean/drive/datatypes.h
data_utils/datasets.yaml
data_utils/fetch_data.py
tests/
```

Online sources checked:

- PufferDrive 3.0 branch: <https://github.com/Emerge-Lab/PufferDrive/tree/3.0>
- Robust Autonomy Emerges from Self-Play: <https://arxiv.org/abs/2502.03349>
- PMLR proceedings page for Robust Autonomy: <https://proceedings.mlr.press/v267/cusumano-towner25a.html>
- BehaviorBench paper: <https://arxiv.org/abs/2605.10034>
- Scaling Self-Play for End-to-End Driving: <https://arxiv.org/abs/2606.19641>
