# PufferDrive Codebase Guide

Verified: **2026-09-08**, against local checkout **`58a7cd815baefcc844cfc31dcae391cf7b735eec`**.
This is a source audit, not a training or evaluation run. The August 24 sync finished at
`2026-08-24 00:52:03 -0400` according to the local Git reflog. The subsequent report commit
`58a7cd81` changed its date and added the policy-history subsection; it did not refresh the whole report.

Unless explicitly labeled historical or experiment-specific, values below describe this checkout's
`puffer_drive` YAML and its runtime code. Examples assume commands run from the repository root
with `source .venv/bin/activate` already executed. Placeholder checkpoint/data paths must be replaced.

### What changed in this audit

| Topic | Old report | Verified behavior |
|---|---|---|
| Actions | Discrete environment | Continuous environment interface with a separate, discrete 12-class policy by default; an action table connects them. |
| Evaluation | `EvalManager`, `validation_gigaflow`, separate mining command | Named benchmark catalog and `puffer eval`; training evaluation is disabled by default. |
| Experiments/checkpoints | Original eight-GPU job described as current; accumulating checkpoint files | Historical example only; explicit output root, periodic-weight pruning, final weights, and distinct resume behavior. |
| Observations/metrics | Nominal bounds and vague rate denominators | 923 training features, raw count/category fields, explicit agent/episode aggregation and distance ratio. |
| Commands/navigation | Missing paths and obsolete test/render flags | Current source links, positional benchmark CLI, HTML replay workflow, and current Makefile targets. |

Further reading: [timestep propagation](investigation_on_dt.md), [PPO and BPTT](bptt_training_step.md),
and [evaluation metrics](evaluation_metric.md). Those are separate reports with their own scope and verification dates.

This report is written for a new reader who knows only this much: PufferDrive is doing self-play reinforcement learning
for driving, and some datasets/maps are converted into binary files. It explains what the repo contains, how the pieces
fit together, what the default configuration runs, how to read training/eval output, and how to develop safely.

## Executive Summary

Source: [default YAML](../../pufferlib/config/puffer_drive.yaml).

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

The original report used an eight-GPU launch as a scale example. With today's default vectorization,
each rank still allocates `20 * 1024 = 20,480` agent slots and a `128`-step rollout horizon.
That gives `2,621,440` collected transitions per rank per PPO update, or `20,971,520` across eight ranks.
These are allocated/collected counts before masks and advantage filtering, not counts of successful or
optimizer-used transitions. The [historical experiment](#historical-experiment-august-2026) is not a live process report.

## Mental Model

Source: [rollout and optimizer loop](../../pufferlib/pufferl.py), [vectorization](../../pufferlib/vector.py).

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
| `pufferlib/pufferl.py` | Main CLI and PPO training/evaluation loop. |
| `pufferlib/vector.py` | Serial, multiprocessing, and Ray vectorization backends. |
| `pufferlib/config/puffer_drive.yaml` | The main experiment config. Most training behavior starts here. |
| `pufferlib/ocean/evaluation_utils/` | Benchmark configuration, reports, replay capture, and HTML rendering helpers. |
| `pufferlib/resources/drive/binaries/` | Bundled binary map/scenario resources. Default training uses `carla/`. |
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

The [upstream 3.0 branch](https://github.com/Emerge-Lab/PufferDrive/tree/3.0) is project context;
this guide describes the local commit identified above, not a claim about the latest remote branch.

[Robust Autonomy Emerges from Self-Play](https://proceedings.mlr.press/v267/cusumano-towner25a.html)
(ICML 2025) describes driving learned entirely through simulated self-play and the GigaFlow simulator,
including training on an eight-GPU node. Its results belong to that paper; they are not measurements of
this checkout or evidence that this repository reproduces every implementation detail.

Two additional background papers, verified from their primary abstract pages on 2026-09-08:

- [Beyond Self-Play and Scale: A Behavior Benchmark for Generalization in Autonomous Driving](https://arxiv.org/abs/2605.10034)
  introduces BehaviorBench, connects PufferDrive policies to nuPlan evaluation, and examines generalization to diverse traffic behavior.
- [Scaling Self-Play for End-to-End Driving](https://arxiv.org/abs/2606.19641)
  studies self-play for end-to-end driving. It is background for visual driving policies, not the source of the flat vector observation architecture below.

## Configuration System

Source: [YAML](../../pufferlib/config/puffer_drive.yaml), [load_config / train](../../pufferlib/pufferl.py), [schema](../../pufferlib/config_schema.py).

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
| `eval` | Benchmark selection, agent capacity, action selection, reports, and replay rendering. |
| `controlled_exp` / `sweep` | Hyperparameter sweep definitions. |

### Current Default Training Config

The default train mode is synthetic GigaFlow:

```yaml
env:
  simulation_mode: gigaflow
  num_agents: 1024
  min_agents_per_env: 1
  max_agents_per_env: 120
  action_type: continuous
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

For `num_envs=20`, `num_workers=auto` becomes 20. `vec.batch_size=auto` becomes
`num_envs // 2 = 10`, so an inference batch contains `10 * 1024 = 10,240` agent slots.
This vector batch size is measured in Python environments; `train.batch_size` below is measured in transitions.

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

Under TorchRun, `load_config()` integer-divides `train.total_timesteps` by `WORLD_SIZE`.
For eight ranks, the default target becomes `62,500,000,000` per rank, totaling `500,000,000,000`.
Rank 0 writes the saved configuration. `train.data_dir` is the run directory itself; the code does not
append `run_name`. Give each new experiment an explicit distinct directory.

### Which defaults actually apply?

| Layer | Example | Meaning |
|---|---|---|
| `Drive.__init__` defaults | `dt=0.1`, discrete actions, classic dynamics | Used when Python callers omit those arguments. They do not define the standard CLI experiment. |
| Main YAML | `env.dt=0.3`, continuous actions, jerk dynamics | Passed through the CLI into the environment constructor. |
| Policy YAML | `policy.action_type=discrete` | Independent of `env.action_type`; not automatically overwritten by it. |
| CLI / experiment launcher | Dotted overrides, dedicated output roots | Change the selected experiment; consult its saved config. |
| Checkpoint/evaluation merge | Saved architecture plus benchmark settings | Can supersede initial defaults; see the evaluation and checkpoint sections. |

The [post-sync baseline notes](../baseline_run_sync_2026-08-24/contexts_and_notes.md) describe an
experiment-specific workflow. Its settings are not the generic defaults of this guide.

## Data, Maps, And Scenario Modes

Source: [map loader](../../pufferlib/ocean/drive/map_data.h), [nuPlan pipeline](../../docs/nuplan_data.md).

There are two conceptual modes:

### `gigaflow`

This is procedural self-play. It uses map geometry, not logged trajectories, to spawn agents and create goals. The bundled CARLA maps are:

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

- In training, the binding partitions the agent budget into random internal C population sizes within the configured bounds, with a final remainder. GigaFlow evaluation instead allocates `max_agents_per_env` slots per scenario.
- Each agent is spawned on a drivable lane from the map grid with randomized dimensions.
- Goals are generated from map lanes (`goal_source=map`) or routes (`goal_source=route`).
- Traffic-light state schedules can be generated procedurally.
- The policy controls vehicles according to `control_mode` and controller settings.

Default training uses `goal_source=map`, so the agent is not following a logged human route. It receives goal
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

Illustrative replay training command (requires the dataset at the specified path):

```bash
puffer train puffer_drive \
  env.map_dir=data/nuplan_mini_train \
  env.num_maps=250 \
  env.simulation_mode=replay \
  env.control_mode=control_sdc_only \
  env.sdc_controller=policy env.non_sdc_controller=replay \
  env.dt=0.1 env.scenario_length=200 \
  run_name=replay_example train.data_dir=experiments/replay_example
```

This trains on logged scenarios while controlling the SDC. The example explicitly requests background replay
and `dt=0.1`; recorded trajectories advance by frame index, so changing the timestep is not automatic trajectory resampling.
`goal_source=gt` is also supported for targets drawn from logged trajectories, but only in replay mode.

## Python Environment Wrapper

Source: [Drive wrapper](../../pufferlib/ocean/drive/drive.py), [binding](../../pufferlib/ocean/drive/binding.c).

The Python env class is:

```text
pufferlib/ocean/drive/drive.py::Drive
```

Its main responsibilities:

1. Accept YAML/CLI values as keyword arguments and validate user-facing modes and controller settings.
2. Compute observation size from C constants and config; discover sorted `.bin` files or a single specified map.
3. Call `binding.shared()` to partition agent slots and `binding.env_init()` for each internal C environment.
4. Create the C vector bundle with `binding.vectorize()`; reset it through `binding.vec_reset()`.
5. Pass actions into C and return observations/rewards/dones/truncations/info, resampling scenarios when configured.

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

For the default YAML:

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

### Action interface versus policy distribution

The default combination is intentional:

```yaml
env:
  action_type: continuous
  dynamics_model: jerk
policy:
  action_type: discrete
```

The environment exposes `Box(-1, 1, shape=(2,), dtype=float32)`. The policy produces 12 logits,
one for each pair of `4` longitudinal and `3` lateral jerk choices. During training,
`sample_logits()` samples an integer class, computes its categorical log-probability, and maps it to
normalized longitudinal/lateral controls through `Drive.action_table`. PPO stores the integer class;
the simulator receives two floats. For example, physical jerk `(-4, 4)` maps to `(-4/15, 1)`.

With `env.action_type=discrete`, the jerk environment instead exposes `MultiDiscrete([12])` and
C decodes the class directly. With `policy.action_type=continuous` and a continuous environment,
the actor emits two Normal-distribution locations and two scale parameters; scale becomes
`softplus(scale) + 1e-4`. These are distinct supported paths, not interchangeable default descriptions.
The default evaluation setting `eval.action_selection=mean` specifically requires the discrete-policy /
continuous-environment combination; use `sample` or `mode` for other combinations.

## C Simulator Core

Source: [drive.h](../../pufferlib/ocean/drive/drive.h), [data structures](../../pufferlib/ocean/drive/datatypes.h).

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

1. Spawn agents at sampled map positions with randomized vehicle dimensions.
2. Reject candidate starts through collision/offroad/stop-line checks.
3. Generate goals from map or route.
4. Initialize agent state, erratic flags, and fixed or randomized reward coefficients.
5. Compute initial metrics and observations.

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

One ordinary step follows this order:

1. Clear reward/done/truncation buffers, set masks from pre-movement agent status, and increment the timestep.
2. Move background agents and active agents using their assigned controller; update seconds-stopped counters.
3. Compute metrics and rewards for active agents that are not already stopped/removed; set terminal flags.
4. If the scenario length or early-reset condition is reached, log the episode. Training resets in C; evaluation freezes the completed slot and returns.
5. Otherwise write observations, then update/regenerate goal targets for agents that reached a waypoint.

Thus ordinary observations describe the post-action state, but goal-target updates happen after observation writing.
A training truncation returns reset observations; the completed evaluation slot is held until Python advances its scenario window.
`terminate_on_goal=1` adds a final-goal early boundary only for replay with `control_sdc_only`.

### Dynamics

The current default uses jerk dynamics:

```text
env.dynamics_model = jerk
env.action_type = continuous
policy.action_type = discrete
```

Discrete physical jerk values (m/s³), also used to build the default policy action table, in `constants.h`:

```c
JERK_LONG[4] = {-15.0f, -4.0f, 0.0f, 4.0f}
JERK_LAT[3]  = {-4.0f, 0.0f, 4.0f}
MAX_SPEED    = 40.0f  // m/s
```

The jerk model integrates longitudinal and lateral acceleration, converts lateral acceleration into curvature/steering,
applies steering-rate and steering-position limits, then updates pose with bicycle-style motion. The jerk command changes acceleration over time; it does not directly set a new position.

### Controllers

Per-agent controller modes:

| Controller | Meaning |
|---|---|
| `policy` | Neural policy action controls the agent. |
| `static` | Agent does not move. |
| `replay` | Agent follows logged trajectory. Mainly replay mode. |
| `idm` | Rule-based Intelligent Driver Model behavior. |

Default YAML controller settings:

```yaml
sdc_controller: policy
non_sdc_controller: policy
non_vehicle_controller: auto
control_mode: control_vehicles
```

Default GigaFlow spawns vehicles and assigns policy controllers. In replay, `control_mode` selects
active slots, while `resolve_agent_controller()` determines who moves each agent. An active slot is not
proof of policy ownership: `c_step()` checks its controller. Replay-by-default background assignment
can take priority over a requested controller; inactive agents requesting policy fall back to static.

`non_vehicle_controller=auto` follows the non-SDC controller, except that non-SDC IDM implies replay
for non-vehicles. A `replay` controller only advances logged motion when `simulation_mode=replay`;
using it in GigaFlow does not create a trajectory to follow. Inspect resolved controller settings in
checkpoint-derived evaluation configurations.

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

Source: [C writers](../../pufferlib/ocean/drive/drive.h), [feature constants](../../pufferlib/ocean/drive/constants.h), [Torch slicing](../../pufferlib/ocean/torch.py).

Observations are flat `float32` vectors. The Gym space advertises `[-1, 1]`, but this is not a strict
bound on every field: valid-slot counts and traffic-control categories are raw numbers, and not all
geometric ratios are clipped. Do not infer a universal clipping operation from the space declaration. They are written by:

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

| Index (zero-based) | Feature |
|---|---|
| 0 | signed speed / max speed |
| 1 | width / normalization |
| 2 | length / normalization |
| 3 | steering angle / steering limit |
| 4 | longitudinal acceleration / absolute lower acceleration limit |
| 5 | lateral acceleration / upper lateral limit |
| 6 | clipped lane-center distance / lane-distance normalization |
| 7 | cosine of lane heading error |
| 8 | current lane speed limit / max speed (unknown lane uses -1 before division) |
| 9 | seconds stopped / max stopped seconds, capped at 1 |

Evaluation adds twice `eval_perceived_size_margin_m` to ego width and length before normalization.

### Reward/Goal Context

If `reward_conditioning=true`, the observation includes 17 reward-conditioning coefficients. The default YAML enables
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

Default YAML road dropout:

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
masked-pooling path, but it is disabled in the default YAML.

## Reward Function

Source: [compute_rewards / generate_reward_coefs](../../pufferlib/ocean/drive/drive.h).

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
| velocity | `coefficient * dt * max(cos(lane heading error), 0)` when signed speed exceeds 2.5 m/s and a lane is assigned; not distance travelled. |
| timestep | Tiny penalty while moving/accelerating. |
| reverse | Penalty for negative signed speed. |
| overspeed | Penalty for exceeding speed limit. |
| ADE | Adds `reward_ade * current_ade` when error is positive; the sign comes from the coefficient. Disabled (`0.0`) by default. |

Default YAML enables:

```yaml
reward_conditioning: true
reward_randomization: true
```

This is important. `generate_reward_coefs()` samples selected coefficients from C's `REWARD_BOUNDS` (or
`REWARD_BOUNDS_LOG` when requested), not from a range implied by each scalar YAML reward value.
Goal radius/speed and dynamics conditioning are included among the 17 coefficients; not all 17 are additive rewards.
Velocity and timestep coefficients are fixed in the randomized branch. With randomization off, configured
scalar values supply the coefficients and dynamics scales are 1. Those coefficients are also
included in the observation, so the same neural network can learn a family of behaviors/styles. This matches the
GigaFlow idea of one parameterized policy producing varied behavior.

Erratic-agent features:

```yaml
partner_blindness_prob: 0.02
partner_blindness_trigger_prob: 0.03
phantom_braking_prob: 0.02
phantom_braking_trigger_prob: 0.03
partner_blindness_duration_seconds: 3.0
phantom_braking_duration_seconds: 3.0
```

Python converts each duration to a step count using `duration_seconds // dt`, then the binding stores
an integer counter. The YAML unit is seconds, not a literal number of steps.

Flagged erratic agents are masked out of PPO training via `env->masks[i] = 0`, but they still exist in the simulation
and affect other agents. This is a robustness trick: the policy sees unusual agents without learning directly from their
corrupted behavior.

## Metrics And Logs

Source: [C accumulation](../../pufferlib/ocean/drive/drive.h), [Python export](../../pufferlib/ocean/drive/binding.c), [metric reduction](../../pufferlib/utils.py).

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

The default uses non-RNN transition PPO, so `_train_ppo_transition()` is active. It filters transitions by absolute
advantage before PPO updates, then synchronizes kept counts across DDP ranks.

### Environment Metrics

Environment logs use `environment/<name>` in training trackers. Logging backends are disabled by the
default YAML; experiment launchers can enable them. The following definitions come from
`compute_rewards()`, `add_log()`, `vec_log()`, and `my_log()`, rather than metric names alone.

| Metric | Per-agent / episode meaning |
|---|---|
| `episode_return` | Sum of that agent's rewards over the episode. |
| `episode_length` | Active-slot step count, including steps after an agent stops until the C episode ends. |
| `collision_rate`, `offroad_rate`, `red_light_violation_rate` | Each agent's binary ever-happened flag; averaging gives the fraction of logged active agents affected. |
| `num_goals_reached` | Number of waypoints reached, averaged over logged active agents. |
| `score` | 1 if goals reached are at least `env.num_goals` and the agent is neither stopped nor removed at episode end; otherwise 0. |
| `dnf_rate` | 1 if no collision/offroad/red-light event occurred and fewer than one goal was reached. This is not simply `1 - score`. |
| `lane_center_rate` | Count of evaluated reward steps with absolute lane-center distance below 0.5 m, divided by `max(env.timestep, 1)`. |
| `velocity_progress_sum` | Despite its name, divided by `max(env.timestep, 1)` in `add_log()`; the underlying signal is the gated heading cosine described above. |
| `avg_speed_per_agent` | Accumulated speed samples divided by `max(env.timestep, 1)`, in m/s. |
| `avg_distance_per_infraction` | Total accumulated distance in meters divided by `max(number of agent-episodes with any of the three infraction flags, 1)`. Multiple flags on one agent count once. |
| `reward_components/<x>` | Per-agent episode sum of the corresponding signed reward contribution, then averaged. |

**Aggregation has multiple levels.** Training C logs sum completed agent-episodes, increment `Log.n`
once per active agent, and `vec_log()` divides by that count. Python then averages received numeric
log values; it does not reweight ordinary metrics by each log's `n`. Distance and infraction totals
are carried separately and their ratio is recomputed by `reduce_environment_metrics()`.
In DDP, rank 0 logs its local environment metrics; the cross-rank environment reduction is commented
out in `mean_and_log()`. Steps and SPS are summed across ranks. Do not describe rank-0 environment
metrics as globally agent-weighted rates.

Evaluation produces one row per completed C scenario, averaging its active agents first. JSON
`metrics_mean` then takes the arithmetic mean of numeric scenario rows, except distance-per-infraction,
which is recomputed from summed distance/infraction totals. With different active-agent counts,
mean-of-scenarios and mean-of-agents differ. `num_scenarios` is requested coverage; `num_episodes`
is the actual number of rows.

The current benchmark catalog explicitly enables both `eval_mode` and `compute_eval_metrics`.
They are different switches: the former controls episode handling, the latter enables heavier metrics
and their export. Training defaults to `compute_eval_metrics=false`. C may still update internal
puffer display fields, but that does not make them fully evaluated/exported benchmark scores.

### Puffer Score

For meaningful exported puffer-score metrics use `compute_eval_metrics=true`. In C:

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

The weighted sum is divided by `19`. Progress ratio is travelled distance divided by
`max(10 m/s * episode_length * dt, 1 m)`; its weighted contribution is capped at 1, and the progress gate
requires a ratio strictly above 0.2. Direction compliance is 1 for at most 2 m wrong-way travel,
0.5 for more than 2 m through 6 m, and 0 above 6 m. TTC's contribution is
`1 - violations / samples` (a violation is TTC below the configured C threshold).

One serious gate failure can zero the puffer score even if smoothness/progress submetrics look decent. When
debugging an eval configured with `compute_eval_metrics=true`, inspect both `puffer_score` and its components.

### Checkpoints, Output Roots, And Resume

`train.data_dir` is the complete output directory. `run_name` identifies the run/logger but is not
appended to that path. With the bare YAML, outputs therefore go directly under `experiments/`.
Use an explicit per-run root in real launch commands.

| File under `train.data_dir` | Contents and lifetime |
|---|---|
| `config.yaml` | Saved configuration, including Git commit metadata; rank 0 writes it. |
| `models/model_puffer_drive_<epoch>.pt` | Policy state dict. The default save interval is 50 updates; each save removes other `model_*.pt` files in that directory. |
| `trainer_state.pt` | Policy, optimizer, scheduler, counters, RNG state, best score, and advantage-filter EMA. Replaced through a temporary file. |
| `best_models/best_trainer_state_<epoch>.pt` | Despite the name, a copy of policy weights, not a full trainer state. Older best files are pruned. |
| `final_model.pt` | Final policy weights written by `close()`; filename configurable through `train.final_model_name`. |

Best-model selection uses `last_stats.puffer_score` if present, otherwise training `score`.
It is not automatically selected from standalone benchmark JSON.

For full resume, use `train.resume_state_path`. Relaunching without `load_model_path` also discovers
`trainer_state.pt` in the output directory automatically. **Training with `load_model_path` is not
guaranteed to be weights-only:** `train()` looks beside the source checkpoint for a trainer state and,
if found, restores optimizer and counters as well. For a fresh fine-tune, the experiment workflows
stage source weights plus `config.yaml` without the source `trainer_state.pt`.

Standalone evaluation loads weights and requires the matching run `config.yaml`. Fine-tune training
merges policy/RNN settings and selected environment fields from that config; standalone evaluation
has a broader checkpoint-environment merge before applying the benchmark.

## Training Algorithm

Source: [PuffeRL](../../pufferlib/pufferl.py).

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
“run validation.” Scheduled benchmark validation is handled by `run_training_evaluation()` and `eval()`.

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

For the default per-rank configuration:

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

For non-RNN transition PPO, `_compute_advantages()` receives a ratio tensor of ones and clipping
limits `1.0, 1.0`, once before all optimization passes. Stored masks become terminal boundaries for
this computation, and masked advantages are zeroed. At a truncation, the collector's optional bootstrap
uses the previous stored value as a heuristic because C has already reset; it is not an exact value
of the unavailable pre-reset final observation.

See [PPO and BPTT](bptt_training_step.md) for the kernel indexing and return equations.

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

Default batching and the historical override differ:

| Setting | Default YAML | Historical eight-GPU example |
|---|---:|---:|
| Logical minibatch | 65,536 | 65,536 |
| Maximum microbatch | 65,536 | 32,768 |
| Accumulated microbatches | 1 | 2 |

The default MLP path flattens the buffer, removes masked transitions, and keeps absolute advantages
at least `0.01 * ema_max`. The EMA uses `0.25 * current_max + 0.75 * previous_ema` after its initial
update. DDP trims retained counts to the minimum across ranks. Each of three update passes reshuffles
the retained transitions. A partial final microbatch is retained; pending gradients are flushed at the
end, so optimizer step counts must be computed from retained data and accumulation, not raw rollout size.

`_ppo_loss()` evaluates stored actions under the current policy; it does not step the simulator.
For a normalized advantage `A` and ratio `r = exp(new_logprob - stored_logprob)`, it minimizes
`mean(max(-A*r, -A*clip(r, 0.8, 1.2)))`. The default value loss is
`0.5 * mean((V - return)^2)`; total loss adds `0.5 * value_loss` and subtracts `0.01 * entropy`.
Microbatch loss is divided by the accumulation count before backpropagation. Gradient norm is clipped
at `0.5` before each optimizer step. The AdamW optimizer uses a cosine learning-rate scheduler when
`anneal_lr=true`.

## Neural Network Architecture

Source: [Drive / DriveBackbone](../../pufferlib/ocean/torch.py), [LSTMWrapper](../../pufferlib/models.py).

The policy class is:

```text
pufferlib/ocean/torch.py::Drive
```

The backbone class is:

```text
pufferlib/ocean/torch.py::DriveBackbone
```

### Does The Policy Receive One Timestep Or A History?

**With the default configuration, the policy receives one current observation snapshot per agent, not a historical
sequence.** The relevant switch is:

```yaml
rnn_name: null
```

At rollout time, `PuffeRL.evaluate()` passes the just-received observation batch directly to
`policy.forward_eval()`. For the feed-forward `Drive` policy, the effective input shape is:

```text
[agents_in_inference_batch, observation_dim]
```

For one agent at simulator timestep `t`, the action and value therefore depend on only that agent's current flat
observation `o_t`:

```text
current C state at t -> compute_observations() -> o_t -> MLP policy -> action a_t, value V(o_t)
```

There is no frame stacking, no input shaped like `[t-127, ..., t]`, and no recurrent hidden state on this path. The
snapshot is not devoid of temporal information, however: its current-state fields include speed, steering angle,
longitudinal/lateral acceleration, and seconds stopped. Those values are consequences or summaries of earlier simulator
steps, but the raw observations from those earlier steps are not supplied to the network.

The rollout buffer can make this easy to misread. It stores:

```text
observations[segments, bptt_horizon, observation_dim]
```

and the default `bptt_horizon` is 128. This preserves temporal order for reward/return and advantage computation. It does
**not** make the default MLP a sequence model. Because `rnn_name` is null, `PuffeRL.train()` selects
`_train_ppo_transition()`, which reshapes the buffer to:

```text
[segments * bptt_horizon, observation_dim]
```

and samples individual transitions for the policy update. Thus, `bptt_horizon=128` is a rollout/storage horizon in the
default run; no backpropagation through a 128-observation history occurs in the policy.

The codebase also defines an optional recurrent path. `rnn_name: Recurrent` selects `LSTMWrapper`; it requires `policy.shared_network: true` and compatible
encoder/LSTM/head dimensions. This is an explanation of the wrapper path, not a claim that changing
only `rnn_name` makes the default Drive configuration a validated recurrent training recipe. During inference, each call still receives the
current `o_t`, but the LSTM carries hidden and cell state from preceding calls. During training, the wrapper consumes
`[trajectory_batch, time, observation_dim]`, with the time dimension governed by `bptt_horizon`. On that path, history
influences the decision through the recurrent state rather than by concatenating all earlier raw observations into one
large observation vector.

Architecture:

1. Slice the flat observation into ego, partners, lanes, boundaries, traffic controls, and context.
2. Encode each group with a small MLP; max-pool the groups with multiple object slots.
3. Concatenate the six encoded vectors.
4. Pass the concatenation through a backbone MLP.
5. Produce action logits through the actor head and a scalar value through the critic head.

The six encoder widths sum to `128 + 256 + 256 + 256 + 128 + 128 = 1152`; the backbone projects this
concatenation through three hidden layers of width 1024. The context encoder receives the 17 conditioning
values plus 9 goal fields. Actor and critic each have a 256-wide hidden head before their output layer.

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

The default discrete jerk actor outputs 12 logits even though the environment accepts two continuous
controls. Training samples a class and maps it through the action table; evaluation can instead use
a probability-weighted physical jerk mean. The optional Normal actor has four outputs for two actions.
See [action interface versus policy distribution](#action-interface-versus-policy-distribution).

## Vectorization And Distributed Training

Source: [vector.py](../../pufferlib/vector.py), [DDP setup](../../pufferlib/pufferl.py).

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

Source: [eval entry point](../../pufferlib/pufferl.py), [benchmark helpers](../../pufferlib/ocean/evaluation_utils/evaluation_utils.py), [generic catalog](../../pufferlib/config/evaluation/benchmark.yaml).

The current entry points are `pufferl.py::eval()` and `run_training_evaluation()`, with helpers in
`pufferlib/ocean/evaluation_utils/`. Configuration lives in
`pufferlib/config/evaluation/benchmark.yaml`. The former `EvalManager`, `validation_gigaflow`,
`--evaluator`, and `--render-backend` workflow is no longer the interface in this checkout.

### Benchmark configuration and precedence

Standalone evaluation starts with composed YAML/CLI arguments, loads the checkpoint's policy/RNN and
accepted environment fields, applies the catalog's shared environment and selected benchmark, then
reapplies explicit CLI overrides. `eval.num_agents` sets each worker's agent capacity; CLI
`env.num_agents` is rejected for this command. The resolved values are saved with the results.

The generic catalog does not explicitly reset the three controller fields. They can be inherited
from checkpoint configuration. The experiment-local
[baseline benchmark catalog](../baseline_run_sync_2026-08-24/override_config/evaluation_benchmarks.yaml)
sets explicit controllers; do not assume its fixes also exist in the generic catalog.
For CARLA policy self-play, explicitly request policy for both vehicle controller fields when the
source checkpoint's controller settings are uncertain.

| Setting | Default training | Generic `carla_fast` benchmark |
|---|---|---|
| Simulation / maps | GigaFlow / 8 CARLA maps | GigaFlow / 8 CARLA maps, 250 requested scenarios |
| `dt` / maximum length | 0.3 s / 2560 steps = 768 s | 0.1 s / 500 steps = 50 s |
| Internal population allocation | randomized, min 1, max 120 | fixed 50 slots per scenario in the GigaFlow evaluation binding; actual count can be reduced by spawn failures |
| Goals / reward randomization | map / true | route / false |
| `compute_eval_metrics` | false | true |
| Traffic-light behavior | stop | stop |
| Termination | early reset enabled | fixed scenario length |
| Lane/boundary dropout | 0.3 / 0.4 | 0 / 0 in standalone evaluation |
| Action selection | sampled class mapped to controls | physical probability-weighted mean by default |

With default slots, standalone evaluation's no-dropout observation width is
`10 + 17 + 9 + 144 + 630 + 450 + 28 + 4 = 1292`.
The object encoders share weights across rows and max-pool, allowing changed slot counts without
changing the backbone input width. This does not mean arbitrary observation-layout changes are compatible.

For `mean`, the discrete probabilities are averaged in physical action space before piecewise
normalization. Averaging normalized longitudinal values first would be different because negative
and positive jerk scales are 15 and 4. `mode` selects the most likely class; `sample` is stochastic.

### Commands and outputs

Replace `experiments/example_run/final_model.pt` with real weights and keep their matching `config.yaml`.

```bash
puffer eval puffer_drive carla_fast \
  load_model_path=experiments/example_run/final_model.pt \
  env.sdc_controller=policy env.non_sdc_controller=policy
```

Multiple benchmark names are a positional comma-separated string, for example `carla_fast,nuplan_single`.
Use the benchmark's recorded-frame timing for replay; don't copy the CARLA training timestep into it.

The default output is under the checkpoint run directory:

```text
eval/<benchmark>[_<output_name>]/<timestamp>/
  resolved_benchmark.yaml
  episode_metrics.csv
  evaluation_summary.json
```

The CSV contains one C scenario per row; JSON reports requested scenarios, emitted episodes, and
numeric means. Check both coverage fields before comparing results.

### Evaluation during training

`train.evaluation_interval_epochs=null` disables scheduled evaluation. To enable it:

```bash
puffer train puffer_drive \
  run_name=example_run train.data_dir=experiments/example_run \
  train.evaluation_interval_epochs=250 \
  'train.evaluation_benchmarks="carla_fast,nuplan_single"'
```

The inner quotes keep the comma value a Hydra string. Rank 0 runs the selected benchmarks with the
live policy at the interval, plus the final epoch if not already evaluated there. RNG state and policy
training mode are restored afterward. Periodic evaluation retains training road dropout, whereas
standalone evaluation normally uses the catalog's zero dropout. Their results need not match.

## Failure Selection And Replay Rendering

Source: [eval replay](../../pufferlib/ocean/evaluation_utils/eval_replay.py), [evaluation operations](../../docs/evaluation.md).

Failure selection now uses `puffer eval`. The old `mine_failures` CLI and `mine.*` configuration
are absent. It selects rows where chosen metric columns are positive, rather than applying an
`episode_return` threshold.

```bash
puffer eval puffer_drive carla_fast \
  load_model_path=experiments/example_run/final_model.pt \
  env.sdc_controller=policy env.non_sdc_controller=policy \
  eval.render_filter=all_infractions eval.max_rendered_failures=10 \
  eval.capture_observations=true
```

`all_infractions` uses collision, at-fault collision, offroad, and red-light flags. For an existing CSV,
add `eval.failure_replay_csv=<path/to/episode_metrics.csv>` with the same checkpoint and benchmark
settings; this skips the standard metrics pass and replays selected map/seed pairs.
`eval.failure_replay_csv` requires a non-null filter. `eval.max_rendered_failures` limits each benchmark.

The filtered workflow writes selected rows, replay metrics, `.replay.zlib` captures, and interactive HTML
under `failures/`, including `rendered_replays/index.html`. Observation panels require
`eval.capture_observations=true`. To capture all scenarios in the initial pass, use
`eval.render_scenarios=true`; that mode cannot be combined with an existing failure CSV.
These are HTML replay outputs. C video rendering is a separate path, not the removed backend flag.

## Historical Experiment (August 2026)

This example is retained from the original report, initially dated 2026-08-02. It is **not a current
process or checkpoint inventory**. Its referenced `project/first_run/status.md` and run directory are
not used as current evidence; the status file is absent from this checkout.

The old report recorded:

```text
experiment directory: experiments/puffer_drive_1785693123
checkpoint observed then: epoch 100
rank-local steps recorded then: 262,144,000
eight-rank sum: 2,097,152,000
```

Its launch was:

```bash
torchrun --standalone --nnodes=1 --nproc-per-node=8 -m pufferlib.pufferl train puffer_drive \
  train.minibatch_size=65536 train.max_minibatch_size=32768
```

The recorded configuration described CARLA self-play, map-sampled goals, randomized conditioning,
jerk dynamics, bf16 AMP, and disabled external loggers. Its then-planned validation interval and GPU
memory success are historical observations, not properties guaranteed by the same command today.
The step arithmetic remains consistent: `20 * 1024 * 128 * 100 = 262,144,000` per rank.
For current launches, supply an explicit run name and output root as shown below.

## How To Develop

Source: [build configuration](../../setup.py), [test targets](../../Makefile).

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

Illustrative bounded CPU training command (not executed in this audit):

```bash
puffer train puffer_drive train.device=cpu vec.backend=Serial env.num_agents=64 \
  vec.num_envs=1 vec.num_workers=1 vec.batch_size=1 \
  train.total_timesteps=2048 train.batch_size=auto train.minibatch_size=256 train.bptt_horizon=16 \
  train.amp=false train.precision=float32 \
  run_name=cpu_example train.data_dir=experiments/cpu_example
```

Default settings with an explicit output directory:

```bash
puffer train puffer_drive run_name=example_run train.data_dir=experiments/example_run
```

8-GPU train:

```bash
torchrun --standalone --nnodes=1 --nproc-per-node=8 -m pufferlib.pufferl train puffer_drive \
  train.minibatch_size=65536 train.max_minibatch_size=32768 \
  run_name=eight_gpu_example train.data_dir=experiments/eight_gpu_example
```

Rebuild C:

```bash
python setup.py build_ext --inplace --force
```

Run tests:

```bash
make test
make test-unit
make test-eval
make test-c
make test-docker-smoke
make test-notebooks
```

`make test` already includes unit, evaluation, C, and notebook suites; individual targets are alternatives
for narrower checks. The Docker smoke suite is separate and requires Docker. These are development
commands, not checks needed for editing this guide.

Debug C with sanitizers:

```bash
DEBUG=1 python setup.py build_ext --inplace --force
CUDA_VISIBLE_DEVICES=None LD_PRELOAD=$(gcc -print-file-name=libasan.so) \
  python -m pufferlib.pufferl train puffer_drive train.device=cpu vec.backend=Serial \
  vec.num_envs=1 vec.num_workers=1 vec.batch_size=1 env.num_agents=64 \
  train.total_timesteps=2048 train.batch_size=auto train.minibatch_size=256 train.bptt_horizon=16 \
  train.amp=false train.precision=float32 \
  run_name=debug_example train.data_dir=experiments/debug_example
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
| Change eval behavior | `pufferlib/pufferl.py::eval()`, `pufferlib/ocean/evaluation_utils/`, benchmark catalog and `eval.*` |
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

- Keep allocation out of stepping, observation, collision, and reward hot paths.
- Use bounded loops and named constants.
- Keep each function's mutations within one subsystem.
- Resolve compiler warnings.
- Validate external inputs before simulation and fail fast on invalid maps/configs.

Follow this. The simulator runs enormous step counts, so tiny hot-path overhead matters.

## Testing Strategy

Source: [Makefile](../../Makefile), [test suites](../../tests).

Test layers:

| Test area | Location | Purpose |
|---|---|---|
| C unit tests | `tests/drive/*.c` | Dynamics, geometry, goals, IDM, map cache, metrics, observations/rewards. |
| Python unit tests | `tests/unit_tests/` | Config, model slicing, action conversion, geometry utilities. |
| Evaluation integration | `tests/eval/` | Catalog, checkpoint/config, report, and replay behavior. |
| Docker smoke train/eval | `tests/smoke_tests/` | End-to-end deterministic CPU training/eval golden checks. |
| Notebook tests | `tests/notebooks/` | Ensure learning/inspection notebooks execute. |

Good default before committing simulator changes:

```bash
python setup.py build_ext --inplace --force
make test-unit
make test-eval
make test-c
```

For training-loop or end-to-end behavior:

```bash
make test-docker-smoke
```

For observation/reward understanding:

```bash
make test-notebooks
```

## Learning Path Through The Code

1. Read [the YAML](../../pufferlib/config/puffer_drive.yaml) and this guide's configuration/action tables. Separate environment defaults, policy defaults, and experiment overrides.
2. Read [the Python wrapper](../../pufferlib/ocean/drive/drive.py), then [constants](../../pufferlib/ocean/drive/constants.h) and [data types](../../pufferlib/ocean/drive/datatypes.h). Follow observation dimensions and the C-init arguments.
3. Read `c_step()`, observation writers, and `compute_rewards()` in [drive.h](../../pufferlib/ocean/drive/drive.h). Track one agent across one step before studying the full vector batch.
4. Read [the policy](../../pufferlib/ocean/torch.py) and [PuffeRL](../../pufferlib/pufferl.py). Follow sampled classes, converted controls, stored observations, advantages, and optimizer replay. Continue with [PPO/BPTT](bptt_training_step.md).
5. Read [evaluation operations](../../docs/evaluation.md) alongside the current benchmark source, then use [the notebooks](../../notebooks/) to inspect observations, rewards, and architecture. Consult [timestep propagation](investigation_on_dt.md) and [metric details](evaluation_metric.md) for focused follow-up.

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
4. Reduce observation slots or increase dropout (which reduces retained road rows); this changes the policy input content.
5. Reduce `vec.num_envs` or `env.num_agents`.

The old report recorded this A6000 microbatch recipe; memory fit has not been retested:

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
- `use_map_cache` and CPU load / worker oversubscription,
- `train.max_minibatch_size`.

### “Evaluation looks worse than training”

Compare resolved settings, not just the labels “training” and “evaluation”:

- Training uses map goals and randomized reward coefficients; the generic benchmark uses route goals and fixed coefficients.
- CARLA training defaults to `dt=0.3`; the generic catalog uses `dt=0.1`, with benchmark-specific scenario lengths.
- Controller ownership can be inherited from a checkpoint unless the catalog or CLI explicitly overrides it.
- Training samples actions; default evaluation uses a physical action mean. Periodic and standalone evaluation also differ in road dropout.
- Training `score` and evaluation `puffer_score` have different definitions and aggregation context.

Read `resolved_benchmark.yaml` beside the report before interpreting the difference.

### “Checkpoint does not load”

Check:

- For standalone eval, use `load_model_path=<path/to/model_*.pt>`.
- Standalone evaluation requires run `config.yaml`; checkpoint merge and subsequent benchmark/CLI overrides must remain compatible.
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

The source links at each major section identify the checked implementation. Additional audit anchors:

| Subject | Source |
|---|---|
| Shared C logging and Python reduction | [env_binding.h](../../pufferlib/ocean/env_binding.h), [utils.py](../../pufferlib/utils.py) |
| Action sampling and conversion | [pytorch.py](../../pufferlib/pytorch.py), [Drive policy](../../pufferlib/ocean/torch.py) |
| Benchmark configuration and aggregation | [evaluation_utils.py](../../pufferlib/ocean/evaluation_utils/evaluation_utils.py) |
| Replay capture / HTML | [eval_replay.py](../../pufferlib/ocean/evaluation_utils/eval_replay.py) |
| Dataset manifest and fetch helper | [datasets.yaml](../../data_utils/datasets.yaml), [fetch_data.py](../../data_utils/fetch_data.py) |
| Build and test commands | [setup.py](../../setup.py), [Makefile](../../Makefile) |

Verification was static: source/config comparison, arithmetic, CLI parsing/composition, and link checks.
No training, benchmark rollouts, rendering, simulator rebuilds, or expensive test suites were run.
The paper summaries above were checked against primary publication/abstract pages; they are background only.
