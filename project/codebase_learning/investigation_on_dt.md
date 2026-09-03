# How `dt` Flows Through PufferDrive Training and Evaluation

This document traces the current implementation at commit `f5ddc413`. It starts
from the two commands used for the third-run CARLA baseline:

```bash
# Training
./project/third_run_nightly_best_config/train/launch_2gpu.sh 0

# Standalone evaluation
CUDA_VISIBLE_DEVICES=0 \
  ./project/third_run_nightly_best_config/eval/evaluate_final_model.sh 0
```

The purpose is to explain what changing `env.dt` actually changes in the code.
It does not diagnose a particular score regression, rank possible causes, or
recommend an experiment. The important starting point is that `dt` is not only
an episode-duration setting. It is used by vehicle integration, control rate,
rewards, metrics, traffic lights, perturbations, PPO discounting choices, and
the interpretation of replay frames.

## Takeaway: a `dt=0.3` policy can operate at `dt=0.1`

Training and evaluation do not need to have equivalent transition dynamics for
the trained policy to function. The default policy acts as a state-feedback
controller: it maps the current observation to a jerk command. When deployed at
`dt=0.1`, the simulator applies that command for 0.1 seconds, produces a new
observation, and asks the policy for another command. Over the 0.3 seconds that
would have been one training step, the deployed policy may therefore make three
different decisions. That is valid behavior, not an error; the policy is being
given more frequent opportunities to react to the evolving state.

The observation is temporally informative even though it is not an observation
history. It includes state variables such as speed, longitudinal and lateral
acceleration, steering angle, and stopped duration, all of which carry the
effects of earlier actions. This can let the feed-forward policy respond
sensibly at the higher decision frequency without receiving an explicit stack
of past frames.

This argument establishes plausibility, not equivalence or a performance
guarantee. The policy was optimized using `dt=0.3` transitions, and neither the
current observation nor the network input explicitly identifies `dt`. At
`dt=0.1`, it will also encounter intermediate states and action consequences at
a frequency not used during training. Those facts may improve, preserve, or
degrade performance; the timestep mismatch alone does not determine the
outcome. The precise conclusion is therefore that a policy trained at
`dt=0.3` can be deployed as a more frequently queried feedback controller at
`dt=0.1`, while its measured behavior remains performance at the new control
frequency rather than an equivalent reproduction of the training process.

## Pipeline at a glance

```text
TRAINING
launch_2gpu.sh
  -> yaml_overrides.py reads nightly_best.yaml
  -> torchrun starts two pufferlib.pufferl processes
  -> load_config() composes the Hydra config
  -> train() creates a vector environment on each rank
  -> Drive.__init__() converts Python settings to C arguments
  -> binding.env_init() writes dt into Drive.dt
  -> c_step()
       -> policy action
       -> move_dynamics(..., dt)
       -> compute_metrics(..., dt)
       -> compute_rewards(..., dt)
       -> observation for the next transition
  -> PuffeRL stores 128-transition segments
  -> V-trace/GAE and PPO update the policy
  -> rank 0 periodically enters the benchmark evaluator

STANDALONE EVALUATION
evaluate_final_model.sh
  -> puffer eval selects carla, nuplan_single, nuplan_multi
  -> load_config() reads CLI overrides
  -> checkpoint config restores policy/observation architecture
  -> evaluation_benchmarks.yaml overwrites benchmark environment settings
  -> each benchmark creates Drive environments with dt=0.1
  -> mean policy action is recomputed every environment step
  -> the same C c_step() path produces trajectories and metrics
  -> episode_metrics.csv and evaluation_summary.json are written
```

The core source locations are:

- [training launcher](../third_run_nightly_best_config/train/launch_2gpu.sh#L57)
- [third-run training overrides](../third_run_nightly_best_config/override_config/nightly_best.yaml#L9)
- [`load_config()` and Hydra composition](../../pufferlib/pufferl.py#L2409)
- [`Drive` Python wrapper](../../pufferlib/ocean/drive/drive.py#L45)
- [Python-to-C binding](../../pufferlib/ocean/drive/binding.c#L1899)
- [`c_step()` simulator loop](../../pufferlib/ocean/drive/drive.h#L4491)
- [`move_dynamics()`](../../pufferlib/ocean/drive/drive.h#L4136)
- [`compute_metrics()`](../../pufferlib/ocean/drive/drive.h#L3233)
- [`compute_rewards()`](../../pufferlib/ocean/drive/drive.h#L3548)
- [PuffeRL rollout construction](../../pufferlib/pufferl.py#L124)
- [standalone evaluator](../../pufferlib/pufferl.py#L1673)

## 1. What the training command resolves

### 1.1 The launcher turns a flat YAML file into Hydra overrides

The launcher does not pass `dt` by hand. It reads the experiment-local YAML and
turns each scalar into a `key=value` argument:

```bash
# project/third_run_nightly_best_config/train/launch_2gpu.sh
mapfile -t CONFIG_ARGS < <(python "${PROJECT_DIR}/yaml_overrides.py" "${CONFIG}")

TRAIN_COMMAND=(
    torchrun --standalone --nnodes=1 --nproc-per-node=2 --max_restarts=0
    -m pufferlib.pufferl train puffer_drive "${CONFIG_ARGS[@]}"
)
```

See [launch_2gpu.sh:57](../third_run_nightly_best_config/train/launch_2gpu.sh#L57)
and [yaml_overrides.py:9](../third_run_nightly_best_config/yaml_overrides.py#L9).
The relevant values are:

```yaml
env.dt: 0.1
env.scenario_length: 7680
env.resample_frequency: 768000

train.total_timesteps: 10_000_000_000
train.gamma: 0.9996665555
train.gae_lambda: 0.9830475725
train.bptt_horizon: 128
```

These come from
[nightly_best.yaml:9](../third_run_nightly_best_config/override_config/nightly_best.yaml#L9).
The launcher separately pins 20 vector environments, vector batch size 10,
logical minibatch 128,000, and physical microbatch 32,000 per rank.

### 1.2 Hydra starts from the repository default and applies the overrides

The base config still declares `env.dt: 0.3` in
[puffer_drive.yaml](../../pufferlib/config/puffer_drive.yaml), but
`load_config()` composes all command-line overrides before converting the result
to an ordinary nested dictionary:

```python
# pufferlib/pufferl.py: load_config()
with initialize_config_dir(config_dir=config_dir, version_base=None):
    cfg = compose(config_name=env_name, overrides=overrides)

cfg["env"] = OmegaConf.merge(OmegaConf.structured(env_schema), cfg["env"])
args = defaultdict(dict, OmegaConf.to_container(cfg, resolve=True, ...))
```

The structured schema declares `dt` as a float at
[config_schema.py:87](../../pufferlib/config_schema.py#L87). It checks the
configuration shape and type, but it does not encode a dataset sampling rate or
check that replay data has the same clock.

For this command, the resolved training environment therefore has `dt=0.1`.
The saved third-run config confirms it at
[`config.yaml`](../../experiments/third_run_nightly_best_config/nightly_best_local_2gpu_2026-08-24_03-36-36_seed0/config.yaml).

### 1.3 Why the saved config says 5B although the experiment is 10B

`torchrun` sets `LOCAL_RANK` and `WORLD_SIZE=2`. `load_config()` divides the
transition target across ranks:

```python
if "LOCAL_RANK" in os.environ:
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    args["train"]["total_timesteps"] //= world_size
```

See [pufferl.py:2442](../../pufferlib/pufferl.py#L2442). Consequently, each
rank runs to 5B local agent transitions and the two ranks together represent
the requested 10B global transitions. The rank-0 `config.yaml` records the
already-divided value, 5B.

Nominal rollout geometry is:

| Quantity | Per rank | Two ranks |
|---|---:|---:|
| Resident policy-agent slots | `20 × 3,200 = 64,000` | 128,000 |
| Transitions per PPO rollout | `64,000 × 128 = 8,192,000` | 16,384,000 |
| Full rollouts needed to pass 10B | — | 611 |

`dt` does not change these tensor counts. It changes how much simulated time
each transition and each 128-transition row represents.

## 2. How `dt` reaches the C simulator

`train()` creates the vector environment from the resolved `args["env"]`:

```python
# pufferlib/pufferl.py
vecenv = vecenv or load_env(env_name, args, seed=env_seed)

# load_env()
return pufferlib.vector.make(make_env, env_kwargs=args["env"], **vec_kwargs)
```

See [`train()`](../../pufferlib/pufferl.py#L1432) and
[`load_env()`](../../pufferlib/pufferl.py#L2356). Each worker constructs a
Python `Drive` instance. `Drive.__init__()` stores the scalar directly:

```python
self.dt = dt
```

The wrapper includes it in every C environment's initialization dictionary:

```python
return {
    ...
    "dt": self.dt,
    "scenario_length": int(self.scenario_length),
    ...
}
```

See [`Drive._env_init_kwargs()`](../../pufferlib/ocean/drive/drive.py#L496).
Finally, `binding.env_init()` copies the Python number into the C `Drive`
structure:

```c
env->dt = (float) unpack(kwargs, "dt");
env->scenario_length = (int) unpack(kwargs, "scenario_length");
```

See [binding.c:1899](../../pufferlib/ocean/drive/binding.c#L1899). From this
point onward, the hot loop reads `env->dt`; Python does not perform the vehicle
integration.

## 3. One environment step is one policy decision interval

`c_step()` increments the integer timestep exactly once, moves every agent,
computes metrics and rewards, and then creates the next observation:

```c
env->timestep++;

if (agent->controller == CONTROLLER_POLICY) {
    move_dynamics(env, i, agent_idx);
} else if (agent->controller == CONTROLLER_REPLAY) {
    move_expert(env, agent_idx);
}

compute_metrics(env, agent_idx, i);
compute_rewards(env, i);
```

See [`c_step()`](../../pufferlib/ocean/drive/drive.h#L4491). PuffeRL obtains an
observation, runs the policy once, and sends one action for each call to this
step. Thus:

| `dt` | Decisions per simulated second | Time represented by one action |
|---:|---:|---:|
| 0.3 s | 3.333 Hz | 300 ms |
| 0.1 s | 10 Hz | 100 ms |

The policy observation contains speed, geometry, steering, acceleration, lane
state, stopped seconds, goals, partners, roads, and traffic controls. It does
**not** contain an explicit `dt` feature; see
[`write_ego_obs()`](../../pufferlib/ocean/drive/drive.h#L3706) and the other
observation writers in the same file. The policy can observe consequences of a
different clock, but it is not directly told which clock is active.

## 4. Discrete policy, continuous environment, and jerk actions

The exact run has two distinct action settings:

- `policy.action_type=discrete` is pinned by the third-run YAML.
- `env.action_type=continuous` is inherited from the main config.

The policy therefore predicts probabilities over the 12 combinations formed by
the physical jerk table:

```c
JERK_LONG = {-15.0, -4.0, 0.0, 4.0};  // m/s^3
JERK_LAT  = { -4.0,  0.0, 4.0};       // m/s^3
```

See [constants.h:139](../../pufferlib/ocean/drive/constants.h#L139). During
training, `sample_logits()` samples a discrete class and maps it back to the
continuous action expected by the environment. During standalone evaluation,
`action_selection=mean` computes the probability-weighted mean in physical
jerk space:

```python
mean_physical = probs @ self.action_table_physical.to(probs.dtype)
...
return torch.stack([mean_long_norm, mean_lat_norm], dim=-1).clamp_(-1.0, 1.0)
```

See [`sample_logits()`](../../pufferlib/pytorch.py#L193) and
[`discrete_probs_to_continuous_mean()`](../../pufferlib/ocean/torch.py#L501).
In both cases, a new command is produced on every environment step. Changing
`dt` therefore changes not only numerical integration but also how often the
policy is allowed to change its command.

## 5. `dt` in jerk vehicle dynamics

The selected dynamics model is `jerk`. The main integration equations in
[`move_dynamics()`](../../pufferlib/ocean/drive/drive.h#L4136) are:

```c
a_long_new = a_long + c_throttle * j_long * dt;
a_lat_new  = a_lat  + c_steer    * j_lat  * dt;

v_new = v + 0.5 * (a_long_new + a_long) * dt;
d     = 0.5 * (v_new + v) * dt;
theta = d * curvature;
```

The steering rate is also expressed per second:

```c
delta_steer = clip(
    steering_angle - agent->steering_angle,
    -0.6f * env->dt,
     0.6f * env->dt);
```

This gives a per-step steering change limit of 0.18 rad at `dt=0.3` and 0.06
rad at `dt=0.1`, while preserving the intended 0.6 rad/s rate. Likewise, a
constant `+4 m/s^3` longitudinal jerk changes acceleration by `+1.2 m/s^2` in
one 0.3-second step but only `+0.4 m/s^2` in one 0.1-second step.

Applying the same command for three 0.1-second steps approximately represents
the same physical duration as one 0.3-second step. It is not numerically
identical because the implementation applies clipping, acceleration and
velocity zero-crossing rules, steering limits, and curvature recomputation at
every discrete step. The policy can also choose three different actions during
those three smaller steps.

`reset_accel_on_stop=false` does not remove these differences. It only controls
whether acceleration is cleared when velocity crosses zero:

```c
if (signed_v * v_new < 0) {
    v_new = 0.0f;
    if (env->reset_accel_on_stop) {
        a_long_new = 0.0f;
        a_lat_new = 0.0f;
    }
}
```

The classic dynamics branch also multiplies acceleration, position, heading,
steering rate, and finite-difference jerk by `dt`, but it is inactive for these
commands. The IDM controller has its own `dt`-based speed and distance update in
[idm.h:646](../../pufferlib/ocean/drive/idm.h#L646); none of the configured
CARLA or nuPlan benchmark vehicle controllers uses IDM.

### 5.1 Geometry checks between discrete poses

A smaller `dt` also shortens the distance and heading change between consecutive
geometry checks. The implementation does not rely only on overlap at the new
pose: important checks sweep across the previous-to-current motion. Vehicle
collision uses [`check_moving_obb_collision()`](../../pufferlib/ocean/drive/drive.h#L1690),
offroad detection uses
[`check_segment_crosses_moving_box()`](../../pufferlib/ocean/drive/drive.h#L1347),
red-light violations test the swept path against the stop line in
[`check_red_light_violation()`](../../pufferlib/ocean/drive/drive.h#L1618), and
goal arrival tests the motion segment in
[`compute_metrics()`](../../pufferlib/ocean/drive/drive.h#L3513).

These swept checks reduce tunneling when one step covers a large distance, but
they do not make `dt=0.3` and `dt=0.1` mathematically identical. The offroad
implementation explicitly notes that its straight chord can under-cover a box
that rotates substantially within one step. Clipping, curved motion, three
opportunities to detect an event, and potentially three different policy
actions remain discretization differences.

## 6. Episode clocks, resets, and scenario resampling

### 6.1 C episode termination

The C environment increments `env->timestep` once per step and truncates at
`scenario_length`:

```c
if (env->timestep == env->scenario_length || early_reset) {
    ...
    env->truncations[i] = 1;
    ...
    c_reset(env);
}
```

See [drive.h:4601](../../pufferlib/ocean/drive/drive.h#L4601). Ignoring early
termination, physical episode duration is:

```text
episode seconds = scenario_length steps × dt seconds/step
```

The second-run and third-run CARLA training settings therefore both represent
768 seconds:

| Training clock | Steps | `dt` | Physical duration |
|---|---:|---:|---:|
| Previous | 2,560 | 0.3 s | 768 s |
| Third run | 7,680 | 0.1 s | 768 s |

### 6.2 Python map/scenario resampling

The Python wrapper has a separate global tick. After C steps all current
environments, it rebuilds the batch when the tick reaches
`resample_frequency`:

```python
binding.vec_step(self.c_envs)
self.tick += 1
...
if self.tick % self.resample_frequency == 0:
    self.tick = 0
    # close and construct the next map/scenario batch
```

See [`Drive.step()`](../../pufferlib/ocean/drive/drive.py#L631). Both settings
resample after 100 nominal full episodes:

```text
256000 / 2560 = 100
768000 / 7680 = 100
```

The ratio is preserved, but a fixed 10B-transition training budget completes
only one-third as many full-length episode equivalents at `dt=0.1`, because
each episode now consumes three times as many transitions.

### 6.3 Traffic-light time

Procedural CARLA traffic-light durations are sampled in seconds and converted
to step counts:

```c
int steps_green  = (int) (dur_green  / dt);
int steps_yellow = (int) (dur_yellow / dt);
int steps_red    = (int) (dur_red    / dt);
```

See [`generate_traffic_light_states()`](../../pufferlib/ocean/drive/drive.h#L2222).
This preserves the intended physical phase durations up to integer truncation.
The 0.1-second clock has finer phase quantization than the 0.3-second clock.

### 6.4 Robustness perturbation durations and triggers

Training defaults flag 2% of agents per episode for partner blindness and 2%
for phantom braking. For flagged agents, the trigger probability is 0.03 **per
step**, not per second. Changing from 0.3 to 0.1 therefore provides roughly
three times as many trigger opportunities in the same physical interval. For
example, conditional on an agent being flagged, the chance of at least one
trigger over three seconds is:

| Clock | Opportunities in 3 s | `1 - (1 - 0.03)^steps` |
|---|---:|---:|
| `dt=0.3` | 10 | 26.26% |
| `dt=0.1` | 30 | 59.90% |

The duration setting is converted from seconds to steps in Python:

```python
self.partner_blindness_duration_seconds = (
    float(partner_blindness_duration_seconds) // self.dt
)
self.phantom_braking_duration_seconds = (
    float(phantom_braking_duration_seconds) // self.dt
)
```

See [drive.py:287](../../pufferlib/ocean/drive/drive.py#L287). Despite the
attribute name, the resulting value is a step count by the time it reaches C.
Python floating-point floor division gives:

```text
3.0 // 0.3 = 10 steps = 3.0 s
3.0 // 0.1 = 29 steps = 2.9 s
```

Partner-blindness triggering occurs in
[`write_partner_obs()`](../../pufferlib/ocean/drive/drive.h#L3764), and phantom
braking triggers in `move_dynamics()`. Evaluation explicitly sets both flag and
trigger probabilities to zero, so these perturbations affect the training path
but not the three standalone benchmarks.

## 7. PPO time: transition counts are not seconds

### 7.1 BPTT horizon

PuffeRL derives its rollout buffer from agent count and the configured
transition horizon:

```python
if config["batch_size"] == "auto":
    config["batch_size"] = total_agents * config["bptt_horizon"]

horizon = config["bptt_horizon"]
```

See [`PuffeRL.__init__()`](../../pufferlib/pufferl.py#L124). A horizon of 128
always means 128 transitions; `dt` is not consulted here:

| Clock | BPTT transitions | Physical BPTT span |
|---|---:|---:|
| `dt=0.3` | 128 | 38.4 s |
| `dt=0.1` | 128 | 12.8 s |

Thus the batch contains the same number and shape of tensors but covers
one-third of the physical time per agent trajectory.

### 7.2 Gamma and lambda are per transition

The V-trace/GAE recursion uses `gamma` and `lambda` once for every stored
transition:

```cpp
delta = reward[t_next]
      + gamma * value[t_next] * nextnonterminal
      - value[t];
last = delta + gamma * lambda * c_t * last * nextnonterminal;
```

See [`puff_advantage_row()`](../../pufferlib/extensions/pufferlib.cpp#L28).
The third-run values were chosen so three 0.1-second decay steps match one old
0.3-second step:

```text
0.9996665555^3 = 0.9990000000186 ≈ 0.999
0.9830475725^3 = 0.9500000000245 ≈ 0.95
```

This preserves discount decay per physical time. It does not make the fixed
128-transition segment equivalent. The recursion starts each row with
`lastpufferlam = 0`, so it does not carry GAE state across BPTT rows. The
approximate remaining `(gamma × lambda)^128` weight at the row boundary is:

| Clock | `(gamma × lambda)^128` |
|---|---:|
| Old 0.3-second settings | 0.00124 |
| New 0.1-second settings | 0.10740 |

The shorter physical row therefore cuts the same physical-time decay process at
a different point, even though gamma and lambda themselves were converted.

### 7.3 Environment truncation bootstrapping is a separate mechanism

When Drive auto-resets at an episode truncation, the next observation already
belongs to the reset episode. With `use_value_bootstrapping=true`, PuffeRL adds
a heuristic bootstrap using the prior stored value:

```python
if l > 0 and config["use_value_bootstrapping"]:
    trunc_mask = (t > 0) & (d == 0)
    r = r + trunc_mask * config["gamma"] * self.values[batch_rows, l - 1]
```

See [pufferl.py:442](../../pufferlib/pufferl.py#L442). This handles Drive
episode truncations. It does not extend the GAE recursion across ordinary
128-transition buffer boundaries.

### 7.4 Fixed transition budget means different physical exposure

The training loop stops on agent transitions, not simulated seconds:

```text
physical agent-time = total agent transitions × dt
```

For the same 10B global transition budget:

| Clock | Global transitions | Nominal physical agent-time |
|---|---:|---:|
| `dt=0.3` | 10B | 3B agent-seconds |
| `dt=0.1` | 10B | 1B agent-seconds |

Similarly, every two-rank PPO rollout contains 16,384,000 transitions in both
settings, but represents about 4,915,200 versus 1,638,400 agent-seconds. This is
physical exposure summed over agents, not wall-clock runtime.

## 8. Reward terms do not all scale the same way

`compute_rewards()` mixes three kinds of terms:

1. rate-like terms explicitly multiplied by `dt`;
2. per-step terms without `dt`;
3. sparse event terms paid when an event occurs.

The active implementation is visible at
[`compute_rewards()`](../../pufferlib/ocean/drive/drive.h#L3548).

| Reward component | Current formula shape | Explicit `dt`? | Time interpretation |
|---|---|---:|---|
| lane alignment | coefficient × `dt` × alignment expression | Yes | Integrated rate-like term |
| lane center | coefficient × `dt` × distance expression | Yes | Integrated rate-like term |
| velocity progress | coefficient × `dt` × progress | Yes | Integrated rate-like term |
| timestep penalty | `-coefficient × dt` while moving | Yes | Cost per simulated second |
| reverse penalty | `-coefficient × dt` while reversing | Yes | Cost per simulated second |
| comfort | `-coefficient × violation_count` each step | No | Per-step penalty |
| overspeed | `-coefficient × binary_overspeed` each step | No | Per-step penalty |
| ADE | coefficient × displacement error each step | No | Per-step; disabled here because coefficient is zero |
| collision | base plus speed-dependent event penalty | No | Sparse event |
| offroad | fixed event penalty | No | Sparse event |
| red light | fixed event penalty | No | Sparse event |
| goal | fixed event bonus | No | Sparse event |

For an otherwise identical continuous trajectory, the rate-like terms are
designed to accumulate similarly over the same physical seconds. Comfort and
overspeed are evaluated three times as often at `dt=0.1` and are not multiplied
by the smaller step duration. Sparse event magnitudes remain the same, but a
fixed transition batch covers less physical time and fewer full episode
equivalents, so its event density is not held constant by configuration.

This table describes the formulas only. The realized rewards can also change
because the trajectories themselves change under finer control and collision
checking.

## 9. Evaluation metrics and their time units

Metrics are computed after movement on every C step. Their accumulators also
mix seconds, distances, per-step samples, and ever-event flags:

| Metric family | Implementation | Effect of uniform `dt` sampling |
|---|---|---|
| multi-lane time | `multi_lane_time += dt` | Accumulates seconds directly |
| distance travelled | `distance += speed × dt` | Integrates meters |
| speed violation | `sum += overspeed_mps × dt` | Integrates excess-speed seconds |
| wrong-way distance | `sum += signed_speed × dt` | Integrates meters |
| stopped duration | `seconds_stopped += dt` | Accumulates seconds directly |
| average speed | sum speed, later divide by steps | Time average for fixed `dt` |
| lane-center/alignment rates | count per step, divide by steps | Fraction of sampled time |
| TTC-within-bound rate | violating samples / all samples | Fraction of sampled time |
| collision/offroad/red-light rate | whether each agent ever triggered event | Event occurrence, not duration |
| episode length | increment by one per step | Unit is transitions, not seconds |

The direct integrations are in
[`compute_metrics()`](../../pufferlib/ocean/drive/drive.h#L3233) and stopped
seconds are updated in [`c_step()`](../../pufferlib/ocean/drive/drive.h#L4543).
Episode aggregation happens in
[`add_log()`](../../pufferlib/ocean/drive/drive.h#L2056).

`progress_ratio` explicitly reconstructs episode seconds:

```c
episode_duration_s = agent_log->episode_length * env->dt;
reference_progress_distance = reference_speed * episode_duration_s;
progress_ratio = distance_since_spawn / reference_progress_distance;
```

The Puffer score similarly uses `duration_steps × dt` for episode duration and
uses the `dt`-integrated speed violation. See
[`calculate_puffer_score()`](../../pufferlib/ocean/drive/drive.h#L2000).

### Existing comfort-score caveat

The reward path increments `comfort_violation_count`, but the Puffer
`comfort_score` reads a different field, `comfort_violation_timestep_count`:

```c
agent_log->comfort_score = calculate_duration_scaled_violation_score(
    agent_log->comfort_violation_timestep_count,
    agent_log->episode_length,
    env->dt);
```

In the current source, `comfort_violation_timestep_count` is declared and read
but has no update site. This is an existing metric implementation issue, not an
effect introduced by choosing 0.1 or 0.3. The training comfort reward uses
`comfort_violation_count` and is a separate path.

## 10. Periodic training evaluation

The training YAML requests `carla_fast` every 100 epochs. Rank 0 calls
`run_training_evaluation()`, which copies the live training arguments and live
policy into the general evaluator:

```python
eval_args = copy.deepcopy(args)
eval_args["eval"]["benchmarks"] = eval_args["train"]["evaluation_benchmarks"]
...
eval(..., args=eval_args, policy=policy, use_training_config=True)
```

See [`run_training_evaluation()`](../../pufferlib/pufferl.py#L2224). The shared
benchmark config also specifies `dt=0.1`; `carla_fast` then supplies 500 steps.
The resulting nominal scenario duration is 50 seconds.

The periodic evaluator intentionally restores the live training lane and
boundary dropout values after loading the shared benchmark environment. This
means it is not identical to standalone final evaluation, but both use the same
0.1-second environment clock in this third-run workflow.

## 11. Standalone evaluation config precedence

The standalone launcher selects all three benchmarks and does not pass an
`env.dt` override directly:

```bash
puffer eval puffer_drive carla,nuplan_single,nuplan_multi \
  load_model_path=.../final_model.pt \
  eval.benchmark_config=.../evaluation_benchmarks.yaml \
  eval.num_agents=300 \
  eval.action_selection=mean
```

See [evaluate_final_model.sh:60](../third_run_nightly_best_config/eval/evaluate_final_model.sh#L60).
The evaluation merge order is:

```text
base puffer_drive config
  -> checkpoint policy, RNN, and recognized environment settings
  -> shared benchmark env settings
  -> selected benchmark's env settings
  -> explicit CLI dot-list overrides
```

The checkpoint import occurs in
[`load_checkpoint_architecture()`](../../pufferlib/ocean/evaluation_utils/evaluation_utils.py#L203).
Then `build_benchmark_args()` applies the shared environment before the
benchmark-specific environment:

```python
args["env"].update(environment_config)
args["env"].update(benchmark_environment_config)
args["env"]["resample_frequency"] = benchmark_environment_config["scenario_length"]
```

See [`build_benchmark_args()`](../../pufferlib/ocean/evaluation_utils/evaluation_utils.py#L258).
Because the shared environment contains `dt: 0.1`, it overwrites any checkpoint
clock. The benchmark-specific blocks do not replace `dt`, so all inherit 0.1.
The evaluator finally reapplies explicit CLI overrides after building each
benchmark; the supplied launcher has no `env.dt=...` CLI override.

The resolved benchmark settings are:

| Benchmark | Simulation | Controllers | Steps | `dt` | Nominal duration |
|---|---|---|---:|---:|---:|
| `carla` | Gigaflow | all vehicles policy | 6,000 | 0.1 s | 600 s |
| `carla_fast` | Gigaflow | all vehicles policy | 500 | 0.1 s | 50 s |
| `nuplan_single` | replay | SDC policy; background replay | 200 | 0.1 s | 20 s |
| `nuplan_multi` | replay | vehicles policy; non-vehicles replay | 200 | 0.1 s | 20 s |

These values are defined in
[evaluation_benchmarks.yaml](../third_run_nightly_best_config/override_config/evaluation_benchmarks.yaml#L1).
The completed seed-0 evaluation also records the fully merged arguments in each
`resolved_benchmark.yaml`, for example the
[CARLA resolution](../../experiments/third_run_nightly_best_config/nightly_best_local_2gpu_2026-08-24_03-36-36_seed0/eval/carla_final_model_mean_metrics/20260824-160508/resolved_benchmark.yaml)
and
[nuPlan-single resolution](../../experiments/third_run_nightly_best_config/nightly_best_local_2gpu_2026-08-24_03-36-36_seed0/eval/nuplan_single_final_model_mean_metrics/20260824-160508/resolved_benchmark.yaml).

## 12. CARLA and nuPlan do not consume the clock identically

### CARLA/Gigaflow

All controlled vehicles call `move_dynamics()` on every step. `dt` defines the
policy interval and every vehicle integration interval. Procedural traffic-light
states are generated using the same `dt`. There is no logged trajectory clock
to satisfy.

### nuPlan replay

The C loop still increments one integer timestep per environment step. A replay
controller uses that integer directly as a logged-array index:

```c
int t = env->timestep;
agent->sim_x = agent->log_trajectory_x[t];
agent->sim_y = agent->log_trajectory_y[t];
agent->sim_heading = agent->log_heading[t];
```

See [`move_expert()`](../../pufferlib/ocean/drive/drive.h#L2799). The replay
agent therefore advances exactly one stored frame per policy step regardless of
the numeric `env->dt` value. The binary loader stores trajectory arrays and a
trajectory length, but no replay-frequency or `dt` metadata is read; see
[map_data.h:370](../../pufferlib/ocean/drive/map_data.h#L370). The repository's
[nuPlan data guide](../../docs/nuplan_data.md) describes 20-second scenarios
used with 200 steps, establishing the intended 10 Hz interpretation outside the
binary itself.

Configured `dt` still affects replay evaluation in two ways:

1. Policy-controlled agents use `move_dynamics()` with `dt` while background
   replay agents advance by logged index.
2. Replay yaw rate is derived from adjacent logged headings divided by
   `env->dt`:

   ```c
   agent->yaw_rate = compute_log_yaw_rate(agent, t, env->dt);
   ```

In `nuplan_single`, only the SDC is policy-controlled and all background agents
are replayed. In `nuplan_multi`, vehicles are policy-controlled while
non-vehicle agents remain replayed. No initialization check verifies that the
configured `dt` matches the source dataset's frame interval.

## 13. What was preserved and what was not

This table summarizes implementation equivalence without interpreting observed
scores.

| Quantity | Preserved by the third-run conversion? | Reason |
|---|---|---|
| Nominal CARLA episode seconds | Yes | `2560×0.3 = 7680×0.1 = 768` |
| Episodes per map-resample interval | Yes | Both ratios are 100 |
| Gamma decay per 0.3 physical seconds | Yes | New gamma cubed is old gamma |
| Lambda decay per 0.3 physical seconds | Yes | New lambda cubed is old lambda |
| Traffic-light phase seconds | Approximately | Seconds are divided by `dt`, then truncated to integer steps |
| Steering-rate limit per second | Yes by formula | Per-step cap is `0.6×dt` |
| Policy decisions per second | No | 3.33 Hz becomes 10 Hz |
| Physical BPTT span | No | 38.4 s becomes 12.8 s |
| Physical exposure in 10B transitions | No | 3B becomes 1B agent-seconds |
| Full episode equivalents in 10B transitions | No | Each full episode uses 3× more transitions |
| GAE continuation at BPTT boundary | No | Horizon stays 128 while gamma/lambda are closer to one |
| Per-second comfort reward scale | No | Comfort penalty has no `dt` factor |
| Per-second overspeed reward scale | No | Overspeed penalty has no `dt` factor |
| Sparse event reward density per fixed transition budget | No guaranteed equivalence | Budget covers less physical time and fewer episode equivalents |
| Perturbation trigger hazard per second | No | Probability remains per step |
| Perturbation duration conversion | Approximately | Floor division can lose one step |
| nuPlan frame advancement | Fixed at one frame per step | Independent of numeric `dt`; compatibility is assumed |
| Explicit clock information in policy observation | No | `dt` is not an observation field |

The central implementation lesson is that changing `dt` changes the definition
of one MDP transition. Rescaling episode lengths and discount coefficients
preserves selected physical-time quantities, but it cannot by itself preserve
the policy decision process, PPO segmentation, reward mixture, scenario count,
or replay-index semantics.

## Code navigation index

Use these entry points to continue reading the implementation:

| Question | File and symbol |
|---|---|
| Where does training set `dt`? | [third-run `nightly_best.yaml`](../third_run_nightly_best_config/override_config/nightly_best.yaml#L9) |
| How does YAML become CLI overrides? | [`yaml_overrides.py`](../third_run_nightly_best_config/yaml_overrides.py#L9) |
| How does Hydra resolve the final config? | [`load_config()`](../../pufferlib/pufferl.py#L2409) |
| Why does each DDP rank show 5B? | [`load_config()` world-size division](../../pufferlib/pufferl.py#L2442) |
| Where is the vector environment built? | [`load_env()`](../../pufferlib/pufferl.py#L2356) |
| Where does Python store and forward `dt`? | [`Drive.__init__()`](../../pufferlib/ocean/drive/drive.py#L45), [`_env_init_kwargs()`](../../pufferlib/ocean/drive/drive.py#L496) |
| Where does C receive it? | [`binding.env_init()`](../../pufferlib/ocean/drive/binding.c#L1899) |
| What happens in one simulator step? | [`c_step()`](../../pufferlib/ocean/drive/drive.h#L4491) |
| How are jerk actions integrated? | [`move_dynamics()`](../../pufferlib/ocean/drive/drive.h#L4136) |
| How are events checked between discrete poses? | [`check_moving_obb_collision()`](../../pufferlib/ocean/drive/drive.h#L1690), [`check_segment_crosses_moving_box()`](../../pufferlib/ocean/drive/drive.h#L1347), [`check_red_light_violation()`](../../pufferlib/ocean/drive/drive.h#L1618) |
| What are the physical jerk choices? | [`JERK_LONG` and `JERK_LAT`](../../pufferlib/ocean/drive/constants.h#L139) |
| How are traffic-light seconds converted? | [`generate_traffic_light_states()`](../../pufferlib/ocean/drive/drive.h#L2222) |
| How do replay agents advance? | [`move_expert()`](../../pufferlib/ocean/drive/drive.h#L2799) |
| Which rewards multiply by `dt`? | [`compute_rewards()`](../../pufferlib/ocean/drive/drive.h#L3548) |
| Which metrics integrate time or distance? | [`compute_metrics()`](../../pufferlib/ocean/drive/drive.h#L3233) |
| How are episode metrics aggregated? | [`add_log()`](../../pufferlib/ocean/drive/drive.h#L2056) |
| How is rollout length chosen? | [`PuffeRL.__init__()`](../../pufferlib/pufferl.py#L124) |
| Where are gamma and lambda applied? | [`puff_advantage_row()`](../../pufferlib/extensions/pufferlib.cpp#L28) |
| What is the truncation bootstrap heuristic? | [PuffeRL rollout storage](../../pufferlib/pufferl.py#L442) |
| How does periodic evaluation start? | [`run_training_evaluation()`](../../pufferlib/pufferl.py#L2224) |
| How does standalone evaluation merge configs? | [`eval()`](../../pufferlib/pufferl.py#L1673), [`build_benchmark_args()`](../../pufferlib/ocean/evaluation_utils/evaluation_utils.py#L258) |
| How is mean action produced? | [`sample_logits()`](../../pufferlib/pytorch.py#L193), [`discrete_probs_to_continuous_mean()`](../../pufferlib/ocean/torch.py#L501) |
| Where are benchmark clocks and controllers defined? | [`evaluation_benchmarks.yaml`](../third_run_nightly_best_config/override_config/evaluation_benchmarks.yaml#L1) |
