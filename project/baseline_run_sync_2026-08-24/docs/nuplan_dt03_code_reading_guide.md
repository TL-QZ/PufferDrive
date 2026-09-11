# Code-reading guide: replay resampling from dt=0.1 to dt=0.3

Use this document as a map through the change. Read the files in order. Each
stop answers one question and tells you what to look for next.

The complete path is:

```text
experiment YAML
    -> Hydra schema and defaults
    -> Drive.__init__ validation
    -> preliminary C scenario load for agent counting
    -> actual C environment load
    -> binary timing validation and array compaction
    -> unchanged reset and step logic
    -> ground-truth export boundary
    -> evaluation override back to dt=0.1
```

## Before reading: the behavior in one example

A nuPlan replay file stores a state every `log_dt=0.1` seconds. The inspected
recording has 201 states:

```text
source index:  0   1   2   3   4   5   6  ... 198 199 200
time (s):    0.0 0.1 0.2 0.3 0.4 0.5 0.6 ... 19.8 19.9 20.0
kept:         X           X           X  ...   X
```

The requested environment step is `dt=0.3`, so the loader computes:

```text
stride = dt / log_dt = 0.3 / 0.1 = 3
retained indices = 0, 3, 6, ..., 198
retained states = floor((201 - 1) / 3) + 1 = 67
transitions = retained states - 1 = 66
duration = 66 * 0.3 seconds = 19.8 seconds
```

Index 200 is not used because it would require a shorter 0.2-second final
transition. There is no interpolation or extrapolation.

## Stop 1: start at the experiment override

Open:

```text
project/baseline_run_sync_2026-08-24/override_config/nuplan_sdc_finetune.yaml
```

Find these fields:

```yaml
env.simulation_mode: replay
env.dt: 0.3
env.resample_replay_to_dt: true
env.init_step: 0
env.init_step_spread: false
env.scenario_length: 66
env.resample_frequency: 66
env.num_maps: 169715
```

What each field controls:

- `simulation_mode: replay` means agents can consume recorded trajectories.
- `dt: 0.3` is the new simulator transition duration.
- `resample_replay_to_dt: true` opts into changing the loaded time grid.
- `init_step` and `init_step_spread` are fixed because resampling currently
  supports only episodes beginning at the first retained state.
- `scenario_length: 66` is a transition count. The loader must retain 67 states.
- `resample_frequency: 66` replaces the scenario after one complete episode.
- `num_maps: 169715` makes every current training `.bin` eligible for random
  sampling. Training draws with replacement, so scenarios may repeat before
  every scenario has appeared.

Do not confuse `scenario_length` with the number of stored states. A trajectory
needs an initial state before its first transition.

Next, verify that this experiment setting is optional rather than global.

## Stop 2: find the schema and disabled default

Open these two files:

```text
pufferlib/config_schema.py
pufferlib/config/puffer_drive.yaml
```

Search for `resample_replay_to_dt`. The schema admits one boolean field, and the
main configuration sets it to `false`. This protects every existing CARLA and
replay workflow unless it explicitly enables resampling.

The standalone C configuration path has the same field in:

```text
pufferlib/ocean/env_config.h
pufferlib/ocean/drive/drive.c
```

`env_config.h` parses only `true`, `false`, `1`, or `0`. `drive.c` copies the
parsed value into the `Drive` struct in both of its construction paths. This
path is separate from the Python extension, so matching defaults matter.

Useful navigation command:

```bash
rg -n "resample_replay_to_dt" \
  pufferlib/config_schema.py pufferlib/config/puffer_drive.yaml \
  pufferlib/ocean/env_config.h pufferlib/ocean/drive/drive.c
```

Next, follow the Hydra value into the Python environment.

## Stop 3: follow configuration through Drive.__init__

Open:

```text
pufferlib/ocean/drive/drive.py
```

Read `Drive.__init__`, starting at its parameters, then search within it for
`self.resample_replay_to_dt`. Python performs cheap configuration checks before
opening any scenario:

```text
enabled -> replay mode only
enabled -> init_step must be 0
enabled -> init_step_spread must be false
enabled -> dt must be finite and positive
enabled -> scenario_length must be positive
```

These checks give an immediate Python `ValueError`. They do not replace the C
checks: C receives untrusted timing metadata from each binary and must validate
that metadata itself.

Continue in the same constructor until the call to `binding.shared(...)`.
This is the first of two scenario-loading paths.

## Stop 4: understand why each scenario is loaded twice

Still in `drive.py`, read the `binding.shared(...)` call before
`super().__init__(buf=buf)`.

Replay scenarios contain different numbers of controllable agents. Python
needs the count before it can divide its NumPy buffers among C environments.
The shared binding therefore loads candidate scenarios, calls
`set_active_agents`, records agent offsets, and releases those temporary loads.

The new arguments passed into this preliminary load are:

```text
resample_replay_to_dt, dt, scenario_length,
init_step, init_step_spread, simulation_mode
```

Now open:

```text
pufferlib/ocean/drive/binding.c
```

Read `my_shared`. Start where it unpacks `simulation_mode` and `init_step`, then
follow `timing_config` into the replay branch. For every candidate `map_file`, it:

1. creates a temporary zeroed `Drive`;
2. copies the timing settings into it;
3. calls `load_map_binary`;
4. selects active agents from the prepared arrays;
5. calls `free_loaded_map` before releasing the temporary environment.

Resampling here is necessary because agent validity at retained timestep zero
determines which agents are active. It also ensures preliminary counting rejects
the same bad scenario that the real environment would reject.

Next, return to `drive.py` and read `_create_c_envs`.

## Stop 5: follow the actual environment construction

`Drive._create_c_envs` uses the offsets from the preliminary pass to create the
real C environments. Its `try/except` makes batch creation atomic: if a later
scenario fails, every earlier `env_id` is closed before the exception returns.

`Drive._env_init_kwargs` builds the dictionary passed to `binding.env_init`.
Find these keys there:

```text
dt, resample_replay_to_dt, init_step_spread,
scenario_length, map_file, simulation_mode
```

Return to `binding.c` and read `my_init`. It copies those values into the real
`Drive`, then calls `init(env)`. If loading fails, the binding raises a Python
error containing both the scenario path and the loader's field-specific reason.

Now open the `Drive` struct and initialization function in:

```text
pufferlib/ocean/drive/drive.h
```

The struct owns `resample_replay_to_dt`, `init_step_spread`, and `load_error`.
Read `init(Drive *env)`: `load_map_binary` runs before grid construction, cache
attachment, active-agent selection, collision filtering, or start-position setup.

That order is the central design choice: temporal data is fully prepared before
any simulator subsystem consumes it.

## Stop 6: read the loader validation before the copy loop

Open:

```text
pufferlib/ocean/drive/map_data.h
```

Read these functions in order:

```text
validate_replay_timing_config
prepare_replay_timing
free_loaded_map
allocate_map_field and read_map_field
load_map_binary
```

`validate_replay_timing_config` validates settings known before opening a file.
`load_map_binary` then reads the external binary with byte-count and allocation
guards. Near its end, after metadata such as `log_length` and `log_dt` is known,
it calls `prepare_replay_timing`.

Inside `prepare_replay_timing`, read the checks before the mutation loop:

1. `log_dt` must be finite and positive.
2. `dt / log_dt` must be a positive integer within absolute tolerance `1e-5`.
3. `log_length` must contain the requested number of full transitions.
4. Every agent `trajectory_size` must equal `log_length`.
5. Every nonempty traffic-light `state_size` must equal `log_length`.

The enough-data check is equivalent to:

```text
floor((source_count - 1) / stride) >= scenario_length
```

The left side is the number of full target-duration transitions available.

Next, read the two compaction loops in the same function.

## Stop 7: inspect exactly what is resampled

For each agent, `prepare_replay_timing` applies `source_idx = sample_idx * stride`
to ten arrays:

| Category | Arrays |
|---|---|
| Position | `log_trajectory_x`, `log_trajectory_y`, `log_trajectory_z` |
| Orientation | `log_heading` |
| Velocity | `log_velocity_x`, `log_velocity_y` |
| Dimensions | `log_length`, `log_width`, `log_height` |
| Presence | `log_valid` |

It applies the same indices to every nonempty `TrafficControlElement.states`
history. The loop compacts in place from low indices to high indices. Because
`source_idx >= sample_idx`, writing one retained sample cannot overwrite a
future source sample.

After compaction:

```text
agent->trajectory_size = retained_count
traffic->state_size = retained_count
env->log_length = retained_count
env->log_dt = env->dt
```

The loader does not change routes, IDs, map geometry, lane graphs, static goals,
or velocity values. Sampling a velocity measurement does not multiply it by the
stride; its unit remains meters per second.

The arrays keep their original allocation capacity. Only their logical lengths
change. That avoids new allocations and makes failure cleanup use the ordinary
owners.

If validation or reading fails, `load_map_binary` jumps to `load_failure`, closes
the file, calls `free_loaded_map`, and returns `-1`. Read `free_loaded_map` beside
the ordinary `c_close` function to compare partial-load cleanup with full
environment cleanup.

Next, verify why cache sharing cannot mix original and resampled trajectories.

## Stop 8: separate shared geometry from private time arrays

In `drive.h`, read `init` from its `map_cache_lookup` call through the cache-hit
and cache-miss branches. Then inspect `SharedMapData` near the top of the file.

Only road geometry, the grid, neighbor offsets, and the lane graph enter
`SharedMapData`. Agents and traffic controls remain owned by each `Drive`.
Therefore these two environments may coexist safely:

```text
environment A: default replay, 201 private states at dt=0.1
environment B: resampled replay, 67 private states at dt=0.3
shared: road elements, grid map, neighbor offsets, lane graph
```

On a cache hit, the newly loaded road geometry is discarded and replaced with
the shared geometry. The newly loaded and resampled agents and traffic lights
stay attached to that environment.

Next, check that the hot step path did not gain resampling work.

## Stop 9: trace reset and stepping through existing arrays

In `drive.h`, search for these functions:

```text
set_start_position
move_expert
c_reset
c_step
```

The resampling change does not add branches or allocations to `c_step`.
`set_start_position` reads retained state zero. On later steps, `move_expert`
uses `env->timestep` as an index into the same `log_*` arrays it used before.
Rewards, collision handling, dynamics, metrics, and observations keep their
existing algorithms. They see a trajectory whose logical timestep is now 0.3
seconds because loading already compacted it and updated `log_dt`.

At `scenario_length == 66`, the normal episode boundary resets the environment.
At Python tick 66, `Drive.step` also reaches `resample_frequency`, closes the
current C batch, selects the next scenario batch, calls `_create_c_envs`, and
sets truncations to mark the external map-switch boundary.

The PPO collector stores those terminal/truncation markers. With
`rnn_name: null`, its 1,000-transition horizon is buffer organization rather
than recurrent backpropagation. Multiple 66-transition episodes can occupy the
same rollout, while the done markers stop advantage propagation across resets.

Next, inspect the one place where states and transitions need different sizes.

## Stop 10: verify the ground-truth export boundary

Return to `drive.py` and read `get_ground_truth_trajectories`. With resampling
enabled it allocates:

```text
state_count = scenario_length - init_step + 1
```

For this experiment, `state_count = 66 + 1 = 67`. The returned temporal arrays
have shape `(num_agents, 1, 67)` after the function adds its singleton scenario
axis.

Now read:

```text
pufferlib/ocean/env_binding.h
```

`validate_resampled_trajectory_outputs` checks dtype, contiguity, agent count,
and the 67-state width before C writes. The vector export also requires every C
environment to use the same resampled state count.

Finally, return to `drive.h` and read
`c_get_global_ground_truth_trajectories`. The enabled path writes exactly
`scenario_length + 1` states. The disabled path retains its old loop bound and
output behavior.

This boundary matters because confusing 66 transitions with 66 states would
either omit the final state or write beyond a 66-column output buffer.

Next, follow the checkpoint into evaluation and confirm the override order.

## Stop 11: see why evaluation remains at dt=0.1

Open:

```text
project/baseline_run_sync_2026-08-24/override_config/evaluation_benchmarks.yaml
```

Its shared environment section explicitly contains:

```yaml
dt: 0.1
resample_replay_to_dt: false
```

Then read `build_benchmark_args` in:

```text
pufferlib/ocean/evaluation_utils/evaluation_utils.py
```

Checkpoint architecture and observation fields may be loaded first. Benchmark
environment values are applied afterward, so the two lines above override the
fine-tuning checkpoint during:

```text
periodic training evaluation
standalone final-model evaluation
CSV-selected failure replay
```

The experiment's train, eval, and render launchers all use the dt03 run prefix,
so results do not mix with the earlier `nuplan_sdc_finetune` experiment.

Next, read the experiment-local resource and resume code.

## Stop 12: finish at launch, resume, and sizing

Read these files in order:

```text
project/baseline_run_sync_2026-08-24/train/finetune_config.py
project/baseline_run_sync_2026-08-24/train/launch_nuplan_sdc_finetune_2gpu.sh
project/baseline_run_sync_2026-08-24/train/profile_nuplan_dt03.py
```

`finetune_config.py` reads the flat experiment YAML, optionally applies a
measured resource file, validates the allowed concurrency and microbatch, and
rejects an incompatible existing run configuration.

The shell launcher locates exactly one CARLA source run for the chosen seed and
stages only `final_model.pt` plus `config.yaml`. It never copies the CARLA
`trainer_state.pt`. A fresh fine-tune therefore starts with checkpoint weights
and new optimizer state. A resume may use only the dt03 run's own trainer state.

`profile_nuplan_dt03.py` is a bounded measurement entrypoint. It exercises the
real checkpoint, environment, rollout collector, and PPO update for candidate
concurrencies 16, 32, and 64. It writes measurements outside production run
discovery and never starts a complete fine-tune. GPU execution still needs to
be run on two usable allocated devices.

## Stop 13: use tests as executable examples

Read the smallest, most direct test first:

```text
tests/drive/test_drive_replay_resampling.c
```

`test_exact_samples` compares every retained field against source index
`sample_idx * 3`. `test_invalid_timing_and_lengths` shows each loader rejection.
`test_cache_and_export_boundary` creates default and resampled instances that
share geometry while keeping different private trajectories.

Then read:

```text
tests/unit_tests/test_replay_resampling.py
```

This covers Python validation, malformed files, partial-creation cleanup,
map switches, export shape checks, and fixed-seed default compatibility.

Finish with:

```text
project/baseline_run_sync_2026-08-24/train/test_dt03_workflow.py
```

This checks resolved Hydra/DDP arithmetic, evaluation merge order, resume
compatibility, profiler selection rules, launcher dry runs, and an optional
real-checkpoint CPU rollout/update/resume integration.

Run the focused executable examples from the repository root:

```bash
source .venv/bin/activate
python -m pytest tests/unit_tests/test_replay_resampling.py \
  project/baseline_run_sync_2026-08-24/train/test_dt03_workflow.py -q
make -C tests/drive build/test_drive_replay_resampling
tests/drive/build/test_drive_replay_resampling
```

## Questions to answer after your own read

Use these as checkpoints. If you can answer them from the code, you have traced
the whole change:

1. Why does a 66-transition episode need 67 stored states?
2. Why must preliminary agent counting use the resampled arrays too?
3. Why can cached geometry be shared while trajectory arrays cannot?
4. Which exact function guarantees that evaluation overrides the checkpoint's
   `dt=0.3` setting?
5. Which functions prove that no resampling allocation or copy occurs in
   `c_step`?
