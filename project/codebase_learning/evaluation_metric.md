# PufferDrive Evaluation Metrics

This document describes the metrics emitted by the standalone benchmark evaluator into:

- `episode_metrics.csv`: one row per completed evaluation scenario;
- `evaluation_summary.json`: aggregates over all emitted CSV rows.

The definitions below follow the current implementation used by the CARLA evaluation in:

```text
experiments/second_run_nightly_best_config/
  nightly_best_local_2gpu_2026-08-14_15-33-43_seed0/
  eval/carla_final_model_mean_metrics/20260815-200339/
```

The resolved CARLA configuration relevant to interpretation is:

```yaml
num_scenarios: 1000
env:
  compute_eval_metrics: true
  dt: 0.1
  scenario_length: 6000
  max_agents_per_env: 50
  num_goals: 3
  collision_behavior: stop
  offroad_behavior: stop
  traffic_light_behavior: stop
```

Thus one complete scenario lasts 6,000 simulator steps, or 600 simulated seconds, and normally contains 50 controlled agents.

## Scope

This document covers evaluation values, supporting denominator fields, and the JSON reduction. It intentionally excludes:

- `map_name`, `scenario_id`, and `seed`, which identify or reproduce the scene;
- `episode_timestep` and `agents_per_batch`, which are evaluator bookkeeping;
- every `reward_components/*` column, which deserves a separate reward deep dive.

## The most important denominator rule

Most CSV values are **averages over controlled agents in one scenario**.

For every controlled agent, C keeps a separate `Log`. At the scenario boundary, `add_log()` combines those agent logs. `my_episode_to_dict()` then divides every ordinary accumulated field by `n`, the number of controlled agents.

Let:

- `N` be the number of controlled agents in a scenario (`n`);
- `T` be the scenario's final simulator timestep;
- `L_i` be agent `i`'s logged episode length;
- `I(condition)` be 1 when a condition is true and 0 otherwise.

For this CARLA run, normally `N = 50`, `T = 6000`, and `L_i = 6000`.

There are three major metric shapes:

1. **Ever-event agent rates**

   ```text
   (number of agents that experienced the event at least once) / N
   ```

   This includes `offroad_rate`, `collision_rate`, `red_light_violation_rate`, and `at_fault_collision_rate`.

2. **Timestep-normalized agent averages**

   ```text
   (1 / N) * sum_i(agent accumulator_i / T)
   ```

   This includes `avg_speed_per_agent`, `lane_center_rate`, `velocity_progress_sum`, and `comfort_violation_count`.

3. **Averages of per-agent final scores**

   ```text
   (1 / N) * sum_i(final agent score_i)
   ```

   This includes `progress_ratio`, `speed_limit_compliance`, `multi_lane_score`, and `puffer_score`.

This means the answer to the motivating question is:

> `offroad_rate` is the proportion of controlled agents that went off-road at least once during the scenario. It is not the proportion of agent-timesteps spent off-road.

For example, `offroad_rate = 0.02` in a 50-agent CARLA row means exactly one of the 50 agents triggered the off-road event at least once.

## Quick-reference table

| CSV metric | Range or unit | Scenario-row meaning | Better direction |
|---|---:|---|---|
| `n` | agents | Number of controlled agents used as the agent-level denominator | Context only |
| `active_agent_count` | agents | Number of policy-controlled slots created for the scenario | Context only |
| `episode_length` | simulator steps | Mean logged episode length over controlled agents | Context-dependent |
| `episode_return` | reward units | Mean cumulative reward per controlled agent | Higher, for the fixed reward definition |
| `offroad_rate` | `[0, 1]` | Fraction of agents that ever triggered off-road | Lower |
| `collision_rate` | `[0, 1]` | Fraction of agents that ever collided | Lower |
| `at_fault_collision_rate` | `[0, 1]` | Fraction of agents that ever had a collision classified as their fault | Lower |
| `red_light_violation_rate` | `[0, 1]` | Fraction of agents that ever triggered a red-light violation | Lower |
| `total_infraction_count` | agents | Number of agents with any collision, off-road, or red-light event | Lower |
| `total_distance_travelled_sum` | meters | Distance accumulated by all controlled agents | Context-dependent |
| `avg_distance_per_infraction` | meters | Total distance divided by the number of agents with any infraction | Higher, with caveats |
| `num_goals_reached` | goals/agent | Mean number of goal waypoints reached per agent | Usually higher |
| `score` | `[0, 1]` | Fraction of agents that reached at least `num_goals` goals and ended neither stopped nor removed | Higher |
| `dnf_rate` | `[0, 1]` | Fraction with no listed infraction and zero goals reached | Lower |
| `avg_speed_per_agent` | m/s | Mean speed over agents and scenario timesteps | Context-dependent |
| `velocity_progress_sum` | `[0, 1]` | Mean positive lane-alignment factor while moving forward faster than 2.5 m/s | Higher |
| `lane_center_rate` | `[0, 1]` | Fraction of agent-timesteps within 0.5 m of selected lane center | Higher |
| `comfort_violation_count` | violations/agent-step | Mean count of acceleration/jerk threshold components violated per agent-step | Lower |
| `progress_ratio` | normally `>= 0` | Distance travelled divided by distance at a 10 m/s reference speed | Higher; 1 means 10 m/s average |
| `making_progress_rate` | `[0, 1]` | Fraction of agents with `progress_ratio > 0.2` | Higher |
| `driving_direction_score` | `{0, .5, 1}` averaged | Mean tiered score from cumulative wrong-way distance | Higher |
| `speed_limit_compliance` | `[0, 1]` | Mean penalty score from time-integrated amount above the speed limit | Higher |
| `comfort_score` | `[0, 1]` | **BROKEN:** intended duration-scaled comfort score; currently always 1 because its accumulator is never updated | Do not interpret |
| `multi_lane_time` | seconds | Mean accumulated time that an agent straddles lanes while moving | Lower |
| `multi_lane_score` | `[0, 1]` | Mean tiered score derived from each agent's multi-lane time | Higher |
| `puffer_score` | `[0, 1]` | Mean gated weighted driving score over agents | Higher |

## Supporting fields

### `n`

`n` is the number of controlled agents whose logs were folded into the row.

```text
n = N
```

In the example CARLA output, every row has `n = 50`. It is the denominator for most per-scenario rates and averages.

### `active_agent_count`

This is the environment's number of policy-controlled agents. It describes the number of active slots created at initialization, not the number of agents still driving at the final timestep. Agents that become stopped or removed do not decrement this field.

For the current fixed-count Gigaflow evaluation, `active_agent_count` and `n` are both 50.

### `episode_length`

Each controlled agent's `episode_length` is incremented once on every environment step, including after that agent has become stopped or removed. The CSV field is the mean over agents:

```text
episode_length = (1 / N) * sum_i(L_i)
```

For this run it is 6,000 steps in every row. With `dt = 0.1`, that corresponds to 600 seconds. An environment-level early reset could make it smaller, but individual early agent stopping does not.

### `episode_return`

For each agent, the environment sums its scalar reward over the episode. The row reports the mean cumulative return over controlled agents:

```text
episode_return = (1 / N) * sum_i(sum_t reward_i,t)
```

After an agent is stopped or removed, reward computation is skipped and it contributes no further reward. This value depends on the configured reward coefficients and therefore should only be compared between evaluations using the same reward definition.

## Infraction metrics

### `offroad_rate`

Each agent owns a persistent binary off-road flag. It becomes 1 after the first timestep where off-road is detected and stays 1 in that agent's episode log.

The scenario row is:

```text
offroad_rate = (1 / N) * sum_i I(agent i was ever off-road)
```

Off-road is triggered when any of the following current checks fires:

- the agent position is outside the spatial grid;
- the nearby-road query returns no road entities;
- the vehicle's swept oriented box crosses a mapped road-edge segment at compatible elevation.

With `offroad_behavior: stop`, the first detected off-road event stops that agent. The metric remains an agent-level incidence rate, not time off-road.

### `collision_rate`

Each agent owns a persistent binary collision flag. It becomes 1 after any detected vehicle collision:

```text
collision_rate = (1 / N) * sum_i I(agent i collided at least once)
```

Collision detection uses moving oriented bounding boxes. This metric includes both at-fault and not-at-fault collisions. With `collision_behavior: stop`, an agent normally stops at its first collision.

### `at_fault_collision_rate`

This uses a second persistent binary flag set only when `is_at_fault_collision()` classifies the controlled agent as responsible:

```text
at_fault_collision_rate =
    (1 / N) * sum_i I(agent i had at least one at-fault collision)
```

The current at-fault heuristic works as follows:

- a controlled agent moving at or below 0.2 m/s is not at fault;
- a moving controlled agent colliding with an agent at or below 0.2 m/s is at fault;
- a collision sufficiently behind the controlled agent is not at fault;
- otherwise, intersection of the controlled agent's front bumper with the other vehicle is at fault;
- otherwise, having no selected lane or straddling beyond the multi-lane threshold is classified as at fault.

This is a simulator heuristic, not a legal or CARLA-standard fault determination.

### `red_light_violation_rate`

Each agent owns a persistent binary red-light flag:

```text
red_light_violation_rate =
    (1 / N) * sum_i I(agent i triggered a red-light violation)
```

The event requires traffic-control observations to be enabled. The check detects either:

- crossing an extended stop-line segment while its associated light is red and the agent is travelling in the controlled direction; or
- changing between lanes controlled by the relevant red light near its stop line.

Yellow is not treated as a violation in this call path. With `traffic_light_behavior: stop`, the first violation stops the agent.

### `total_infraction_count`

Despite the name, this does **not** count every infraction event. For each agent, collision, off-road, and red-light flags are combined with logical OR:

```text
agent_has_infraction_i =
    offroad_i OR collision_i OR red_light_i

total_infraction_count = sum_i I(agent_has_infraction_i)
```

Therefore:

- its unit is agents, not events;
- one agent with multiple infraction types contributes only 1;
- at-fault collision is not an additional count—it is already a subset of collision;
- the value lies between 0 and `n`.

### `total_distance_travelled_sum`

At each evaluated timestep, the agent adds:

```text
sim_speed * dt
```

to `distance_since_spawn`. The CSV row sums final distance over all controlled agents:

```text
total_distance_travelled_sum = sum_i(distance_since_spawn_i)
```

The unit is meters. Stopped or removed agents stop accumulating distance because their metric computation is skipped.

### `avg_distance_per_infraction`

For a CSV scenario row:

```text
avg_distance_per_infraction =
    total_distance_travelled_sum / max(total_infraction_count, 1)
```

The denominator is the number of agents that had any listed infraction, not the number of event occurrences.

Important zero-infraction caveat: when `total_infraction_count = 0`, the denominator is forced to 1. The result is therefore the scenario's total distance, not infinity, missing data, or a statistically estimated failure distance.

## Goal and completion metrics

### `num_goals_reached`

An agent increments its counter whenever its swept motion segment comes within the configured goal radius of its current waypoint at compatible elevation. The row reports:

```text
num_goals_reached = (1 / N) * sum_i(goal_count_reached_i)
```

This is an average number of goal waypoints per agent, so it is not bounded by 1. It can also exceed configured `num_goals`. In this run `num_goals = 3`, but after the active goal set is exhausted, the finite-mode route logic regenerates a new goal set and the cumulative reached counter continues increasing. That is why the example summary reports about 70.37 goals per agent.

### `score`

At the scenario boundary, each agent contributes 1 if:

```text
num_goals_reached_i >= env.num_goals
AND agent is not removed
AND agent is not stopped
```

The CSV field is:

```text
score = successful_agents / N
```

For this run, reaching at least three goals is enough to satisfy the goal-count part. However, an agent that reached three goals and was later stopped by an infraction does not count as successful.

This `score` is separate from `puffer_score`.

### `dnf_rate`

An agent is counted as did-not-finish only when all four conditions hold:

```text
no off-road event
AND no collision
AND no red-light violation
AND zero goals reached
```

Then:

```text
dnf_rate = DNF_agents / N
```

This is a narrow definition. An agent that reaches no goal but collides is counted in `collision_rate`, not `dnf_rate`. Therefore `dnf_rate` is not simply `1 - score`, and these fields do not form a clean exhaustive partition.

## Motion, lane, and comfort metrics

### `avg_speed_per_agent`

Each actively simulated agent accumulates its non-negative scalar `sim_speed`. At the scenario boundary:

```text
avg_speed_per_agent =
    (1 / N) * sum_i(sum_t sim_speed_i,t / T)
```

The unit is m/s. The implementation divides by the environment's full final timestep `T`, not by the number of timesteps during which a particular agent remained active. Thus, after an agent stops or is removed, its remaining implicit zero-speed timesteps lower its scenario average.

### `velocity_progress_sum`

Despite the name, this is neither a distance sum nor speed in m/s. At each evaluated timestep:

```text
if signed_speed > 2.5 m/s and a lane is selected:
    velocity_progress = max(cos(agent_heading - lane_heading), 0)
else:
    velocity_progress = 0
```

The CSV value is the average over full scenario timesteps and agents:

```text
velocity_progress_sum =
    (1 / N) * sum_i(sum_t velocity_progress_i,t / T)
```

It lies in `[0, 1]`. A value near 1 means agents spent most of the scenario moving forward faster than 2.5 m/s and closely aligned with their selected lanes. A stopped, slow, lane-less, perpendicular, or wrong-way agent contributes zero at those timesteps.

### `lane_center_rate`

At every evaluated timestep, the agent contributes 1 when the absolute signed distance from its center to the selected lane center is below 0.5 m:

```text
lane_center_rate =
    (1 / N) * sum_i(sum_t I(abs(lane_center_distance_i,t) < 0.5 m) / T)
```

This is effectively an agent-timestep proportion. As with the speed metric, stopped/removed remainder is included implicitly as zero because normalization uses the full scenario timestep.

### `comfort_violation_count`

At every evaluated timestep, the code forms three possible violation contributions:

```text
I(abs(longitudinal_acceleration) > 3 m/s^2)
+ I(abs(lateral_acceleration) > 3 m/s^2)
+ I(abs(longitudinal_jerk) > 5 m/s^3
    OR abs(lateral_jerk) > 5 m/s^3)
```

The instantaneous count is therefore 0, 1, 2, or 3. The CSV metric is:

```text
comfort_violation_count =
    (1 / N) * sum_i(sum_t instantaneous_violation_count_i,t / T)
```

It is the average number of violated comfort components per agent-step. It is not a raw count and not exactly the proportion of timesteps with any comfort violation; because acceleration components are counted separately, it can theoretically exceed 1.

## Puffer-score metrics

These columns are emitted only when `compute_eval_metrics: true`. They are computed per agent first and then averaged over the scenario's controlled agents.

### `progress_ratio`

For each agent:

```text
episode_duration_seconds_i = L_i * dt
reference_distance_i = max(10 m/s * episode_duration_seconds_i, 1 m)
progress_ratio_i = distance_since_spawn_i / reference_distance_i

progress_ratio = (1 / N) * sum_i(progress_ratio_i)
```

For a complete 600-second episode, the reference distance is 6,000 m. Because travelled distance is integrated speed, `progress_ratio` is approximately average speed divided by 10 m/s. A ratio of 0.5 corresponds roughly to 5 m/s average speed; a ratio of 1 corresponds to 10 m/s. The CSV field is not capped, although the contribution used inside `puffer_score` is capped at 1.

### `making_progress_rate`

Each agent passes the progress gate only if its final ratio is strictly greater than 0.2:

```text
making_progress_rate =
    (1 / N) * sum_i I(progress_ratio_i > 0.2)
```

Since the reference is 10 m/s, this threshold corresponds roughly to more than 2 m/s average progress over the logged episode.

### `driving_direction_score`

When an agent is moving forward faster than 2.5 m/s but its lane-heading cosine is negative, the code accumulates travelled distance as `wrong_way_distance`.

The final per-agent score is tiered:

```text
1.0 if wrong_way_distance <= 2 m
0.5 if 2 m < wrong_way_distance <= 6 m
0.0 if wrong_way_distance > 6 m
```

The CSV metric is the mean of those per-agent values. Consequently, a scenario-row value need not itself be one of the three tier values.

### `speed_limit_compliance`

For each agent and evaluated timestep, the implementation accumulates only the amount above the selected lane's speed limit:

```text
speed_violation_sum_i += max(sim_speed_i,t - speed_limit_i,t, 0) * dt
```

If no valid positive lane speed limit is available, the fallback target speed is 15 m/s. The final score is:

```text
speed_limit_compliance_i =
    max(0, 1 - speed_violation_sum_i / episode_duration_seconds)

speed_limit_compliance =
    (1 / N) * sum_i(speed_limit_compliance_i)
```

This is not the fraction of timesteps under the limit. Algebraically, the subtracted term is the time-average positive overspeed measured in m/s. An average excess of 0.2 m/s produces a score of 0.8; an average excess of at least 1 m/s clamps the score to zero.

There is a separate binary overspeed signal used by the reward code at `speed_limit + 2 m/s`; that binary threshold is not how this evaluation score is calculated.

### `multi_lane_time`

At each evaluated timestep, the code computes:

```text
vehicle_edge_distance = abs(lane_center_distance) + vehicle_width / 2
```

For agent `i` at timestep `t`, the timer increment is:

```text
delta_multi_lane_time_i,t =
    dt * I(vehicle_edge_distance_i,t > 2.05 m AND sim_speed_i,t > 0)
```

Here `I(condition)` is 1 when the condition is true and 0 otherwise. The per-agent value is the cumulative sum over all evaluated timesteps:

```text
multi_lane_time_i =
    sum_t dt * I(vehicle_edge_distance_i,t > 2.05 m AND sim_speed_i,t > 0)
```

With this evaluation's `dt = 0.1`, each qualifying timestep adds 0.1 seconds. Qualifying timesteps do not need to be consecutive: for example, 23 qualifying timesteps anywhere in the episode produce `multi_lane_time_i = 2.3` seconds.

The 2.05 m threshold is `3.7 / 2 + 0.2`, using a nominal 3.7 m lane width plus 0.2 m margin. The name is approximate: the code tests whether the vehicle footprint exceeds the selected lane's nominal envelope; it does not explicitly confirm occupancy of a second identified lane.

The CSV value is then the mean cumulative time over agents:

```text
multi_lane_time = (1 / N) * sum_i(multi_lane_time_i)
```

Its unit is average seconds per agent. It is not a fraction of episode time and is not the total accumulated time across all agents.

### `multi_lane_score`

Each agent's final tier is:

```text
1.0 if multi_lane_time <= 3.4 seconds
0.5 if 3.4 < multi_lane_time <= 5.7 seconds
0.0 if multi_lane_time > 5.7 seconds
```

The CSV field is the mean tier score over agents.

### `comfort_score`

> **BROKEN METRIC:** `comfort_score` is not correctly calculated in the current implementation. It is still emitted and used by `puffer_score`, but it does not measure the agents' actual comfort.

The intended formula is:

```text
duration_seconds = max(L_i * dt, dt)
windows = max(ceil(duration_seconds / 10 seconds), 1)
comfort_score_i = clamp(1 - comfort_violation_timestep_count_i / windows, 0, 1)
```

The defect is that `comfort_violation_timestep_count` is declared and read but never incremented anywhere in the current implementation. The separately working `comfort_violation_count` accumulator does not feed this formula.

Current effective behavior is therefore:

```text
comfort_violation_timestep_count_i = 0
comfort_score_i = 1
comfort_score = 1
```

This is confirmed by all 1,000 rows of the example evaluation: every row has `comfort_score = 1.0`, and the summary mean is exactly `1.0`. Until the accumulator is correctly defined and updated:

- do not use `comfort_score` to compare driving comfort;
- do not interpret a value of 1 as evidence of comfortable driving;
- use the separately computed `comfort_violation_count` when inspecting current comfort behavior;
- treat `puffer_score` as a score whose comfort component has been replaced by a constant rather than a measured quantity.

### `puffer_score`

The score is calculated separately for every controlled agent and then averaged:

```text
puffer_score = (1 / N) * sum_i(puffer_score_i)
```

The per-agent score has a multiplicative gate and a weighted continuous component.

#### Multiplicative gate

```text
no_at_fault_i = 1 if no at-fault collision, else 0
no_offroad_i = 1 if no off-road event, else 0
no_red_light_i = 1 if no red-light event, else 0
making_progress_i = 1 if progress_ratio_i > 0.2, else 0
direction_i = 1, 0.5, or 0 from wrong-way distance

multiplier_i =
    no_at_fault_i
    * no_offroad_i
    * no_red_light_i
    * making_progress_i
    * direction_i
```

Notice that any collision does not automatically zero the score: the collision must be classified as at fault. Off-road and red-light events always zero it. Wrong-way distance between 2 and 6 m halves it; more than 6 m zeros it.

#### Weighted component

```text
weighted_average_i = (
    5 * ttc_score_i
    + 5 * min(progress_ratio_i, 1)
    + 4 * speed_limit_compliance_i
    + 3 * multi_lane_score_i
    + 2 * comfort_score_i
) / 19

puffer_score_i = multiplier_i * weighted_average_i
```

#### Consequence of the broken `comfort_score`

Because the current implementation always supplies `comfort_score_i = 1`, the comfort term always contributes `2 / 19` to the weighted average before application of the multiplier. The erroneous contribution to an agent's final score is:

```text
broken_comfort_contribution_i = multiplier_i * (2 / 19)
```

| Agent multiplier | Contribution to final `puffer_score_i` from the broken comfort term |
|---:|---:|
| `0` | `0` |
| `0.5` | approximately `0.0526` |
| `1` | approximately `0.1053` |

Thus the comfort value is the same for every agent, but its absolute contribution to the final score is not: safety/progress gates scale it through `multiplier_i`. A fully gate-passing agent receives an unearned `0.1053` score contribution even if its acceleration and jerk are uncomfortable; a zero-multiplier agent receives no inflation.

Evaluations produced by the same broken implementation are still internally comparable under the restricted interpretation that comfort is a constant bonus rather than a measured dimension. They cannot reveal comfort improvements or regressions, and their `puffer_score` values are not directly comparable to results produced after this defect is corrected.

`ttc_score` is not emitted as a CSV column, but it affects `puffer_score`. On every evaluated timestep, the simulator computes minimum vehicle time-to-collision (TTC), forces TTC to zero on a collision timestep, and counts a violation when TTC is below 0.95 seconds:

```text
ttc_score_i = 1 - ttc_violation_timesteps_i / ttc_sample_timesteps_i
```

Because `comfort_score` is currently fixed at 1, every agent presently receives the full `2/19` comfort contribution before multiplication by the safety/progress gates.

## How `evaluation_summary.json` is computed

The CSV rows are collected into a Pandas dataframe. `seed` is explicitly removed from numeric aggregation. For nearly every remaining numeric column, the summary uses the arithmetic mean over emitted scenario rows:

```text
metrics_mean[metric] =
    sum_scenario_rows(metric) / num_episodes
```

This is a **scenario-weighted mean**, not a direct agent-weighted mean across the whole benchmark. In this CARLA run every scenario has 50 agents, so those two interpretations coincide for per-agent metrics. They would differ if scenario agent counts varied.

The JSON top-level counters are:

- `num_scenarios`: the requested benchmark count, here 1,000;
- `num_episodes`: the number of episode rows actually emitted, here 1,000.

These should match in a successful complete benchmark. If they do not, the means describe only the emitted rows.

### Special handling of distance and infractions

Two CSV fields are not averaged into same-named JSON fields:

```text
metrics_mean.total_distance_travelled =
    sum_rows(total_distance_travelled_sum)

metrics_mean.total_infractions =
    sum_rows(total_infraction_count)

metrics_mean.avg_distance_per_infraction =
    metrics_mean.total_distance_travelled
    / max(metrics_mean.total_infractions, 1)
```

Therefore:

- CSV `total_distance_travelled_sum` is a per-scenario total;
- JSON `total_distance_travelled` is the total over the entire evaluation;
- CSV `total_infraction_count` is a per-scenario count of affected agents;
- JSON `total_infractions` is the sum over all scenarios;
- JSON `avg_distance_per_infraction` is the global ratio of total distance to total affected-agent count, not the arithmetic mean of the CSV row ratios.

For the example evaluation:

```text
total_distance_travelled = 144,200,500.484375 m
total_infractions         = 6,041.000023841858 affected-agent counts
avg_distance_per_infraction ~= 23,870.303 m
```

The non-integer-looking global infraction value is floating-point accumulation noise; the underlying per-scenario counts represent integer agents.

## Worked interpretation of one example row

The first Town01 row begins with approximately:

```text
n                              = 50
offroad_rate                   = 0.02
collision_rate                 = 0.02
red_light_violation_rate       = 0.04
num_goals_reached              = 101.84
score                          = 0.92
avg_speed_per_agent            = 5.031
total_infraction_count         = 4
at_fault_collision_rate        = 0.02
puffer_score                   = 0.787
progress_ratio                 = 0.503
```

This means:

- 1 of 50 agents went off-road at least once;
- 1 of 50 agents collided at least once;
- 2 of 50 agents violated a red light at least once;
- 4 distinct agents had at least one of those three infraction types, so there was no overlap among those affected agents in this row;
- the agents reached 101.84 goal waypoints on average;
- 46 of 50 agents reached at least three goals and ended neither stopped nor removed;
- mean speed over the complete scenario was about 5.03 m/s;
- the mean progress ratio of 0.503 is consistent with about 5.03 m/s relative to the 10 m/s reference;
- 1 of 50 agents had an at-fault collision;
- after the gates and weighted components were evaluated per agent, their mean `puffer_score` was about 0.787.

## Interpretation cautions

1. A field ending in `_rate` does not guarantee an agent-timestep denominator. Infraction rates are ever-event agent fractions; lane-center rate is an agent-timestep proportion.
2. `total_infraction_count` counts affected agents, not events.
3. `avg_distance_per_infraction` uses a denominator floor of 1, so a zero-infraction scenario reports total distance rather than an infinite or missing value.
4. `score` and `puffer_score` are different metrics.
5. `num_goals_reached` can greatly exceed configured `num_goals` because goal sets regenerate.
6. `speed_limit_compliance` is based on average amount of overspeed, not percent of time compliant.
7. `multi_lane_time` is seconds, while `multi_lane_score` is a tiered per-agent score.
8. **`comfort_score` is broken:** `comfort_violation_count` works, but the separate accumulator read by `comfort_score` is never updated. This also gives fully gate-passing agents an unearned `2/19`, or approximately `0.1053`, contribution to `puffer_score`.
9. JSON ordinary means weight scenarios equally. This matters whenever scenarios have different controlled-agent counts or durations.

## Source trail

The primary implementation points are:

- [`pufferlib/ocean/drive/drive.h`](../../pufferlib/ocean/drive/drive.h)
  - `struct Log`: raw accumulators;
  - `compute_metrics()`: geometry, motion, comfort, infraction, TTC, and goal signals;
  - `compute_rewards()`: persistent event flags and timestep accumulators;
  - `calculate_duration_scaled_violation_score()`: intended comfort score;
  - `calculate_puffer_score()`: score gates and weighted formula;
  - `add_log()`: per-agent to per-scenario aggregation.
- [`pufferlib/ocean/drive/binding.c`](../../pufferlib/ocean/drive/binding.c)
  - `my_episode_to_dict()`: divides accumulated fields by the scenario's `n`;
  - `my_log()`: names the CSV metrics and reconstructs raw scenario totals.
- [`pufferlib/ocean/env_binding.h`](../../pufferlib/ocean/env_binding.h)
  - `vec_per_episode_log()`: emits one row per completed frozen evaluation environment.
- [`pufferlib/ocean/evaluation_utils/evaluation_utils.py`](../../pufferlib/ocean/evaluation_utils/evaluation_utils.py)
  - `_build_eval_report()`: creates the dataframe and summary;
  - `_write_eval_reports()`: writes CSV and JSON.
- [`pufferlib/utils.py`](../../pufferlib/utils.py)
  - `reduce_environment_metrics()`: ordinary means and special global distance/infraction reduction.
- [`pufferlib/ocean/drive/constants.h`](../../pufferlib/ocean/drive/constants.h)
  - thresholds used by lane alignment, TTC, multi-lane occupancy, duration windows, and reference progress.
- [`project/second_run_nightly_best_config/evaluation_benchmarks.yaml`](../second_run_nightly_best_config/evaluation_benchmarks.yaml)
  - benchmark-level behaviors and evaluation overrides.
- [`docs/evaluation.md`](../../docs/evaluation.md)
  - evaluator launch and output layout.
