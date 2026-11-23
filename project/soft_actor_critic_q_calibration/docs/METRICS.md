# Per-rollout evaluation metrics

New collections save all 40 numeric fields from the existing `eval_carla_fast` evaluation's `metrics_mean` schema. Each branch JSON has its own `metrics_mean` dictionary alongside Q estimates, critic targets, returns, and termination flags. This dictionary describes one probing agent and one continuation, not an average over all agents or repeats.

## Measurement interval

The baseline is captured after reconstructing the selected initial state, immediately before the forced action. Measurement ends on the probing agent's first terminal, truncation, or validity gap. A completed branch's measurements are frozen even when other copies in its vector batch continue.

The private native binding snapshots counters without resetting simulation state. Additive counters use endpoint minus baseline. Rates and nonlinear scores are recomputed for that interval, using the native scoring functions and field serializer. Existing policy inputs, rewards, dynamics, and termination settings are retained. `compute_eval_metrics` is enabled for both reference and branch collection so the optional evaluation counters exist; the original and resolved configurations are recorded in the manifest.

## Interpretation

| Field | Meaning for a single branch |
|---|---|
| `num_goals_reached` | Goals reached during the measured continuation; excludes reference-prefix goals. |
| `total_distance_travelled` | Native accumulated speed × dt during the interval, not straight-line displacement. |
| `episode_return` | Undiscounted native reward sum for the interval. Existing discounted and soft returns remain separate. |
| `score` | 1 if interval goal count reaches configured `num_goals` and the agent ends neither removed nor stopped; otherwise 0. Prefix goals do not count toward this interval criterion. |
| `total_infractions` | 1 if any collision, off-road, or red-light event occurs in the interval; otherwise 0. It is not a count of separate incidents. |
| `avg_distance_per_infraction` | Interval distance / max(1, interval infraction indicator), using the native convention. |
| `puffer_score` and component scores | Recomputed from interval progress, safety events, TTC, speed, comfort, lane, and direction measurements—not final score minus initial score. |
| `episode_length`, `episode_timestep` | Number of measured simulation steps. |
| `n` | 1: this record represents one probing agent. |
| `active_agent_count`, `agents_per_batch` | Scene and vector-batch context counts, not the probing agent's outcomes. |

Reward-component names remain `reward_components/<component>`. Safety rates become single-rollout indicators; averaging them across continuations gives empirical event frequencies. Normalized speed, comfort-count, velocity-progress, and lane-center fields use the measured interval's step count.

Censored rollouts remain partial observations. A score at a scene cutoff is an observed-window score, not evidence of eventual success or failure. The existing horizon, terminal, truncation, and censoring fields must accompany comparisons.

## Files

- `records/probes/<probe>.json`: each `branches[]` entry contains `metrics_mean`; probe-level `metrics_scope` documents the interval and context fields.
- `records/probes/<probe>.npz`: one branch-aligned array per field, named `eval_metrics__<field>` with `/` replaced by `__`.
- `records/action_summary.csv`: additional `eval_metrics/<field>_mean`, `_sem`, and `_count` columns summarize repeated continuations for each initial state and action.

Existing completed runs are not rewritten or filled with synthetic zeros. Their missing metrics remain unavailable. New collection uses the expanded schema; existing plotting commands are unchanged so metrics can be selected later.

## Initial state of the probing car

Each new probe saves the probing car’s pre-action policy input as the one-dimensional float32 NPZ array `initial_ego_observation`. The collector copies it after replaying the reference prefix and verifying the observation fingerprint, before applying any forced action. The probe ID links this single vector to all actions and continuation seeds sharing that initial condition.

For grouping rollouts, the matching JSON contains `probe.initial_ego_geometry`. Its distances are horizontal distances in meters from the ego center. Partner distance is center-to-center. Lane and road-boundary distances use the closest point on the observed finite segment. Goal distances distinguish the next ordered waypoint from the nearest remaining waypoint represented in the goal slots. A missing object has a null distance. The observation range, available slots, and road dropout determine which objects the policy sees. The run manifest contains the resolved observation configuration.
