# Condition B — W&B metric guide

Scope: metric sections emitted by the current distillation trainer; excludes `system`. Names below are the logged keys, regardless of how a W&B workspace arranges its panels. Defaults refer to [condition_b.yaml](../config/condition_b.yaml), checked 2026-09-23. Run overrides take precedence.

## Dashboard map

| Section | Question it answers | Default two-GPU logging schedule |
|---|---|---|
| `train` | Is the student fitting the training windows? | Update 1, then every 10 optimizer updates; final pending losses flushed |
| `validation` | Does it fit held-out teacher trajectories? | Every 50 optimizer updates and at completion |
| `representation` | Are latents collapsing or policy distributions changing? | Every 100 optimizer updates |
| `eval` | Can the student actually drive? | After 50 reuse epochs: once after training on each collection; final evaluation if needed |
| `collection` | How much fresh simulator data was collected? | After each collection |
| `data` | How many valid windows are available and reused? | After each completed reuse epoch |
| `throughput` | How quickly are training windows processed? | Alongside regular training logs |
| `progress` / `event_index` | Where is this measurement in the run? | Every logged event |

**Start with driving performance in `eval`, then held-out KL in `validation`.** Lower training loss alone does not establish better driving.

## Axes: `progress` and `event_index`

| Full key | Meaning |
|---|---|
| `progress/optimizer_step` | Completed optimizer updates; accumulation microbatches do not increment it |
| `progress/simulator_transitions` | Cumulative training agent-transitions across both GPUs; excludes validation/evaluation and reuse |
| `progress/collection_round_idx` | Zero-based collection index |
| `progress/update_epoch_idx` | Zero-based reuse-epoch index within a collection |
| `event_index` | Monotonic logging-event counter, also supplied as W&B's logging step |

Choose `progress/optimizer_step` for learning curves and `progress/simulator_transitions` for sample-efficiency comparisons. Several events can share one optimizer step. Transition count stays flat while a collection is reused. Final events can show the next sampler cursor rather than the last completed epoch.

## `train`: optimization

| Key after `train/` | Meaning | How to read it |
|---|---|---|
| `loss_total` | Weighted sum of the three component losses | Lower is better fitting; inspect components too |
| `loss_distillation` | Soft cross-entropy against teacher action distributions, averaged over four chunk slots; includes temperature-squared scaling | Lower is better; minimum is not generally zero because teacher entropy remains |
| `loss_jepa` | Squared L2 distance between normalized predicted endpoint latent and EMA-target endpoint latent | Lower means better latent prediction conditioned on logged actions; approximately 0–4 |
| `loss_variance` | Mean deficit of latent standard deviation below the configured floor | Lower means fewer collapsed dimensions; not a driving score |
| `gradient_norm` | Global parameter-gradient L2 norm before clipping, from the latest update | Compare with clipping threshold 1.0; spikes indicate optimization instability |
| `learning_rate` | Current optimizer learning rate | Default constant `0.0001` |

Default: `loss_total = loss_distillation + loss_jepa + 0.1 * loss_variance`. Component charts show **unweighted** losses. Training losses are window-weighted means across both GPUs and the logging interval; gradient norm and learning rate are latest-update values. Variance statistics are computed within each local microbatch, not over the global optimizer batch.

## `validation`: held-out trajectory fit

Uses a fixed held-out teacher-driven collection with a separate seed. These measurements do **not** involve the student driving. Values are window-weighted over validation batches.

| Key after `validation/` | Meaning | Desired behavior |
|---|---|---|
| `loss_total` | Same weighted objective as training | Falls along with training loss; a widening gap suggests overfitting |
| `loss_distillation` | Held-out soft cross-entropy | Lower |
| `loss_jepa` | Held-out normalized endpoint prediction error | Lower |
| `loss_variance` | Held-out latent variance penalty | Lower, without sacrificing driving performance |
| `teacher_kl/slot_0` | KL(teacher || student) for the action at time t | Lower; 0 means matching distributions |
| `teacher_kl/slot_1` | Same KL for time t+1 | Lower |
| `teacher_kl/slot_2` | Same KL for time t+2 | Lower |
| `teacher_kl/slot_3` | Same KL for time t+3 | Lower |
| `teacher_kl/mean` | Mean KL across all four slots | Main distribution-matching diagnostic |
| `latent_std` | Mean per-dimension context-latent standard deviation, calculated within validation batches | Near zero suggests collapse |
| `latent_norm` | Mean L2 norm of context latents | Watch scale drift; larger is not inherently better |

KL uses untempered probabilities and natural logarithms (nats), even if the loss temperature is overridden. With the default execution horizon of 1, only slot 0 is executed before replanning; later slots remain training targets.

## `representation`: training-batch diagnostics

Computed on at most **128 windows from rank 0**, not the whole dataset or both GPUs. These are snapshots, not interval averages.

| Key after `representation/` | Meaning | How to read it |
|---|---|---|
| `latent_std_mean` | Average standard deviation across latent dimensions | Near zero suggests widespread collapse |
| `latent_std_min` | Smallest per-dimension standard deviation | Near zero flags at least one inactive dimension |
| `latent_norm_mean` | Average latent L2 norm | Track scale drift |
| `effective_rank` | Exponential entropy of normalized singular values of centered latents | Low suggests limited representation diversity; at most 127 for 128 centered samples, despite 1,024 latent dimensions |
| `student_entropy` | Average categorical entropy across student action slots, in nats | Low means sharper predictions; high means more diffuse predictions |
| `teacher_entropy` | Same entropy for teacher labels on those windows | Reference for student uncertainty, not a target to minimize |

Entropy uses temperature 1. With 12 classes, the maximum is `ln(12) ≈ 2.485`. Entropy matching alone does not imply matching action distributions.

## `collection`, `data`, and `throughput`

| Full key | Meaning / units |
|---|---|
| `collection/seconds` | Rank-0 elapsed collection phase, including waiting for rank synchronization; seconds |
| `collection/transitions` | Fresh agent-transitions collected in this round across both GPUs |
| `collection/rank_transitions` | Fresh agent-transitions collected by rank 0; not the global total |
| `data/global_windows` | Valid training windows in the current pooled collection across ranks; excludes windows crossing invalid boundaries |
| `data/rank_windows` | Windows assigned to each rank for one reuse epoch, including any balancing repeats |
| `data/repeated_windows` | Extra windows added globally to balance rank lengths; **not** the 50 reuse epochs |
| `data/rank_repeated_windows` | Current implementation reports global repeats integer-divided by world size; not an exact per-rank duplicate count |
| `throughput/training_windows_per_second` | Global processed windows divided by wall time since the previous training log; includes repeated training passes |

Throughput can include intervening validation, evaluation, collection and checkpoint overhead; it is not pure GPU compute speed or fresh simulator transitions/second. Old collections remain on disk, but `data/global_windows` describes the current collection pool, not all retained data.

## `eval`: student driving

Full default prefix: `eval/training_native/carla_fast/student/`. Older toy runs use `carla` instead of `carla_fast`. The prefix encodes output name, benchmark, and policy role. The current training path logs student evaluation; the monitor also supports a teacher role, but training does not automatically produce teacher curves.

Default evaluation: 250 scenarios, 500 simulator steps, seed 42, 20 environments on rank 0, student self-play, mean actions, execute one action then replan. These are simulator results, separate from held-out loss validation.

Keys below follow the prefix above. Rates are fractions, not percentages. Most values are averaged across episode rows after per-agent normalization within each episode; total distance and total infractions are summed instead.

### Driving quality

| Metric suffix | Meaning / interpretation |
|---|---|
| `puffer_score` | Composite driving score gated by safety, progress and driving direction; higher is better |
| `score` | Fraction reaching all goals without removal or stopping; higher is better |
| `episode_return` | Mean cumulative reward per agent episode; higher is better for the configured reward |
| `num_goals_reached` | Mean goals reached per agent; higher is generally better |
| `dnf_rate` | Fraction with no offroad/collision/red-light flag **and** fewer than one goal; lower is better. This is the code's specific non-completion proxy |
| `collision_rate` | Fraction of agents with any collision; lower is better |
| `at_fault_collision_rate` | Fraction with an at-fault collision; lower is better |
| `offroad_rate` | Fraction with any offroad event; lower is better |
| `red_light_violation_rate` | Fraction with any red-light violation; lower is better |
| `total_infractions` | Total agents with at least one offroad/collision/red-light flag; **not** number of individual events |
| `total_distance_travelled` | Total agent distance across episodes, meters; compare using the same evaluation budget |
| `avg_distance_per_infraction` | Total distance / max(total infractions, 1), meters; higher is generally better. With zero infractions it equals total distance |
| `avg_speed_per_agent` | Mean time-averaged speed, m/s; faster is not automatically better |
| `velocity_progress_sum` | Mean per-timestep forward-progress metric; despite its name, this logged value is normalized |
| `progress_ratio` | Agent distance / reference distance; higher means more progress |
| `making_progress_rate` | Fraction with progress ratio above 0.2; higher is better |
| `lane_center_rate` | Mean fraction of timesteps within 0.5 m of lane center; higher is better |
| `driving_direction_score` | Score 1 for wrong-way distance ≤2 m, 0.5 for ≤6 m, otherwise 0; higher is better |
| `speed_limit_compliance` | Speed-compliance score, clipped at zero; higher is better |
| `comfort_violation_count` | Mean violations per agent per timestep; **not** a raw count; lower is better |
| `comfort_score` | Duration-scaled comfort score in [0, 1]; higher is better |
| `multi_lane_time` | Mean time exceeding the multi-lane threshold while moving, seconds; lower is better |
| `multi_lane_score` | Tiered score 1, 0.5 or 0 based on multi-lane time; higher is better |

### Episode and population counters

| Metric suffix | Meaning |
|---|---|
| `active_agent_count` | Active agents in a Drive environment; averaged across episode rows |
| `agents_per_batch` | Agent count reported for the evaluation batch; metadata |
| `n` | Agents contributing to an environment's episode log; averaged across episode rows |
| `episode_timestep` | Environment timestep at episode end; simulator steps |
| `episode_length` | Mean episode length per agent, steps; interpret alongside completion and early termination |

### Reward components

These are cumulative contributions per agent episode, then averaged across episodes; units are reward points. Penalties are normally negative. They explain `episode_return`; they are not additional student-training losses.

| Metric suffix | Contribution |
|---|---|
| `reward_components/collision` | Collision penalty |
| `reward_components/offroad` | Offroad penalty |
| `reward_components/red_light` | Red-light penalty |
| `reward_components/goal` | Goal reward |
| `reward_components/lane_align` | Lane-heading alignment reward |
| `reward_components/lane_center` | Lane-centering reward |
| `reward_components/comfort` | Comfort penalty |
| `reward_components/velocity` | Forward-progress reward |
| `reward_components/timestep` | Timestep penalty |
| `reward_components/reverse` | Reverse-motion penalty |
| `reward_components/overspeed` | Overspeed penalty |
| `reward_components/ade` | Average-displacement-error reward term; sign depends on its coefficient |

The normal benchmark path forwards `summary.metrics_mean` only. Separate summary fields `num_scenarios` and `num_episodes` are not automatically W&B charts. A fallback evaluator can additionally emit `num_scenarios` and `num_timesteps`; these describe evaluation volume, not driving quality.

Driving definitions: `my_log` / `my_episode_to_dict` in `pufferlib/ocean/drive/binding.c`, `add_log` in `pufferlib/ocean/drive/drive.h`, and `reduce_environment_metrics` in `pufferlib/utils.py`.


## Sources and local inspection

| Source | Owns |
|---|---|
| [monitoring.py](../monitoring.py) | W&B namespaces, event counter, progress fields |
| [distributed_train.py](../distributed_train.py) | Two-GPU aggregation and logging schedule |
| [train.py](../train.py) | Validation and representation diagnostics; legacy single-GPU path |
| [model.py](../model.py) | Loss definitions |
| [evaluate.py](../evaluate.py) | Driving-evaluation dispatch and metric forwarding |

The same events are saved locally at `experiments/jepa_distill/runs/<run_id>/metrics.jsonl`. This guide describes emitted metrics and checks their names against a completed toy run; it does not assume a particular saved W&B panel layout. Evaluation fields are forwarded from the benchmark summary, so benchmark changes can add fields.
