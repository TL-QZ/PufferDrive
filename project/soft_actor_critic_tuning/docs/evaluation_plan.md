# SAC16 final CARLA evaluation and analysis plan

Evaluate the 16 final actors from `sac16_grid_20261012T032408Z` using the
baseline's full CARLA protocol, then compare the four tuning parameters.
Town comparisons are excluded. Preserve the existing four metric groups in
every plotting view.

Execution rule: the user launches real evaluations and analysis. Agent
verification is limited to dry runs and temporary CPU test fixtures.

Status: evaluation and analysis scripts are prepared. CPU dry-run and synthetic
analysis checks passed; real evaluation remains pending the user's launch.
All 16 checkpoints exist and training completed. The current saved evaluations
are `carla_fast`; full final CARLA evaluation is still needed.

## 1. Trial identity and grouping

Read the [saved sweep manifest](../runs/sac16_grid_20261012T032408Z/manifest.json)
and cross-check each trial's saved learner configuration. Keep existing trial
IDs and checkpoint paths. Use trial IDs and explicit parameter values in
legends, titles and exported tables.

| Update epochs | Replay refresh | Initial alpha 0.1, tau 0.005 | Initial alpha 0.1, tau 0.01 | Initial alpha 0.2, tau 0.005 | Initial alpha 0.2, tau 0.01 |
| --- | --- | --- | --- | --- | --- |
| 3 | 5% | 01 | 02 | 03 | 04 |
| 3 | 10% | 05 | 06 | 07 | 08 |
| 10 | 5% | 09 | 10 | 11 | 12 |
| 10 | 10% | 13 | 14 | 15 | 16 |

Export `trial_index.csv` with trial ID, all four parameter values, training
seed, actual training duration, collected agent steps, learner updates,
checkpoint path and the explicitly selected final evaluation report paths.
Derive identities from the saved run, rather than today's editable configs.

## 2. Full CARLA evaluation

Adapt the existing [baseline evaluator](../../baseline_run_sync_2026-08-24/eval/evaluate_final_model.sh)
in a thin experiment-local launcher under `scripts/`. Use the existing SAC
evaluation entry point and shared evaluator. No learner or simulator changes
are planned.

Use the baseline's [benchmark configuration](../../baseline_run_sync_2026-08-24/override_config/evaluation_benchmarks.yaml),
selecting only `carla`: 1,000 scenarios across eight towns, benchmark seed 42,
6,000 steps per scenario, `dt=0.1`, Gigaflow and mean action selection.
Match the recorded baseline evaluation execution settings: one GPU per
checkpoint, BF16 inference, compilation disabled for this final evaluation,
20 multiprocessing workers, and `eval.num_agents=300` with at most 50 active
agents per scenario. Training compilation remains enabled in the sweep config.

Set these evaluation options explicitly: SAC's standalone evaluation defaults
do not inherit all execution and benchmark settings from the training config.
Use metrics-only evaluation. Retain `episode_metrics.csv`,
`evaluation_summary.json` and `resolved_benchmark.yaml` under each trial's
`learner/eval/carla_final_model_mean_metrics/<timestamp>/` directory. Reuse the
baseline W&B reporting to the corresponding training run's `final_eval_carla/`
summary fields.

Evaluate one checkpoint first to validate the protocol and measure runtime,
then queue the remaining trials. Start with two concurrent evaluations, each
using one GPU and 20 workers; expose GPU selection and concurrency in the
launcher. Record exact output paths so analysis never silently chooses the
newest evaluation or mixes evaluation sessions.

## 3. Analysis code and metric groups

Put new plotting and analysis code in the existing `analysis/` folder, with
`analysis/compare_carla.py` as the planned entry point. Reuse definitions,
units and plotting conventions from
[plot_benchmark_comparison.py](../../metric_analysis/plot_benchmark_comparison.py).
Existing shared plotters remain the reference; their three-model/seed-label
interface does not directly represent this 16-configuration experiment.

Preserve these four groups and their existing metric membership:

| Output basename | Metric family |
| --- | --- |
| `infraction_metrics` | Infractions and distance per infraction |
| `goal_completion_metrics` | Goal completion, score, DNF and controlled-agent count |
| `motion_lane_comfort_metrics` | Speed, motion, lane centering and comfort violations |
| `puffer_score_metrics` | Puffer score and its component metrics |

Read aggregate metrics from the summary JSONs. Preserve the existing
`infractions_per_scenario = total_infractions / num_scenarios` calculation and
the evaluator's global distance-per-infraction ratio. Use the scenario CSVs
to verify coverage and matching scenario identities. Do not produce per-town
plots. Keep units, metric-specific scales and improvement directions visible;
retain the known-broken annotation for `comfort_score`.

## 4. Comparison views and named plot output

Each view produces one figure per metric group, with independently scaled
metric subplots:

| View | Layout and purpose |
| --- | --- |
| Overview | A 4-by-4 grid per metric: rows group epochs and initial alpha; columns group refresh ratio and tau. Label cells with trial IDs and metric values. The four blocks correspond to epochs × refresh ratio. |
| Parameter effects | For each of the four parameters, show eight matched high-minus-low comparisons while holding the other three settings fixed, plus their mean and a zero reference. Label each pair with its background settings. |
| Alpha–tau interactions | Show the two alpha values across the two tau values, separated by epochs and refresh ratio, to expose changes in their combined effect. |

Examples of isolated comparisons are 01–02 for tau, 01–03 for initial alpha,
01–05 for refresh ratio and 01–09 for epochs. Generate all matched pairs from
the trial table rather than maintaining handwritten pair lists.

Write generated plots and analysis tables under a named subfolder of `plots/`.
The planned default name identifies the sweep and benchmark; allow an explicit
output name for later analyses:

```text
plots/sac16_grid_20261012T032408Z_carla_final/
  trial_index.csv
  aggregate_metrics.csv
  parameter_effects.csv
  overview/                         # four metric-group figures
  parameter_effects/<parameter>/     # four figures for each parameter
  interactions/                     # four metric-group figures
```

All trials use training seed 0; evaluation seed 42 identifies benchmark
scenarios, not additional training replicates. Pair averages describe effects
within this single-seed grid. These checkpoints have roughly equal wall-time
budgets but approximately 27–73 million collected agent rows; include actual
time and steps when interpreting endpoint performance. Initial alpha is a
starting value, since temperature is subsequently learned.

The existing fast evaluations have zero `score` and `puffer_score` for all 16
trials. The full benchmark may differ. Retain component metrics so comparisons
remain informative when a composite metric is constant.

## 5. Completion checks

Before plotting, require all 16 explicitly selected full CARLA summaries to
contain 1,000 scenarios, 1,000 completed episodes and the required finite
metrics. Check matching benchmark settings and scenario identities, and verify
that trial labels agree with the saved parameter values.

Verify the four metric-group figures in every view, inspect representative
outputs visually, and check a matched pair against its source metrics. Keep
checkpoint files and existing evaluation reports unchanged. The user starts
evaluation and plot generation using the [launch instructions](how_to_evaluate.md).
