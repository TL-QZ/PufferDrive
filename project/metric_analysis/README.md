# CARLA metric analysis

`plot_carla_metrics.py` compares one to three trained model seeds on the CARLA
evaluation. It produces one figure for each metric group, with an independently
scaled subplot for every metric. Each town has one box per supplied model seed;
a standard 1,000-scenario evaluation contributes 125 results per town.

From the repository root, supply one to three CSVs in model-seed order:

```bash
source .venv/bin/activate
python project/metric_analysis/plot_carla_metrics.py \
    --carla-csv path/to/episode_metrics.csv \
    --model-seed 0 \
    --output-dir path/to/output
```

If `--output-dir` is omitted, outputs are written to
`project/metric_analysis/output/carla`:

- `infraction_metrics.png`
- `goal_completion_metrics.png`
- `motion_lane_comfort_metrics.png`
- `puffer_score_metrics.png`

## Seed labels

The plot legend uses `--model-seed`. The `seed` column inside each CSV is the
per-scenario simulation seed, so it is not used as the legend grouping variable.

## Metric selection

The metric groups are explicit near the top of the script so they are easy to
extend. `total_distance_travelled_sum` and all `reward_components/*` columns
are intentionally excluded. `comfort_score` is plotted for completeness but
annotated as broken because its current value is always 1.

## CARLA and nuPlan benchmark comparison

`plot_benchmark_comparison.py` compares CARLA, nuPlan single, and nuPlan multi
across one to three trained model seeds. It reads the aggregate
`metrics_mean` values from each run's `evaluation_summary.json`. Each benchmark
has one bar per model seed, with compact value labels.

Pass one summary per benchmark. List multiple paths in seed order when comparing
two or three seeds:

```bash
python project/metric_analysis/plot_benchmark_comparison.py \
    --carla-json path/to/carla/evaluation_summary.json \
    --nuplan-single-json path/to/nuplan_single/evaluation_summary.json \
    --nuplan-multi-json path/to/nuplan_multi/evaluation_summary.json \
    --model-seed 0 \
    --output-dir path/to/output
```

The goal and completion figure also plots `n`, the mean number of controlled
agents per scenario. It is 50 for CARLA, 1 for nuPlan single, and 16.547 for
nuPlan multi in these evaluation summaries.

The JSON's `total_infractions` is an evaluation-wide total. The comparison
normalizes it by `num_scenarios` and plots the result as
`infractions_per_scenario`. The other plotted values come directly from
`metrics_mean`.

Metrics are shown in raw units without duration or population normalization.
CARLA episodes last 600 seconds, while nuPlan episodes last 20 seconds, so use
caution when comparing duration-dependent metrics such as
`num_goals_reached` and `multi_lane_time` across benchmarks.

## Before and after nuPlan fine-tuning

`plot_finetune_comparison.py` compares one original CARLA-trained model with
the corresponding model after nuPlan-single fine-tuning. It uses the same
three final evaluation benchmarks and four aggregate metric figures as the
benchmark comparison above, but each benchmark has two bars: before and after
fine-tuning.

The default outputs for seed 0 are written to
`project/metric_analysis/output/finetune_comparison/seed0`. A different output
directory can be supplied with `--output-dir`.

Supply the six `--before-<benchmark>-json` and `--after-<benchmark>-json`
options; use `--help` for the complete command. The baseline experiment wrapper
in `project/baseline_run_sync_2026-08-24/plot` fills these paths automatically.
