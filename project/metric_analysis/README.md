# CARLA metric analysis

`plot_carla_metrics.py` compares the three trained model seeds on the CARLA
evaluation. It produces one figure for each metric group, with an independently
scaled subplot for every metric. Each town has three side-by-side box plots,
one per model seed, and each box contains 125 scenario results.

From the repository root, run:

```bash
source .venv/bin/activate
python project/metric_analysis/plot_carla_metrics.py
```

The default outputs are written to `project/metric_analysis/output/carla`:

- `infraction_metrics.png`
- `goal_completion_metrics.png`
- `motion_lane_comfort_metrics.png`
- `puffer_score_metrics.png`

Use a different output directory when needed:

```bash
python project/metric_analysis/plot_carla_metrics.py --output-dir /tmp/carla-metric-plots
```

## Seed labels

The plot legend's model seeds 0, 1, and 2 come from the three training-run
directories configured near the top of the script. The `seed` column inside
each CSV is instead the per-scenario simulation seed, so it is not used as the
legend grouping variable.

## Metric selection

The metric groups are explicit near the top of the script so they are easy to
extend. `total_distance_travelled_sum` and all `reward_components/*` columns
are intentionally excluded. `comfort_score` is plotted for completeness but
annotated as broken because its current value is always 1.

## CARLA and nuPlan benchmark comparison

`plot_benchmark_comparison.py` compares CARLA, nuPlan single, and nuPlan multi
across the same three trained model seeds. It reads the aggregate
`metrics_mean` values from each run's `evaluation_summary.json`. Each benchmark
has three side-by-side bars, one per model seed, with compact value labels.

From the repository root, run:

```bash
source .venv/bin/activate
python project/metric_analysis/plot_benchmark_comparison.py
```

The four figures are written to
`project/metric_analysis/output/benchmark_comparison` by default. Override the
location when needed:

```bash
python project/metric_analysis/plot_benchmark_comparison.py \
    --output-dir /tmp/benchmark-comparison-plots
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

Choose the training seed with `--seed`:

```bash
source .venv/bin/activate
python project/metric_analysis/plot_finetune_comparison.py --seed 0
```

The default outputs for seed 0 are written to
`project/metric_analysis/output/finetune_comparison/seed0`. A different output
directory can be supplied with `--output-dir`.

The available pairs are listed explicitly in `ORIGINAL_RUN_BY_SEED` and
`FINETUNED_RUN_BY_SEED` near the top of the script. To compare a future seed,
add its original and fine-tuned experiment directories to those dictionaries,
then pass that seed on the command line. Requesting a seed without both entries
produces an error listing the configured paired seeds.
