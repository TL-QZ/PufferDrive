# CARLA metric comparison plots

## Scope

Add one plotting script under `project/jepa_distill/plot/`. Compare the synced baseline with the four `2026-09-26` chunk evaluations and the four no-warm-start full-evaluation CSVs. Keep generated figures and reports under `project/jepa_distill/analysis_output/`.

## Inputs and validation

- Baseline: the specified `evaluation_summary.json`; chunk series: each `k1`–`k4` standalone summary; no-warm-start series: each `source_episode_metrics.csv` (1,000 source episodes, excluding `failures/` artifacts).
- Require 1,000 episodes per input, finite required metrics, matching CARLA protocol (`dt=0.1`, 6,000 steps, seed 42, mean action selection), expected k metadata, and all four no-warm-start CSV files before creating outputs.
- Derive ordinary no-warm-start metrics as column means. Derive `infractions_per_scenario` as `sum(total_infraction_count) / row_count` and `avg_distance_per_infraction` as `sum(total_distance_travelled_sum) / max(sum(total_infraction_count), 1)`.

## Outputs

- Three-series k=1 comparison and four-point k curves with the baseline as a horizontal reference, each with a six-panel overview and four metric-group figures; save PNG and PDF versions.
- One nine-row aggregate CSV with source paths and counts, JSON provenance, and a short Markdown summary with key scores, the full Puffer-score curve, aggregation notes, and schedule details.
- Reuse `METRIC_GROUPS` and `METRIC_UNITS` from `project/metric_analysis/plot_finetune_comparison.py`. Note that there is one checkpoint per series, so no error bars are shown; `comfort_score` is a known broken metric.

## Interpretation notes

At evaluation `dt=0.1`, k=1 replans each step; k=2, 3, and 4 hold each selected slot for three steps and replan every 6, 9, and 12 steps. The chunk checkpoint and no-warm-start final model are different checkpoints, so this comparison does not isolate warm-start effects.
