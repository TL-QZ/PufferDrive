# CARLA comparison plots

From the repository root, activate the environment and run:

```bash
source .venv/bin/activate
python project/jepa_distill/plot/plot_carla_comparison.py
```

The defaults load the synced baseline summary, the four `carla_chunk_eval_20260926/k{k}` summaries, and the four full `source_episode_metrics.csv` files from `carla_renders_no_warm_start/run_k{k}/k{k}`. The script checks all inputs, the 1,000-row counts, CARLA protocol fields, expected k schedules, finite metrics, and matching map/seed sets before it creates output files. It does not read the selected-failure summaries.

Input and output defaults are resolved from the script location, so invoking the script by its absolute path also works from another current directory.

The output defaults to `project/jepa_distill/analysis_output/carla_comparison/`:

- `k1_vs_baseline.png/.pdf`: six panels comparing Baseline, JEPA (2026-09-26), and JEPA (no warm start) at k=1.
- `performance_vs_k.png/.pdf`: six panels showing both JEPA curves for k=1–4 and a dashed baseline reference.
- `details/<comparison>/<metric_group>.png/.pdf`: the four shared metric-group figures for each comparison.
- `aggregate_metrics.csv`, `provenance.json`, and `summary.md`: nine aggregate records, source paths/counts/protocols, Puffer-score values, and aggregation notes.

No-warm-start CSV means use the evaluator reductions for `infractions_per_scenario` and `avg_distance_per_infraction`. The two JEPA series use different checkpoints (`checkpoint.pt` and `final_model.pt`), so their difference does not isolate a warm-start effect. Each series has one checkpoint and no error bars; `comfort_score` is a known broken metric.

Override any input root or output location with `--baseline-json`, `--chunk-eval-root`, `--no-warm-start-root`, or `--output-dir`. For example:

```bash
python project/jepa_distill/plot/plot_carla_comparison.py \
  --output-dir /tmp/carla_comparison
```

## nuPlan comparison plots

From the repository root, activate the environment and pass the two completed evaluation roots:

```bash
source .venv/bin/activate
python project/jepa_distill/plot/plot_nuplan_comparison.py \
  --jepa-root experiments/jepa_distill/nuplan_eval_jepa_20260927 \
  --no-warm-start-root experiments/jepa_distill/nuplan_eval_no_warm_start_20260927
```

For each benchmark (`nuplan_single` and `nuplan_multi`), the plotter reads `run_k1/k1` through `run_k4/k4` beneath both roots. Each run must contain `protocol.json` and `eval/<benchmark>_act_k<k>/standalone/evaluation_summary.json`, plus that summary's adjacent `episode_metrics.csv`. The default baseline root is `experiments/baseline_run_sync_2026-08-24/baseline_run_sync_2026-08-24_2026-08-26_16-23-18_seed0/eval`; exactly one immediate `*/evaluation_summary.json` is required under each `<benchmark>_final_model_mean_metrics/` directory. Override it with `--baseline-eval-root` if needed.

The loader checks all inputs for both benchmarks before writing any plots: 22 finite metrics, 1,000 scenarios and episodes, expected seed-42 mean-action protocol fields, one checkpoint per series, and matching map/seed pairs across each benchmark's nine CSVs. The schedule is k=1 replanning every step; k=2, 3, and 4 hold each slot for three steps and replan every 6, 9, and 12 steps. No simulator run is started.

Output defaults to `project/jepa_distill/analysis_output/nuplan_comparison/<benchmark>/`. Each benchmark gets two overview figures, eight metric-group detail figures, and `aggregate_metrics.csv`, `provenance.json`, and `summary.md`. Figures are saved as PNG and PDF. The shared series label `JEPA (2026-09-26)` is kept for consistency with the CARLA plots; `provenance.json` records the exact checkpoint path for every record.

Use `--output-dir /tmp/nuplan_comparison` to write figures and reports elsewhere.
