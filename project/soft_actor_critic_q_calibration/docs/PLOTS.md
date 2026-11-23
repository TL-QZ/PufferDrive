# Q-calibration plot index

Collected experiment data and run summaries live in `runs/<run>/<model>/records/`. Images and at most one README per plot folder live in `runs/<run>/<model>/plots/`. Derived tables, JSON counts, and additional notes live in `runs/<run>/<model>/records/analysis/`.

## First experiment — PQL

| View | Figure |
|---|---|
| Critic errors versus action probability | [Six-panel figure](../runs/first_20261114/pql/plots/target_error_plots/critic_target_errors.png) |
| Min-Q errors grouped by action rank | [Rank-grouped figure](../runs/first_20261114/pql/plots/target_error_plots/min_q_target_errors_by_rank.png) |
| Critic errors versus target | [Target-axis figure](../runs/first_20261114/pql/plots/target_error_plots/critic_errors_vs_target.png) |
| Target, probability, and critic error | [3D figure](../runs/first_20261114/pql/plots/target_error_plots/critic_errors_3d.png) |
| Raw Q outputs with target overlays | [Six-panel figure](../runs/first_20261114/pql/plots/raw_q_plots/critic_raw_q.png) |
| Raw min-Q and targets grouped by rank | [Rank-grouped figure](../runs/first_20261114/pql/plots/raw_q_plots/min_q_raw_by_rank.png) |
| Targets versus driving outcomes | [Full-range figure](../runs/first_20261114/pql/plots/driving_quality_plots/target_vs_driving_quality.png) |
| Median-centered targets versus outcomes | [Full-range figure](../runs/first_20261114/pql/plots/driving_quality_plots/centered_target_vs_driving_quality.png) |
| Median-centered targets, X from −0.4 to 0.2 | [Zoomed figure](../runs/first_20261114/pql/plots/driving_quality_plots/centered_target_vs_driving_quality_zoom.png) |

Plotting commands still take the source run directory as input. They resolve the separate plot destination automatically using `plot_paths.py`. For example:

```bash
source .venv/bin/activate
MPLCONFIGDIR=/tmp/q_calibration_matplotlib python -m project.soft_actor_critic_q_calibration.development.plot_driving_quality \
  project/soft_actor_critic_q_calibration/runs/first_20261114/pql \
  --center-target --x-limits -0.4 0.2
```
