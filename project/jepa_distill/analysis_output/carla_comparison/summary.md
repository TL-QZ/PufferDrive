# CARLA metric comparison

All nine episode CSVs contain the same 1,000 map/seed pairs. The baseline and both JEPA series use CARLA at `dt=0.1`, 6,000 steps, seed 42, and mean action selection.

## Puffer score

| Evaluation | Puffer score |
|---|---:|
| Baseline | 0.67882 |
| JEPA (2026-09-26), k=1 | 0.67789 |
| JEPA (no warm start), k=1 | 0.69780 |

| k | JEPA (2026-09-26) | JEPA (no warm start) |
|---:|---:|---:|
| 1 | 0.67789 | 0.69780 |
| 2 | 0.66077 | 0.64869 |
| 3 | 0.27456 | 0.29997 |
| 4 | 0.09948 | 0.09821 |

## Aggregation and interpretation

Aggregate JSON values are used for the baseline and chunk series. No-warm-start ordinary metrics are CSV means; `infractions_per_scenario = sum(total_infraction_count) / row_count`, and `avg_distance_per_infraction = sum(total_distance_travelled_sum) / max(sum(total_infraction_count), 1)`, matching the evaluator reducer.

At evaluation `dt=0.1`, k=1 replans each step; k=2, 3, and 4 hold each selected slot for three steps and replan every 6, 9, and 12 steps. One checkpoint is available per series, so plots have no error bars. `comfort_score` is a known broken metric.

The 2026-09-26 chunk series uses `checkpoint.pt`; the no-warm-start series uses a separately trained `final_model.pt`. This comparison does not isolate warm-start effects.
