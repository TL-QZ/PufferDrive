# nuplan_multi metric comparison

All nine episode CSVs contain the same 1,000 map/seed pairs. Summary JSON supplies the 22 plotted metrics; episode CSVs validate scenario identity and counts.

## Puffer score at k=1

| Evaluation | Puffer score |
|---|---:|
| Baseline | 0.49146 |
| JEPA (2026-09-26) | 0.49890 |
| JEPA (no warm start) | 0.50163 |

## Puffer score by chunk schedule

| k | JEPA (2026-09-26) | JEPA (no warm start) |
|---:|---:|---:|
| 1 | 0.49890 | 0.50163 |
| 2 | 0.48440 | 0.48490 |
| 3 | 0.44395 | 0.44833 |
| 4 | 0.36815 | 0.38215 |

## Protocol and interpretation

Both JEPA series and the baseline use seed 42, mean actions, `dt=0.1`, and 200-step episodes. At this rate, k=1 replans every step; k=2, 3, and 4 hold each selected slot for three steps and replan every 6, 9, and 12 steps. The `comfort_score` metric is known to be broken.

The shared series label `JEPA (2026-09-26)` is retained for consistency with the CARLA plots. Exact checkpoint paths for every record are in `provenance.json`.
