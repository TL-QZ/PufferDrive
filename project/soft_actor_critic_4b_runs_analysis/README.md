# SAC 4B runs and PPO comparison

**The current final-checkpoint comparison is in [RESULTS.md](RESULTS.md).** It uses the seven jobs in [native_manifest.json](native_manifest.json), checks each native benchmark catalog and resolved spacing, and refreshes statuses and metrics from the evaluation artifacts. Pending jobs remain blank in the reports.

## Full-CARLA comparison protocol

All seven checkpoints use seed 42, 1,000 scenarios, 6,000 simulation steps, `dt=0.1`, mean actions, and no rendering. Goal spacing follows each run's benchmark protocol:

| Goal spacing | Checkpoints |
|---:|---|
| 10 m | n-step 4, n-step 2, PQL round 4.1, PQL official round 1 |
| 30 m | TES round 2, PPO 4B, PQL official round 2 |

TES and PPO reuse their previously completed 30 m full-CARLA results; the summarizer verifies their saved resolved benchmark settings against the native catalogs. The other five jobs use separate output directories and the catalog selected for each run. No 30 m result substitutes for a 10 m job. The previous six-job common-30 m reports are copied to [historical_common30m](historical_common30m/); the original `manifest.json` and result folders remain in place.

The full report shows rates as percentages. Its total-distance column converts the sum across episode agents to kilometres; distance per infraction divides the summed distance by summed infractions. The machine-readable CSV retains raw rate fractions and both source distance metrics. See [verification.json](verification.json) for per-job status, checkpoint/config/catalog hashes, resolved goal spacing, episode counts, and map/seed identity checks.

## Training comparison

The learning-curve analysis is separate from the final-checkpoint evaluations. It matches observed PPO and SAC `carla_fast` points by global received-agent steps, without using evaluation scores. Read [learning_curve_report.md](learning_curve_report.md) for matched point counts, alignment offsets, coverage, and limitations; see [setup_inventory.md](setup_inventory.md) and [configuration_differences.csv](configuration_differences.csv) for run and configuration details.

These are individual checkpoints, not replicated algorithm estimates. The n-step-2, PQL round-4.1, and PQL official round-2 checkpoints include learner-only recovery with fresh replay and environments. Training settings differ across runs, so score gaps cannot be assigned to one configuration change.

## Artifacts and reproduction

Generated results, plots, render galleries, checkpoints, logs, and verification reports are local-only and excluded by this project's `.gitignore`. The links below resolve after those artifacts have been generated locally. Scripts, benchmark catalogs, manifests, and technique documentation remain available for version control.

| Artifact | Contents |
|---|---|
| [RESULTS.md](RESULTS.md) | Generated seven-job comparison and current validation state |
| [full_carla_report.md](full_carla_report.md) | Full-CARLA protocol, comparison table, and aggregation notes |
| [full_carla_metrics.csv](full_carla_metrics.csv) | All summary metrics and per-case metadata |
| [native_manifest.json](native_manifest.json) | Checkpoint provenance, per-job catalogs and hashes, goal spacing, and exclusions |
| [verification.json](verification.json) | Completion, protocol, hash, resolved-spacing, and scenario-pair checks |
| [resource_summary.json](resource_summary.json) | Resource samples from the native batch, filtered by its manifest creation time |
| [historical_common30m](historical_common30m/) | Archived reports from the earlier shared 30 m comparison |

Activate the repository environment before running the report generator:

```bash
source .venv/bin/activate
python -B project/soft_actor_critic_4b_runs_analysis/summarize_full_evaluations.py
```

The summarizer reads existing evaluation artifacts and never launches jobs. Re-run it after evaluations finish to refresh the reports. The evaluation runner writes per-case status JSON and separate logs; `resource_usage.jsonl` is retained as the raw monitoring history.
