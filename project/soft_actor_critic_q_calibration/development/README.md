# Development

Shared implementation for all runs. Run commands from the repository root after `source .venv/bin/activate`.

| Location | Purpose |
|---|---|
| `run_probes.py`, `probe_env.py`, `models.py`, `probe_returns.py` | Collection, simulator snapshots, checkpoint loading, return/target calculations |
| `collect_geometry.py`, `geometry_batch.py`, `plot_geometry.py` | Reference distance collection across the eight CARLA maps and per-model cumulative distributions. [Second run](../runs/second_20261122/README.md) |
| `ppo_models.py`, `collect_ppo_geometry.py`, `select_ppo_binned_probes.py`, `collect_ppo_probes.py` | PPO reference collection, state selection and forced-action continuations. [PPO value-target definition](../docs/PPO_ROUND_THREE.md) |
| `compare_round_three.py` | Regenerate the pooled PQL, TES and PPO comparison from the three models’ saved probes. |
| `launch_first.py`, `configs/` | Experiment configuration and launcher |
| `plot_*.py`, `summarize_probes.py`, `analyze_run.py` | Analysis from saved records |
| `run_layout.py`, `plot_paths.py` | Shared output organization and per-model entry points |
| `native_probe/`, `tests/` | Private native binding and verification |

```bash
python -m pytest project/soft_actor_critic_q_calibration/development/tests -q
python project/soft_actor_critic_q_calibration/runs/first_20261114/pql/scripts/analyze.py --help
python -m project.soft_actor_critic_q_calibration.development.launch_first --output project/soft_actor_critic_q_calibration/runs/NEW_RUN --dry-run
```

The launcher example only previews commands. Actual collection requires an explicitly authorized run. Model scripts read `records/` and write figures to `plots/`, with supporting tables/counts in `records/analysis/`.

Historical manifests retain their original source hashes and paths. Relocation does not change the recorded measurements; those completed runs are for analysis, not resumption with modified source.

[Design and validation](../docs/README.md) · [Project navigation](../README.md)

## Compare forced actions in an HTML replay

From the repository root, replay one probing situation and continuation repeat:

```bash
source .venv/bin/activate
python project/soft_actor_critic_q_calibration/runs/first_20261114/pql/scripts/analyze.py \
  --render-probe episode_000__frame_00015__ego_0041 --repeat 0
```

Use the filename stem from the model’s `records/probes/` directory. The command prints the self-contained HTML path under `renders/`. Open it in a browser, select a forced first action, and use the time slider or playback controls. All action choices share the same reference scene, probing car, and continuation repeat. The standard episode viewer shows the full scene. Car 0 is the probing car. Each first-action button shows the smaller online critic estimate, the training target, and the observed discounted reward return recorded for that action and continuation repeat. Lighter cells indicate higher targets within this grid. The comparison toggle beside **world cam** shows all 12 actions on a shared timeline in the same jerk-grid order. Rendering uses the saved checkpoint and its recorded inference device by default.
