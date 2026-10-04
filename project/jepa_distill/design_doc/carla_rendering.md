# Distilled-student CARLA rendering

Render failures from completed student evaluations as interactive HTML, using the same PufferDrive CARLA-map replay pipeline as the baseline. The default selects the first **10 scenarios with any infraction per k**, in evaluation CSV order. It does not rerun the full benchmark.

## Launch

One GPU, all four schedules sequentially:

```bash
CUDA_VISIBLE_DEVICES=2 project/jepa_distill/scripts/render_carla.sh \
  --eval-root experiments/jepa_distill/carla_eval_no_warm_start_4gpu_20260927 \
  --output-dir experiments/jepa_distill/carla_renders_no_warm_start_sequential
```

Four GPUs, one schedule on each GPU:

```bash
CUDA_VISIBLE_DEVICES=2,3,4,5 project/jepa_distill/scripts/render_carla_4gpu.sh \
  --eval-root experiments/jepa_distill/carla_eval_no_warm_start_4gpu_20260927 \
  --output-dir experiments/jepa_distill/carla_renders_no_warm_start
```

The parallel command maps k=1/2/3/4 to GPU 2/3/4/5. Each process sees its GPU as `cuda:0`. Both commands require a fresh output directory. The parallel launcher validates all four inputs before creating output or starting GPU jobs, waits for all children, reports failures, and terminates their process groups on interruption.

| Option | Meaning |
| --- | --- |
| `--k 1` | Render one schedule with the single-GPU script; default `all` runs sequentially. |
| `--failure-mode collision` | Select `all_infractions` (default), `collision`, `at_fault_collision`, `offroad`, or `red_light`. |
| `--max-failures 1` | Limit scenarios per k; default 10, `all` means unlimited. |
| `--workers 2` | Simulator workers per GPU; default 2. Numeric-library thread counts default to 1. |
| `DRY_RUN=1` or `--dry-run` | Validate paths, metadata, and selections without loading checkpoints, allocating GPUs, or creating files. |

## Inputs and behavior

`--eval-root` accepts either evaluation layout: `kN/` from the sequential evaluation launcher, or `run_kN/kN/` from the four-GPU launcher. Each k must have its protocol, episode CSV, and resolved benchmark configuration. Ambiguous matches, missing inputs, and inconsistent k metadata are errors.

The renderer loads the checkpoint recorded in that k's protocol and uses the existing `ChunkPolicyAdapter`. At `dt=0.1`, k=1 replans every step; k=2/3/4 hold each selected slot for three steps and replan after 6/9/12 steps. The saved environment and inference settings, including mean controls and BF16, are restored. Device and worker counts are launch-time resource choices. Replay uses the recorded inference batch size and clears cached chunks between replay waves and at episode resets.

Source evaluation files are never modified. Copies of the metrics and provenance go into the new render output. Zero matching scenarios is a successful empty selection and is reported without an HTML-index claim. These are policy re-simulations from recorded map/seed pairs; the output contains new replay metrics rather than replacing the original evaluation metrics.

## Outputs

For the single-GPU launcher, each k is under `<output-dir>/kN/`. For the parallel launcher, it is under `<output-dir>/run_kN/kN/`, with console logs at `<output-dir>/kN.log`.

Within each k directory, the native replay pipeline writes:

- `failures/selected_failures.csv`: selected source scenarios.
- `failures/episode_metrics.csv` and `evaluation_summary.json`: replay metrics.
- `failures/replays/*.replay.zlib`: captured trajectories and policy observations/actions.
- `failures/rendered_replays/*.html` and `index.html`: interactive pages and gallery.

The k directory also contains `source_protocol.json`, `source_episode_metrics.csv`, `source_resolved_benchmark.yaml`, an output-local `benchmark_catalog.yaml`, `selected_failures.csv`, and `render_report.json`. The report records the input checkpoint, selected count, recorded batch size, and resulting HTML index. HTML generation uses the baseline renderer's own thread pool, independently of the simulator-worker count.

## Verification

Verified on 2026-09-27:

- **43 tests passed** across `test_render_carla.py`, `test_render_carla_launchers.py`, `test_evaluate_carla.py`, and `test_evaluate.py`. Coverage includes both layouts, filters/count limits, empty selections, immutable inputs, metadata-only dry runs, GPU dispatch, child failures, SIGINT/SIGTERM cleanup, padded inference, and chunk resets across replay waves.
- **Real four-GPU rendering passed for k=1/2/3/4**, using the command above with `--max-failures 1 --workers 1` and output `/tmp/jepa_render_smoke_20260927`. Each job replayed one full 6,000-step scenario and produced one nonempty replay file, HTML page, and gallery index. CUDA BF16 was enabled; observed GPU memory was approximately 1.65 GB per job.
- Saved environment settings matched the source evaluations exactly. All four original CSV SHA-256 hashes remained unchanged. Replayed map/seed, collision/offroad/red-light rates, and total distance matched the selected source rows.
- Python compilation, shell syntax, and diff whitespace checks passed. Independent review found no remaining material defects.

Machine-readable smoke verification is `/tmp/jepa_render_smoke_20260927/verification.json`; indexes are `/tmp/jepa_render_smoke_20260927/run_kN/kN/failures/rendered_replays/index.html`. Temporary smoke outputs may be removed by system cleanup.

Observation capture produces large artifacts: these full-length smoke scenarios generated roughly 310–352 MB replay files and 413–469 MB self-contained HTML pages each. Browser interaction was not manually inspected. The normal 10-per-k batch is left for the user to launch.
