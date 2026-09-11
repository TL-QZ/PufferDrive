# Collaborator Setup and Experiment Guide

## Current state of this repository

This repository began as a fork of the original
[Emerge-Lab/PufferDrive](https://github.com/Emerge-Lab/PufferDrive) repository. There are now two
Git remotes with different roles:

| Name | Repository | Meaning |
|---|---|---|
| `origin` | [TL-QZ/PufferDrive](https://github.com/TL-QZ/PufferDrive) | Our fork. Collaborators clone this repository and work from its `develop-3.0` branch. |
| `upstream` | [Emerge-Lab/PufferDrive](https://github.com/Emerge-Lab/PufferDrive) | The original repository. We use it only as the source of upstream changes that may later be reviewed and integrated. |

`develop-3.0` is the primary working branch of our fork. It includes our experiment workflows,
documentation, and other changes that are not present in the original repository. It is therefore
not interchangeable with upstream's `3.0` branch.

The August 24 sync brought upstream history through commit `8adcec26` into this fork. From that
shared point, the two branches developed separately. As checked on **2026-09-08**, our
`develop-3.0` is at `58a7cd81`, while the refreshed upstream `3.0` is at `dee651c8`: upstream has
12 commits not yet integrated into our branch, and our branch has 9 fork-specific commits not
present upstream. This is a snapshot, not a permanent property of either branch. To check the
relationship again:

```bash
git fetch --prune upstream
git rev-list --left-right --count upstream/3.0...develop-3.0
# First number: upstream-only commits; second number: develop-3.0-only commits
```

The rest of this guide explains how to set up this fork and run its **August 24 post-sync baseline**:
CARLA self-play → nuPlan fine-tuning → evaluation → failure replay → plots. Commands assume
Bash and the repository root as the working directory. Results can differ across commits, hardware,
CUDA/PyTorch versions, and other dependencies, so record them with each experiment. For implementation
details, see the [codebase guide](pufferdrive_codebase_guide.md).

## 1. Set up the environment

### Clone this fork

If this workspace already contains the repository on `develop-3.0`, skip this step. Otherwise,
clone our fork and start from `develop-3.0`; do not clone the original Emerge-Lab repository because
it does not contain all the experiment workflows documented here.

```bash
git clone --branch develop-3.0 https://github.com/TL-QZ/PufferDrive.git
cd PufferDrive
git branch --show-current  # Expected: develop-3.0
git rev-parse HEAD         # Record this commit with the experiment
```

To reproduce an older result, use its recorded commit and saved configuration rather than assuming
the latest `develop-3.0` produces the same result.

### Create the Python environment and build the simulator

For the full baseline, use **Linux with NVIDIA GPUs**: two GPUs for training/fine-tuning and one GPU
for evaluation/rendering. From the repository root, follow the
[local installation instructions](../../README.md#install-local):

```bash
uv venv
source .venv/bin/activate
uv pip install -e .

# Build C extensions (required after any .h/.c change)
python setup.py build_ext --inplace --force
```

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) first if it is unavailable.
The experiment scripts explicitly activate **`<repo>/.venv/bin/activate`**; activating a different
virtual environment in your terminal does not override that choice.

If your machine differs from the original setup, check these before changing package versions:

| Requirement | What to check |
|---|---|
| Python and packages | Dependencies come from [pyproject.toml](../../pyproject.toml) and [setup.py](../../setup.py). They are not a fully pinned reproduction environment; retain the versions from a working run when available. |
| CUDA and PyTorch | Use a [PyTorch build compatible with your machine](https://pytorch.org/get-started/locally/). GPU extension compilation also needs a CUDA development toolkit/compiler, not only a driver. |
| C/C++ build tools | The simulator is compiled locally. Missing compiler or development-library errors must be resolved before training. |
| No administrator access | Use the compiler, CUDA modules, or container provided by your own compute system. Python packages can live in your own venv; changing Python packages cannot replace a missing system toolchain. |

The root README and `docs/cluster_training.md` also contain Singularity, overlay, scheduler, and
filesystem settings created by the original NYU developers for their cluster. Those settings are
site-specific; ignore them unless you are working in a compatible environment. For another cluster,
adapt the local installation above using that cluster's own documentation.

### Verify the installation

Run these with the venv active on the machine where you will run PufferDrive:

```bash
nvidia-smi
python -c 'import torch; from pufferlib import _C; from pufferlib.ocean.drive import binding; print("torch:", torch.__version__, "CUDA:", torch.version.cuda, "visible GPUs:", torch.cuda.device_count())'
wandb login --verify
```

The imports should succeed, and PyTorch should see your allocated GPUs. The baseline configs enable
Weights & Biases; its evaluation scripts also verify login. Use an account with access to the intended
project. These launchers do not provide a general `wandb=false` argument.

Every time any `.c` or `.h` file changes—including after a Git sync—rebuild before running PufferDrive:

```bash
# Build C extensions (required after any .h/.c change)
python setup.py build_ext --inplace --force
```

Also rebuild after replacing the PyTorch build. For available tests, run `make help`; the current
targets include `test-unit`, `test-eval`, `test-c`, and `test-notebooks` (`make test` runs those local
suites).

## 2. Prepare the data

The **eight CARLA map binaries are already included** under
`pufferlib/resources/drive/binaries/carla/`. This workflow uses those maps in PufferDrive's simulator;
it does not require a separate CARLA server.

nuPlan is downloaded separately. The [dataset manifest](../../data_utils/datasets.yaml) lists these
approximate sizes for **preconverted bins**, not the original sensor dataset:

| Dataset | Approximate storage | Baseline usage |
|---|---:|---|
| `nuplan_train` | 475 GB | Fine-tuning reads 10,000 maps from `data/nuplan_train`. |
| `nuplan_val` | 48 GB | Each nuPlan benchmark evaluates 1,000 scenarios from `data/nuplan_val`. |
| `nuplan_mini_train`, `nuplan_mini_val` | 10 GB each | Smaller development datasets; not drop-in replacements for the baseline's configured paths/counts. |

Allow roughly **523 GB for the full preconverted pair**, plus environment, checkpoint, replay, and plot
storage. Inspect the download before starting it:

```bash
python data_utils/fetch_data.py --list
python data_utils/fetch_data.py nuplan_train nuplan_val --dry-run
python data_utils/fetch_data.py nuplan_train nuplan_val
```

These entries are public and do not require AWS credentials. Downloads use the AWS CLI dependency;
read the manifest's dataset license before use. A bare `fetch_data.py` downloads the **mini** pair,
which is not what the baseline expects.

For another disk, pass `--data-root /path/to/data`. Then update the fine-tune YAML's `env.map_dir`
and the benchmark YAML's nuPlan `map_dir` entries to the downloaded directories. The fetch destination
does not automatically update experiment configs. Sharing read-only data is fine; use separate run outputs.

For conversion from raw nuPlan, follow [nuPlan data preparation](../../docs/nuplan_data.md).
For storage/download details, see [data storage](../../docs/data_storage.md).

## 3. Understand the repository structure

Most of our experiment-specific work lives under `project/`. The simulator and training framework
remain under `pufferlib/`, so a collaborator can usually understand or reproduce an experiment by
starting in its `project/<experiment>/` directory and following the paths referenced there.

| Path | Purpose |
|---|---|
| `project/` | Our experiment configs, launchers, evaluation/rendering scripts, analysis tools, and explanatory notes. |
| `pufferlib/ocean/drive/` | The C driving simulator, Python environment wrapper, bindings, and visualization code. |
| `pufferlib/` | The shared PPO training/evaluation framework, model code, and base configuration. |
| `pufferlib/resources/` and `data_utils/` | Bundled map resources plus dataset download/conversion tools. |
| `tests/`, `docs/`, `notebooks/`, and `scripts/` | Shared validation, upstream documentation, learning material, and general utilities. |

Inside `project/`, each major experiment keeps its own files together:

| Path | What is there |
|---|---|
| `baseline_run_sync_2026-08-24/` | The current baseline recipe described by this guide. |
| `second_run_nightly_best_config/` and `third_run_nightly_best_config/` | Older experiment snapshots retained for reference. |
| `metric_analysis/` | Standalone scripts for plotting and comparing saved evaluation results. |
| `codebase_learning/` | This setup guide and source-grounded explanations of the codebase, timing, PPO, and metrics. |

The baseline directory separates `override_config/`, `train/`, `eval/`, and `render/`. Its
`contexts_and_notes.md` records the experiment's intent and decisions, while `yaml_overrides.py`
converts the experiment YAML into command-line overrides. Generated runs are written under
`experiments/`, not back into the experiment recipe directory.

## 4. Run the baseline in order

Use [baseline_run_sync_2026-08-24](../baseline_run_sync_2026-08-24/docs/contexts_and_notes.md).
Its configs are in `override_config/`, with launchers in `train/`, `eval/`, and `render/`.
Older experiment folders are historical recipes; do not mix their configs or checkpoints into this run.

| Stage | Budget / timing | GPUs |
|---|---|---:|
| CARLA self-play | 10 billion global agent steps; `dt=0.3` | 2 |
| nuPlan SDC fine-tune | 100 million global agent steps; `dt=0.1` | 2 |
| Final evaluation | CARLA, nuPlan single, nuPlan multi; `dt=0.1` | 1 |
| Failure replay | Selected scenarios from final evaluation | 1 |

These are full experiments, not installation tests. The CARLA training/evaluation timestep difference
is part of this recorded recipe; preserve it when reproducing the baseline. GPU memory and completion
time depend on the hardware. A dry run checks the command, not whether training fits in GPU memory.

The examples below use **seed 0**. Repeat for seeds `1` and `2` when reproducing the three-seed comparison.
Replace GPU indices with those assigned to you. In the same terminal, define:

```bash
BASELINE=project/baseline_run_sync_2026-08-24
```

1. **Inspect the CARLA launch.** Check the printed config, GPU indices, and output directory.

   ```bash
   CUDA_VISIBLE_DEVICES=0,1 DRY_RUN=1 bash "$BASELINE/train/launch_2gpu.sh" 0
   ```

2. **Train the CARLA policy.** Wait for completion and confirm `final_model.pt` and `config.yaml` exist in the printed run directory.

   ```bash
   CUDA_VISIBLE_DEVICES=0,1 bash "$BASELINE/train/launch_2gpu.sh" 0
   ```

3. **Fine-tune that checkpoint on nuPlan.** The launcher discovers the completed CARLA run for the same seed.

   ```bash
   CUDA_VISIBLE_DEVICES=0,1 bash "$BASELINE/train/launch_nuplan_sdc_finetune_2gpu.sh" 0
   ```

4. **Evaluate both final checkpoints.** Each command runs all three final benchmarks.

   ```bash
   CUDA_VISIBLE_DEVICES=0 bash "$BASELINE/eval/evaluate_final_model.sh" 0
   CUDA_VISIBLE_DEVICES=0 bash "$BASELINE/eval/evaluate_nuplan_sdc_finetune.sh" 0
   ```

5. **Inspect failures.** These examples render at most 10 nuPlan-single offroad failures from each model.

   ```bash
   CUDA_VISIBLE_DEVICES=0 bash "$BASELINE/render/render_eval_failures.sh" 0 nuplan_single offroad 10
   CUDA_VISIBLE_DEVICES=0 bash "$BASELINE/render/render_nuplan_sdc_finetune_failures.sh" 0 nuplan_single offroad 10
   ```

All these launchers accept `DRY_RUN=1`. Later-stage dry runs still require their source checkpoints
or evaluation CSVs to exist. No matching failures means there may be no failure pages to inspect.

### Find outputs and resume safely

| Output | Location / meaning |
|---|---|
| CARLA runs | `experiments/baseline_run_sync_2026-08-24/<run_name>/` |
| Fine-tuned runs | `experiments/baseline_run_sync_2026-08-24/nuplan_sdc_finetune/<run_name>/` |
| Final evaluation | Within each run: `eval/<benchmark>_final_model_mean_metrics/<timestamp>/`, containing `resolved_benchmark.yaml`, `episode_metrics.csv`, and `evaluation_summary.json`. |
| Failure pages | The render script prints the selected source CSV and analysis output directory; open the generated HTML index. |

Record the **printed run name**. By default, each new training invocation creates a timestamped name.
To resume, pass `RUN_NAME=the_exact_original_run_name` to the same training launcher and seed.
Its own `trainer_state.pt` supplies the resume state; a new name starts a new run.

Discovery scripts require **exactly one matching run per seed** in the relevant root. If several match,
select/archive the intended run deliberately or adapt a copied workflow to an explicit path; do not
assume the newest is selected. The render scripts select the latest matching evaluation CSV, so inspect
their printed source path before replaying.

The fine-tune launcher stages only the CARLA `final_model.pt` and `config.yaml`. Keep that behavior:
copying the source `trainer_state.pt` can resume the old optimizer and step counters instead of starting
a fresh fine-tune. Preserve full run directories when transferring results, including checkpoint configs.

## 5. Plot the correct results

The scripts in [metric_analysis](../metric_analysis/README.md) are standalone and readable, but
**their current input mappings still point to older second-run results**. They do not automatically
select the baseline you just ran. Updating those inputs is required before plotting new results.

| Script | Input mapping to review |
|---|---|
| [plot_carla_metrics.py](../metric_analysis/plot_carla_metrics.py) | `CARLA_CSV_BY_MODEL_SEED`: one final CARLA episode CSV per model seed. |
| [plot_benchmark_comparison.py](../metric_analysis/plot_benchmark_comparison.py) | `BENCHMARK_JSON_BY_MODEL_SEED`: final summary JSONs for each seed and benchmark. |
| [plot_finetune_comparison.py](../metric_analysis/plot_finetune_comparison.py) | `ORIGINAL_RUN_BY_SEED` and `FINETUNED_RUN_BY_SEED`: paired run directories. Each must contain exactly one matching final summary per benchmark. |

For a new experiment, copy the relevant plotting script into its experiment folder or update the explicit
mapping in a reviewed change. If copying, also adjust its `REPO_ROOT` calculation for the new file location.
Check model seed, pretrain/fine-tune stage, evaluation timestamp, benchmark config, and coverage
(`num_scenarios` versus `num_episodes`). Model seeds and per-scenario CSV seeds are different.

After updating the inputs, choose an experiment-specific output directory, for example:

```bash
python project/metric_analysis/plot_finetune_comparison.py --seed 0 \
  --output-dir project/metric_analysis/output/baseline_run_sync_2026-08-24/finetune/seed0
```

Each script writes several PNGs grouped by metrics. `--output-dir` changes the destination, **not the
input results**. Keep figure annotations consistent with the evaluated code: existing scripts retain
warnings about older metric behavior, which should be reviewed before reusing plots for new results.

## 6. Make changes without losing reproducibility

Keep experiment-specific configs and scripts under `project/<experiment>/`, with a distinct output root.
This fork also has changes outside `project/`; it is **not an untouched copy of upstream 3.0**.
Treat `develop-3.0` as the integration branch and work on a separate branch in your own checkout:

```bash
git switch -c experiment/my-experiment
```

Before merging, review the diff and run checks appropriate to the change: shell syntax/dry runs for
launchers, Python tests for Python changes, and a C rebuild plus relevant tests for simulator changes.
Keep datasets, generated outputs, credentials, and virtual environments out of commits.

Review upstream changes before integrating them; a sync can change simulation or metric semantics.
Record the code revision, dependency versions, saved training config, resolved benchmark config,
and source checkpoint with each experiment. After a sync, rebuild changed C code and recheck the
experiment before comparing new results to an older baseline.

**First milestone:** complete the installation checks and the seed-0 CARLA dry run before launching training.
