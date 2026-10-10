# SAC implementation plan — selected capabilities

Updated 2026-10-09. This plan contains the 20 selected gaps, numbered consecutively within each section. Previous learner options C1/C2/C3/C5 are now B1–B4; previous execution items B1/B3/B4/B5 are now C1–C4. Progress: all 20 selected items are implemented, including E3 portable Python actor export. CPU verification and remaining validation limitations are recorded below. C visualizer inference is outside the selected export scope.

**Current foundation:** collection and random replay refresh, shuffled full-replay epochs, SAC updates, logging/dashboard, LR scheduling and learner checkpoints work with supplied components. A real CPU Drive CLI smoke passed: 64 received agent rows, 12 training cycles and 24 learner updates, including scenario/map resets. This verifies integration, not training quality or GPU performance. Keep the existing PPO-style structure and naming.

**Reuse approach:** reuse existing utilities directly or copy the corresponding code with necessary SAC adaptations. Preserve existing names and structure; identify where new SAC behavior is required.

## A — Basic Drive integration

| ID | Capability | Existing code to reuse | SAC adaptation |
|---|---|---|---|
| A1 | Drive actor and twin critics | [`DriveBackbone`](../../pufferlib/ocean/torch.py#L17), [`Drive`](../../pufferlib/ocean/torch.py#L311), its observation encoding and discrete-action conversion methods, and [`load_policy()`](../../pufferlib/pufferl.py#L2367). | Reuse the encoder and action mapping. Implemented logits-only `DriveSAC` and independent twin-Q `DriveSACCritic` in [torch_sac.py](../../pufferlib/ocean/torch_sac.py). Four CPU tests cover gradients, parameter ownership and PPO action-map parity. Startup wiring is implemented. |
| A2 | Replay storage | [PPO tensor allocation](../../pufferlib/pufferl.py#L179), indexed minibatch reads in `_train_ppo_transition()`, and [SAC replay insertion/replacement](../../pufferlib/pufferl_sac.py#L912). | Implemented [TensorReplay](../../pufferlib/sac_replay.py) with copied fixed-capacity storage behind `len`, `add`, `replace` and `get_batch`. Eight CPU tests cover ownership, replacement, ordering and capacity. Startup wiring, refresh and epoch scheduling are implemented. |
| A3 | Reset-boundary transport | [`Drive.step()`](../../pufferlib/ocean/drive/drive.py#L670), [C truncation/reset path](../../pufferlib/ocean/drive/drive.h#L4622), vector transport in [vector.py](../../pufferlib/vector.py), and [`PuffeSAC.evaluate()`](../../pufferlib/pufferl_sac.py#L912). | Implemented opt-in C final-observation capture, map-resample preservation and sparse vector info transport. SAC matches final observations by agent ID, bootstraps truncations and zeros terminal discounts. Paired real environments verified the pre-reset state; focused tests cover slot mapping and learner discounts. |
| A4 | Configuration and startup | [`load_config()`](../../pufferlib/pufferl.py#L2409), `main()`, [`load_env()`](../../pufferlib/pufferl.py#L2356), [`train()`](../../pufferlib/pufferl.py#L1432), `_get_git_metadata()` and `_save_experiment_config()`. | Implemented [SAC config](../../pufferlib/config/puffer_drive_sac.yaml), `load_config()`, `load_policy()` and component construction in `train()`. Run with `python -m pufferlib.pufferl_sac train puffer_drive [overrides]` (or `puffer-sac` after installation). Resolved config and metadata are saved; real CPU CLI smoke and startup/cleanup tests passed. |

## B — Learner options

| ID | Capability | Existing code to reuse | SAC adaptation |
|---|---|---|---|
| B1 | Gradient accumulation | [Minibatch/accumulation setup](../../pufferlib/pufferl.py#L219) and [backward/step block in `_train_ppo_transition()`](../../pufferlib/pufferl.py#L813). | Implemented row-weighted microbatch accumulation with critic → actor → temperature → target ordering once per logical batch. Numerical tests match full-batch parameters and metrics across uneven microbatches and final batches; epoch coverage and counters are checked. |
| B2 | Gradient clipping | [`clip_grad_norm_` calls before optimizer steps](../../pufferlib/pufferl.py#L818). | Implemented optional `max_grad_norm` clipping for actor and critic after accumulation and before stepping. Tests verify clipping occurs once per completed logical update. |
| B3 | Optimizer flexibility | [Adam/AdamW/Muon selection in `PuffeRL.__init__()`](../../pufferlib/pufferl.py#L252). | Implemented Adam/AdamW/Muon for actor and critics; temperature retains Adam. Current Heavyball `Muon` and legacy `ForeachMuon` imports are supported. Real installed Muon steps passed; checkpoint tests reject incompatible optimizer settings, including older v5 checkpoints. |
| B4 | Automatic batch sizing | [Automatic batch/horizon sizing](../../pufferlib/pufferl.py#L154) and [minibatch-size setup](../../pufferlib/pufferl.py#L219). | Implemented opt-in `auto`: replay capacity = max(warmup, received agent batch size, logical minibatch size); minibatch = min(256, explicit capacity), or 256 if both are auto. Explicit defaults remain 10000/256. Sizes are resolved before replay construction and saved in run configuration; tests cover sizing and persistence. |

## C — Execution and memory

| ID | Capability | Existing code to reuse | SAC adaptation |
|---|---|---|---|
| C1 | DDP | [Distributed setup in `train()`](../../pufferlib/pufferl.py#L1490), `derive_rank_seeds()`, `dist_sum()`, `dist_mean()` and rank-zero logging/checkpoint gates. | Implemented torchrun startup, rank-local replay, synchronized actor/critic/temperature updates and rank-zero outputs. All ranks fill replay before training; collection synchronizes counts once per call. `global_step` remains local for PPO's dashboard; `agent_steps`, insert counts and sampled rows are global. Two Gloo tests cover compiled updates against a combined-batch oracle, unequal received counts, temperature gradients, fresh-replay resume, warmup and shared early stop. A real two-rank CPU Drive smoke passed: 68 received rows for a 64-row budget (vector-batch overshoot), 7 cycles and 14 learner updates, with finite checkpoints. NCCL/GPU execution remains unverified. |
| C2 | Mixed precision | [Precision checks and storage setup](../../pufferlib/pufferl.py#L167), [AMP context construction](../../pufferlib/pufferl.py#L303) and forward/loss autocast usage. | Implemented `amp=true precision=bfloat16` for network forwards; probabilities, Q targets, losses and temperature stay FP32. The SAC backbone adapts masked pooling to the encoder output dtype. CPU BF16 checks cover learner arithmetic and Drive networks. CUDA BF16 remains unverified. |
| C3 | Compilation | [Compile block in `PuffeRL.__init__()`](../../pufferlib/pufferl.py#L239), `base_policy()`, state-key cleanup and CUDA graph step markers in `train()`. | Implemented existing `compile`, `compile_mode` and `compile_fullgraph` options for actor and online/target critics. Base modules own checkpoints and Polyak updates. Actual CPU Inductor updates match eager parameters and losses across repeated updates and tail microbatches; checkpoint restore passes. CUDA graph modes remain unverified. |
| C4 | CPU storage/offload | [Pinned CPU observation allocation](../../pufferlib/pufferl.py#L179), collection copies in `evaluate()`, and [indexed nonblocking minibatch transfer](../../pufferlib/pufferl.py#L788). | Implemented `cpu_offload`: CPU replay by default, learner-device replay when false. CUDA offload pins all five storage fields and explicitly allocates pinned sampled batches for nonblocking transfers. CPU storage/placement tests pass; the pinning test is skipped on this host because its CUDA allocator is unavailable. |

## D — Run management

| ID | Capability | Existing code to reuse | SAC adaptation |
|---|---|---|---|
| D1 | Logger construction | [Logger classes](../../pufferlib/pufferl.py#L1257) and the logger-selection block in `train()`. | Implemented: startup reuses W&B and TensorBoard and adapts Neptune artifact uploads for SAC and preserves an injected logger. C1 restricts construction, logging and artifact writes to rank zero. External logging services were not contacted during validation. |
| D2 | Architecture-aware loading and fine-tuning | [`load_policy()`](../../pufferlib/pufferl.py#L2367), configuration restoration in `train()`, and [`load_checkpoint_architecture()`](../../pufferlib/ocean/evaluation_utils/evaluation_utils.py#L203). | Implemented source architecture restoration using PPO's environment whitelist. `load_model_path` or remote `load_id` warms only the actor; critics, optimizers, temperature and counters start fresh. W&B downloads include source config; SAC Neptune uploads portable model/config fields. Rank zero downloads and shares the path, requiring shared storage across ranks. Real CPU warmstart preserved all 44 actor tensors and target scenario settings. Provider APIs are mocked in tests; live services remain unverified. |
| D3 | Automatic same-run checkpoint discovery | [Checkpoint-discovery block in `train()`](../../pufferlib/pufferl.py#L1544). | Implemented `trainer_state.pt` discovery in `train.data_dir`, with same-run identity validation. Explicit learner resume takes precedence; actor warmstart bypasses discovery, and `train.fresh_run=true` disables automatic discovery. Conflicting explicit sources fail. Learner resume restores models, optimizers, temperature, counters and RNG into a fresh environment/replay; exact continuation remains deferred. Failed restore preserves the existing config. Real CPU autodiscovery resumed 68 → 100 agent steps and 14 → 28 learner updates after refilling replay. Loading/artifact/startup checks: 17 passed. |

## E — Evaluation and experiment workflows

| ID | Capability | Existing code to reuse | SAC adaptation |
|---|---|---|---|
| E1 | Standalone benchmark evaluation | [`eval()`](../../pufferlib/pufferl.py#L1673), [`_run_eval_rollout()`](../../pufferlib/pufferl.py#L2027), `report_eval_to_wandb()` and benchmark/report utilities. | Implemented `eval` CLI and `DriveSAC.forward_eval`, reusing PPO benchmark orchestration with the supplied SAC actor. Actor construction uses the selected benchmark dataset. A real one-scenario CPU evaluation wrote episode CSV and summary JSON, including a run with a nonexistent training-map path. Replay/render options use the SAC capture adapter described in E3. |
| E2 | Evaluation during training | [`run_training_evaluation()`](../../pufferlib/pufferl.py#L2224) and interval/final-evaluation blocks in `train()`. | Implemented `train.evaluation_interval_epochs` and `train.evaluation_benchmarks`, using PPO validation and `run_training_evaluation` unchanged. Rank zero evaluates at intervals and at the final cycle if needed, with global agent steps and existing RNG/mode restoration. A real 20-row CPU run evaluated at epoch 3/step 16 and final epoch 4/step 20. Five focused evaluation tests passed; disabled evaluation preserves injected-config compatibility. Multi-rank scheduling remains statically reviewed. |
| E3 | Export and rendering | [`export()`](../../pufferlib/pufferl.py#L1958), [`_render_eval_failures()`](../../pufferlib/pufferl.py#L2259), [`EvalReplayCapture`](../../pufferlib/ocean/evaluation_utils/eval_replay.py#L20) and `_render_eval_replays()`. | Rendering implemented: [SAC evaluator](../../pufferlib/sac_evaluation.py) mirrors PPO's rendering orchestration; [SACReplayCapture](../../pufferlib/ocean/evaluation_utils/sac_eval_replay.py) omits values and [sac_viz.py](../../pufferlib/sac_viz.py) reuses the viewer without its value panel. Real CPU checks covered scenario capture with observations and replay-from-CSV; Chrome displayed the scene, observation view, entropy and action probabilities without JavaScript exceptions. Seven focused tests passed. The HTML adapter checks shared-template compatibility. Portable Python export implemented in `pufferl_sac.export()` and `load_export()`: actor-only `actor.pt` plus reconstruction `config.yaml`, loaded without starting a simulator or accessing maps. Classic/jerk round trips and existing loading/startup checks passed (15 tests). A real checkpoint export exactly matched the original actor on four real Drive observations and all action mappings. The legacy C visualizer is outside the selected scope. See the command below. |
| E4 | Sweeps and controlled experiments | [`sweep()`](../../pufferlib/pufferl.py#L1856), `controlled_exp()`, `downsample()` and existing search algorithms. | Implemented `sweep` and `controlled_exp` commands in [sac_workflows.py](../../pufferlib/sac_workflows.py), using existing search classes and PPO downsampling. SAC `train()` returns log history. [Workflow config](../../pufferlib/config/sac_workflows.yaml) supplies SAC search spaces and an eight-combination grid; inherited PPO axes are disabled. Trials use unique directories and preserve search RNG across training. A real two-trial CPU sweep completed with distinct parameters and artifacts; focused tests cover the grid and metric observations. These command drivers require one process; ordinary SAC training still supports DDP. |
| E5 | Profiling and autotuning commands | [`profile()`](../../pufferlib/pufferl.py#L1938), [`autotune()`](../../pufferlib/pufferl.py#L2004) and [`vector.autotune()`](../../pufferlib/vector.py#L830). | Implemented `profile` around SAC startup/training with an explicit received-row budget (`profile.total_timesteps`), CPU/CUDA profiler activities and a per-run Chrome trace. A real 20-row CPU profile produced a readable trace containing training events. `autotune` reuses vector throughput tuning with Drive settings and final-observation transport; `vec.batch_size` counts environment instances. A bounded one-environment smoke completed Multiprocessing and Serial candidates. Eight workflow tests passed. These checks validate execution, not representative performance; workflow commands require one process. |

**Delivery order:** complete A1–A4 for a short real Drive training run, then proceed through B (learner options), C (execution), D (run management) and E (evaluation/workflows) in bounded changes. Use focused checks for each change rather than building a large scaffold test suite.

**Completion and validation limits:** all selected implementation items are complete. Replay/environment continuation remains deferred and recurrence was not selected. GPU/NCCL, CUDA BF16/graphs/pinning, live logger providers, live multi-rank evaluation scheduling and multi-worker/multi-wave replay rendering are not runtime-verified here. CPU evidence includes real Drive collection/training, reset-boundary checks, Inductor/eager parity, BF16, two-process training, warmstart/resume, benchmark evaluation, browser rendering, sweeps, profiling, autotuning and actor export. These bounded checks establish execution and numerical contracts, not training quality or production performance.

## Portable actor export

Use the actor checkpoint (`final_model.pt` or a saved actor snapshot), with its run's `config.yaml` available through the existing checkpoint-directory convention:

```bash
source .venv/bin/activate
python -m pufferlib.pufferl_sac export puffer_drive \
  load_model_path=/path/to/run/final_model.pt \
  +export_path=/path/to/portable_actor
```

The output directory contains `actor.pt` and `config.yaml`; existing export files are not overwritten. Without `+export_path`, output goes beside the checkpoint into `<checkpoint_stem>_portable_actor`. Copy both files together. The bundle excludes critics, optimizers, replay and environment state, and drops the source map path. It requires a compatible PufferDrive/PyTorch installation with the Drive binding; it is not a standalone C binary.

```python
import torch
from pufferlib.pufferl_sac import load_export

actor = load_export("/path/to/portable_actor", device="cpu")
# observations: FP32 [B, observation_size], using the saved Drive observation settings.
with torch.inference_mode():
    logits = actor(observations)
    action_ids = logits.argmax(dim=-1)  # deterministic inference
    actions = actor.discrete_actions_to_continuous(action_ids)
```

`actions` are the normalized two-component Drive controls; physical controls are available as `actor.action_table_physical[action_ids]`. Sampling from `Categorical(logits=logits)` gives stochastic actions instead. The plain `actor.pt` plus config also works with existing SAC checkpoint loading; exporting does not alter the original run.

Source of truth: [SAC scaffold](../../pufferlib/pufferl_sac.py) and [PPO route](../../pufferlib/pufferl.py). Archived documents are not maintained.
