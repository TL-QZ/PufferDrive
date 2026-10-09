# SAC implementation plan — selected capabilities

Updated 2026-10-09. This plan contains the 20 selected gaps, numbered consecutively within each section. Previous learner options C1/C2/C3/C5 are now B1–B4; previous execution items B1/B3/B4/B5 are now C1–C4. Progress: A1 networks and A2 storage are implemented and CPU-tested; component startup remains A4. Other items remain open.

**Current foundation:** collection and random replay refresh, shuffled full-replay epochs, SAC updates, logging/dashboard, LR scheduling and learner checkpoints work with supplied components. A small CPU smoke passed; real Drive training is not yet validated. Keep the existing PPO-style structure and naming.

**Reuse approach:** reuse existing utilities directly or copy the corresponding code with necessary SAC adaptations. Preserve existing names and structure; identify where new SAC behavior is required.

## A — Basic Drive integration

| ID | Capability | Existing code to reuse | SAC adaptation |
|---|---|---|---|
| A1 | Drive actor and twin critics | [`DriveBackbone`](../../pufferlib/ocean/torch.py#L17), [`Drive`](../../pufferlib/ocean/torch.py#L311), its observation encoding and discrete-action conversion methods, and [`load_policy()`](../../pufferlib/pufferl.py#L2367). | Reuse the encoder and action mapping. Implemented logits-only `DriveSAC` and independent twin-Q `DriveSACCritic` in [torch_sac.py](../../pufferlib/ocean/torch_sac.py). Four CPU tests cover gradients, parameter ownership and PPO action-map parity. Startup wiring remains A4. |
| A2 | Replay storage | [PPO tensor allocation](../../pufferlib/pufferl.py#L179), indexed minibatch reads in `_train_ppo_transition()`, and [SAC replay insertion/replacement](../../pufferlib/pufferl_sac.py#L253). | Implemented [TensorReplay](../../pufferlib/sac_replay.py) with copied fixed-capacity storage behind `len`, `add`, `replace` and `get_batch`. Eight CPU tests cover ownership, replacement, ordering and capacity. Startup wiring remains A4; refresh/epoch scheduling already exists. |
| A3 | Reset-boundary transport | [`Drive.step()`](../../pufferlib/ocean/drive/drive.py#L655), [C truncation/reset path](../../pufferlib/ocean/drive/drive.h#L4609), vector transport in [vector.py](../../pufferlib/vector.py), and [`PuffeSAC.evaluate()`](../../pufferlib/pufferl_sac.py#L253). | Extend these existing paths to transport pre-reset final observations with correct slot alignment and masks/terminal flags. This transport is new work; replace SAC's truncation error with correct bootstrapping, not PPO's approximate workaround. |
| A4 | Configuration and startup | [`load_config()`](../../pufferlib/pufferl.py#L2409), `main()`, [`load_env()`](../../pufferlib/pufferl.py#L2356), [`train()`](../../pufferlib/pufferl.py#L1432), `_get_git_metadata()` and `_save_experiment_config()`. | Preserve Hydra/CLI and run-setup structure; add SAC settings, construct actor/critics/replay and select `PuffeSAC`. Save resolved configuration and metadata using existing helpers. |

## B — Learner options

| ID | Capability | Existing code to reuse | SAC adaptation |
|---|---|---|---|
| B1 | Gradient accumulation | [Minibatch/accumulation setup](../../pufferlib/pufferl.py#L219) and [backward/step block in `_train_ppo_transition()`](../../pufferlib/pufferl.py#L813). | Apply microbatch accumulation to SAC's separate critic, actor and temperature updates. Preserve update ordering and handle partial groups with correct weighting; update targets once per completed learner update. |
| B2 | Gradient clipping | [`clip_grad_norm_` calls before optimizer steps](../../pufferlib/pufferl.py#L818). | Reuse clipping for actor and critic parameter groups after accumulation and before their optimizer steps; expose the configured norm limit. |
| B3 | Optimizer flexibility | [Adam/AdamW/Muon selection in `PuffeRL.__init__()`](../../pufferlib/pufferl.py#L252). | Reuse optimizer settings and selection for actor/critics where compatible. Keep a suitable separate optimizer for scalar log-temperature and checkpoint all optimizer states. |
| B4 | Automatic batch sizing | [Automatic batch/horizon sizing](../../pufferlib/pufferl.py#L154) and [minibatch-size setup](../../pufferlib/pufferl.py#L219). | Reuse configuration resolution and size checks, but define SAC sizing without PPO's rollout-horizon assumption. `batch_size` remains replay capacity; choose replay/minibatch defaults accordingly. |

## C — Execution and memory

| ID | Capability | Existing code to reuse | SAC adaptation |
|---|---|---|---|
| C1 | DDP | [Distributed setup in `train()`](../../pufferlib/pufferl.py#L1490), `derive_rank_seeds()`, `dist_sum()`, `dist_mean()` and rank-zero logging/checkpoint gates. | Reuse process/device/seed setup. Synchronize actor, critics and temperature updates; define per-rank replay ownership, aligned update counts, target updates and global counters. More than replacing the trainer class is required. |
| C2 | Mixed precision | [Precision checks and storage setup](../../pufferlib/pufferl.py#L167), [AMP context construction](../../pufferlib/pufferl.py#L303) and forward/loss autocast usage. | Reuse configured BF16 autocast; apply it to actor/critic forwards while retaining appropriate FP32 probability, target and temperature calculations. |
| C3 | Compilation | [Compile block in `PuffeRL.__init__()`](../../pufferlib/pufferl.py#L239), `base_policy()`, state-key cleanup and CUDA graph step markers in `train()`. | Apply the existing compile options to SAC actor/critic paths and retain uncompiled modules for checkpoints/targets. Verify separate update phases and tail minibatches. |
| C4 | CPU storage/offload | [Pinned CPU observation allocation](../../pufferlib/pufferl.py#L179), collection copies in `evaluate()`, and [indexed nonblocking minibatch transfer](../../pufferlib/pufferl.py#L788). | Apply storage placement and transfer controls to replay observations and next observations, with consistent placement for other fields. SAC already accepts CPU replay tensors; connect pinning/nonblocking transfers. |

## D — Run management

| ID | Capability | Existing code to reuse | SAC adaptation |
|---|---|---|---|
| D1 | Logger construction | [Logger classes](../../pufferlib/pufferl.py#L1257) and the logger-selection block in `train()`. | Reuse W&B, Neptune and TensorBoard classes; connect configuration and rank-zero construction. Supplying a compatible logger already works. |
| D2 | Architecture-aware loading and fine-tuning | [`load_policy()`](../../pufferlib/pufferl.py#L2367), configuration restoration in `train()`, and [`load_checkpoint_architecture()`](../../pufferlib/ocean/evaluation_utils/evaluation_utils.py#L203). | Retain configuration/artifact loading; construct compatible SAC networks and observation/action settings. Distinguish weight warm-start from learner resume; restore SAC critics, temperature and multiple optimizers through its own state loader. |
| D3 | Automatic same-run checkpoint discovery | [Checkpoint-discovery block in `train()`](../../pufferlib/pufferl.py#L1544). | Copy discovery logic and call SAC's existing `load_training_state()`, preserving explicit-path precedence and adding a fresh-run control. |

## E — Evaluation and experiment workflows

| ID | Capability | Existing code to reuse | SAC adaptation |
|---|---|---|---|
| E1 | Standalone benchmark evaluation | [`eval()`](../../pufferlib/pufferl.py#L1673), [`_run_eval_rollout()`](../../pufferlib/pufferl.py#L2027), `report_eval_to_wandb()` and benchmark/report utilities. | Preserve benchmark orchestration and reporting. Adapt actor loading, `forward_eval` output and action mapping; the existing rollout expects `(logits, value)`. |
| E2 | Evaluation during training | [`run_training_evaluation()`](../../pufferlib/pufferl.py#L2224) and interval/final-evaluation blocks in `train()`. | Connect the E1-compatible actor, SAC counters, logger and run directory; retain evaluation scheduling and RNG/mode restoration. |
| E3 | Export and rendering | [`export()`](../../pufferlib/pufferl.py#L1958), [`_render_eval_failures()`](../../pufferlib/pufferl.py#L2259), [`EvalReplayCapture`](../../pufferlib/ocean/evaluation_utils/eval_replay.py#L20) and `_render_eval_replays()`. | Preserve rendering flow. Adapt capture's PPO value-field dependency and verify exported actor parameter ordering/layout against the binary consumer. |
| E4 | Sweeps and controlled experiments | [`sweep()`](../../pufferlib/pufferl.py#L1856), `controlled_exp()`, `downsample()` and existing search algorithms. | Call SAC `train()`, return the expected metric history, and configure SAC hyperparameter search spaces. Keep search/grid mechanics. |
| E5 | Profiling and autotuning commands | [`profile()`](../../pufferlib/pufferl.py#L1938), [`autotune()`](../../pufferlib/pufferl.py#L2004) and [`vector.autotune()`](../../pufferlib/vector.py#L830). | Keep the profiling loop; construct SAC and its dependencies, correcting stale configuration/constructor calls. Reuse environment throughput tuning with the appropriate environment configuration; it does not search SAC learning hyperparameters. |

**Delivery order:** complete A1–A4 for a short real Drive training run, then proceed through B (learner options), C (execution), D (run management) and E (evaluation/workflows) in bounded changes. Use focused checks for each change rather than building a large scaffold test suite.

**Next implementation topic:** A3 — final-observation transport, followed by A4 startup wiring.

Source of truth: [SAC scaffold](../../pufferlib/pufferl_sac.py) and [PPO route](../../pufferlib/pufferl.py). Archived documents are not maintained.
