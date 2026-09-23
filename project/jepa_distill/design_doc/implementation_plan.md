# Condition B — implementation contracts

**Current execution recipe:** see [running.md](running.md) for single-/multi-GPU launch commands and [verification.md](verification.md) for dated test and runtime evidence. The contracts below describe the original method; distributed batching and streaming details are in the run guide.

**Read first:** [method decisions](condition_b_method.md), then the two tables below. Configurations: [training/collection](../config/condition_b.yaml) and [evaluation](../config/evaluation.yaml).

**Monitoring:** [W&B contract and metric table](#8-wb-monitoring) and [monitoring.py](../monitoring.py). Training and standalone validation were verified in the same dedicated student W&B run.

## 1. Review map

| Module | Public surface | Responsibility |
| --- | --- | --- |
| [teacher.py](../teacher.py) | `resolve_teacher_config`, `load_teacher` | Resolve clean environment layout; strictly load and freeze the selected PPO teacher. |
| [collect.py](../collect.py) | `collect_dataset` | Collect one fresh round from a caller-owned frozen teacher and live environment; write round-specific shards and manifest. |
| [dataset.py](../dataset.py) | `TrainingSample`, `TrainingBatch`, `WindowReference`, `validate_manifest`, `build_window_index`, `TrajectoryDataset` | Validate metadata; reject invalid windows; read individual windows for batching. |
| [model.py](../model.py) | `ConditionBModel`, `ModelOutputs`, `LossTerms`; model methods for losses and `update_target_encoder` | Context/target encoders, chunk decoder, action-conditioned predictor, losses, and EMA. |
| [train.py](../train.py), [evaluate.py](../evaluate.py) | Train/validate/save/resume; `StudentPolicyAdapter`, `evaluate_student` | Repeated collection/student-update loop; slot-zero policy integration with existing evaluation. |

| Fixed choice | Contract |
| --- | --- |
| Teacher and student | CARLA teacher from the method document; warm-start only its actor backbone. Teacher remains frozen. |
| Temporal window | `K=4`, `H=1`, collection `dt=0.3`: predict 1.2 seconds ahead; replan every 0.3 seconds. |
| Heads | Decoder `1024→256→48`; predictor `1032→1024→1024→1024`; ReLU hidden layers. Four joint categorical distributions, each with 12 classes. |
| Training | Clean observations, soft cross-entropy, normalized endpoint JEPA, variance penalty on unnormalized context latents. |
| Deployment | Context encoder + decoder only; all policy-controlled vehicles use the student. No learned critic or target decoder. |

## 2. Teacher and configuration

- Resolve repository-relative paths against the repository root. Read the selected teacher's `config.yaml`, apply only `env_overrides`, then derive the effective observation layout and action table. Do not infer settings from current generic defaults.
- Removing road dropout changes retained slot counts. Build teacher and student against the same clean layout; check weight keys/shapes strictly. Persist layout metadata so offline training can construct the backbone without a live simulator.
- Teacher loading uses only model weights, sets evaluation mode, and disables gradients. Load the complete Drive state dict for compatibility, but ignore its value output. Student construction copies only actor-backbone weights and initializes a separate EMA copy; the student excludes the PPO critic and optimizer.
- Config files are explicit defaults, not working launchers. Validate required null fields and positive limits before runtime setup. Collection requires `dataset_id`, `transitions_per_round`, total transition/disk limits; training requires fixed `validation_manifest`, `run_id`, step limit and validation/checkpoint intervals; evaluation requires checkpoint, `run_id` and scenario count. `collection.num_collections=1` selects one fresh collection by default; `training.update_epochs` sets full passes over each collection.
- Training supports a legacy single-GPU path and a DDP path with accumulated microbatches; AdamW, no AMP/compile. YAML is authoritative; pass loss/EMA values explicitly. Reject incompatible dimensions, action types, schema, temperature, variance floor, epsilon, loss weights, and EMA momentum.

## 3. Dataset and collection

```text
Frozen teacher + clean Drive environment
  -> fresh round shards + manifest -> current-round window index
  -> TrainingSample -> TrainingBatch -> update_epochs shuffled passes
  -> next teacher collection (retain student / optimizer / EMA / environment)
```

| Tensor | Individual sample | Training batch | Meaning |
| --- | --- | --- | --- |
| `observations` | `[5,D_o]` | `[B,5,D_o]` | Current through endpoint observations, inclusive. |
| `executed_controls` | `[4,2]` | `[B,4,2]` | Actual normalized longitudinal/lateral controls sent to the environment; order and physical jerk scales come from the manifest. |
| `teacher_logits` | `[4,12]` | `[B,4,12]` | Full teacher output at each action's observation, not sampled log-probabilities. |

- Write float32 NumPy shards, storing each trajectory once. The JSON manifest records schema version, checkpoint hash, effective config/layout, physical and normalized action tables, code revision, seeds, counts, and shard paths. `WindowReference(shard_idx, trajectory_idx, start_step_idx)` locates each window; dataset construction and indexing require an explicit split.
- Preserve scene instance, reset generation, agent lifetime, timestep, eligibility, and termination/truncation events. Accept only four continuous transitions with five valid observations within a round. Reject boundary crossings or missing endpoints; do not pad or substitute reset observations. Drop/count incomplete round tails.
- Copy zero-copy environment arrays before the next step. Align each returned transition outcome with the action just executed. C masks are computed before movement; prove endpoint eligibility using actual lifecycle events, not the mask alone.
- Collection samples teacher categories and converts through the teacher's action table. Save actual continuous controls and full logits. Clean collection disables road dropout and blind/phantom behavior, while retaining dynamics and feature definitions.
- Prepare fixed validation/test episodes using disjoint seeds; only training collections refresh. Continue the live training environment between rounds without repeating its seed. Episode/split identity remains stable even when an episode spans collection rounds; windows stay round-local. Track held-out collection cost separately. This tests new episodes on the same maps, not unseen maps.

**Retention:** use only current-round windows for optimization; archived shards do not enter later rounds automatically. Count all stored shards toward the disk cap and stop before exceeding it; no implicit deletion. Shrink the final collection to the remaining transition budget; fail clearly if it contains no valid windows. An optimizer-step cap may stop partway through a reuse epoch, with position recorded.

**Boundary gate:** real vector collection and focused tests verify masks and reset/termination rejection. Collection requires stable vector-slot ordering and rejects unsupported ordering rather than stitching different agents together.

## 4. Model, losses, and EMA

```text
observations[:,0] -> f -> context latent -> decoder -> [B,4,12] logits
                          |
executed_controls --------+-> predictor -> predicted endpoint
observations[:,4] ------------> g (EMA, no gradients) -> target endpoint
```

| Component / objective | Exact contract | Gradient recipients |
| --- | --- | --- |
| Distillation | `T² * mean[-sum(softmax(teacher/T) * log_softmax(student/T))]`; sum over classes, mean over batch/time. Teacher labels detached. | Context encoder, decoder |
| JEPA | Unit-normalize predicted/target vectors with epsilon; sum squared difference over latent dimensions, mean over batch. Target detached. | Context encoder, predictor |
| Variance | Mean `relu(gamma - sqrt(var(context, dim=batch, correction=0) + epsilon))`; use unnormalized latents and a batch of at least two samples. | Context encoder |
| Combined loss | `lambda_D*distillation + lambda_J*jepa + lambda_V*variance`. `LossTerms` keeps all components as scalar tensors. | As above |
| EMA | `target = tau*target + (1-tau)*online`, once after each successful optimizer step. Freeze target; keep it in evaluation mode even when student is training. Copy any non-parameter buffers explicitly. | None |

- Predictor receives logged controls, not predicted actions. `ModelOutputs` contains context latents `[B,1024]`, chunk logits `[B,4,12]`, predicted endpoint `[B,1024]`, and target endpoint `[B,1024]`.
- Current/future teacher labels differ: chunk slot `k` matches the teacher at `o[t+k]`, while the decoder sees only `o[t]`. No future observations enter the decoder.
- EMA does not guarantee useful representations. Track per-dimension standard deviation, latent norms, periodic effective rank, and teacher KL by slot.
- Constructors and forwards remain stubs. Importing modules creates no neural networks and imports no Drive bindings at runtime; runtime-specific types belong behind `TYPE_CHECKING`.

## 5. Repeated collection and checkpoint lifecycle

1. Construct teacher, student, optimizer, EMA target, and training environment once. `train` receives fixed validation batches and owns repeated collection rather than receiving one fixed training iterable.
2. Collect `transitions_per_round` fresh teacher-driven interactions, persist their round identity, and index complete windows. Do not update teacher weights.
3. Run `training.update_epochs` full shuffled passes over current-round windows, using `training.batch_size` minibatches. Require at least two valid windows; merge a final singleton into the preceding minibatch so variance remains defined without dropping data. Apply AdamW and then EMA once per successful update.
4. Validate/log/checkpoint at configured optimizer-step intervals. Also run student driving evaluation through `evaluate_student` at `driving_evaluation.interval_steps` and completion, using separate simulator episodes and the training-owned monitor. Restore RNG and module modes; retain the last evaluated step/results in checkpoints to avoid duplicate final evaluations. Repeat until collection rounds or budget caps finish.
5. Save model/optimizer/RNG, sampler, round index, total interactions, reuse-epoch/minibatch position, current manifest identity, collector progress, and monitoring state. Resume saved-current-round updates exactly; after them, restart collection at a logged fresh episode unless simulator state serialization has been implemented. Do not claim RNG restores simulator state; never use teacher `trainer_state.pt`.

**Reuse example:** 1,024 valid windows, batch size 256, `update_epochs=1` gives four optimizer steps before the next collection. `update_epochs=3` gives twelve steps over that same collection. Collection rounds, reuse epochs, and optimizer steps are distinct counters. The schedule mirrors [`pufferl.py`](../../../pufferlib/pufferl.py)'s outer `evaluate()`/`train()` loop and inner `update_epochs` loop; B still uses distillation/JEPA/variance losses rather than PPO.

**Default single collection:** `num_collections=1, update_epochs=10` means collect once and make ten training passes, then finish. `num_collections=3, update_epochs=10` means three fresh collections with ten passes each. Budget caps can end either run early. In existing PPO logs, the outer `epoch` counter counts collection/update cycles; it is not the inner `update_epochs` pass count.

**Outputs:** dataset artifacts under `experiments/jepa_distill/datasets/<dataset_id>/`; student artifacts under `experiments/jepa_distill/runs/<run_id>/`. Fresh runs must not overwrite an existing run; resume requires an explicit Condition B checkpoint.

## 6. Evaluation adapter

| Interface | Future behavior |
| --- | --- |
| `StudentPolicyAdapter.student` | Registered student model; avoid `.module` because `pufferl.base_policy` unwraps that attribute. |
| `is_continuous = False` | Policy predicts categorical logits even though the environment accepts continuous controls. |
| `forward_eval(observations, state=None)` | Return a one-element tuple containing slot-zero logits `[B,12]`, plus unused zeros `[B,1]` for evaluator compatibility. Reject non-null recurrent state. Stub raises now; no fake outputs. |
| Action conversion methods | Preserve Drive's class-to-control table; for mean actions average physical controls before piecewise normalization. |
| `evaluate_student` | Construct model/adapter externally; call the existing evaluator with the injected policy and explicit student output path. Avoid PPO checkpoint loading for a Condition B checkpoint. |

- Resolve benchmark environment from student metadata and the catalog; apply evaluation overrides last. Use the evaluator's live-policy/config path and disable reporting into the teacher's W&B run. Clean overrides must survive that path's training-config dropout merge.
- Primary benchmark is CARLA at `dt=0.3`; transform scenario and resampling step counts to preserve the catalog's duration. Full CARLA's `6000 × 0.1 s` becomes `2000 × 0.3 s`. Reject a non-integral conversion rather than silently rounding.
- Apply identical scenarios, seeds, mean-action selection, controller population, and timestep to teacher/student comparisons. Native and optional `dt=0.1` transfer scores use distinct output names; preserve replay timing for any later nuPlan benchmark.
- Rendering, focal-student populations, corruption experiments, and sweeps are deferred. CLI launchers are implemented; no additional public PufferLib registration is needed.

## 7. Checks and next implementation order

| Stage | Required check before continuing |
| --- | --- |
| **Configuration + imports** | Python syntax/imports, YAML parsing, launch commands; package import creates no simulator or model. |
| **Teacher + dataset** | Strict weight load with clean layout; known executed controls/logits reproduced; boundary windows rejected; disjoint split membership. |
| **Model + objectives** | Shapes, frozen/detached targets, expected gradient recipients, scalar loss reference calculations, and one-step EMA arithmetic. |
| **Training** | Two collections advance to fresh interactions; teacher unchanged, student/optimizer/EMA retained; epochs give expected update counts; saved-batch resume reproduces the next update; validation stays held out. |
| **Evaluation** | Adapter conversion matches teacher action semantics; valid controls at `H=1`; matched native-timestep scenarios; no writes/reporting to teacher outputs. |

**Verification evidence:** [verification.md](verification.md). Large production budgets and acceptable teacher-to-student performance gaps remain experiment decisions.

## 8. W&B monitoring

**Ownership:** one new student W&B run per training run or future search trial, retained across all collections and reuse epochs. Never initialize logging from the teacher's W&B metadata. Store the teacher checkpoint hash as provenance only.

| Placeholder | Contract |
| --- | --- |
| `MetricProgress` | Successful optimizer steps, training simulator transitions, collection-round index, reuse-epoch index. |
| `WandbMonitor(...)` | Initialize a new student run or resume its checkpoint identity; record sanitized resolved config and provenance. Lazy SDK import only when enabled. |
| `log_metrics(metrics, progress=...)` | Detached scalar training, validation, representation, throughput, and progress metrics. |
| `log_evaluation(..., benchmark_name, output_name, policy_role, progress)` | Driving results labeled by benchmark, timing protocol, and student/teacher role; all belong to the student run. |
| `state_dict()` / `finish(exit_code=...)` | Checkpoint logging identity/cursor; flush and close only caller-owned sessions, including failure cleanup. |

### Metrics and cadence

| Family | Keys / values | When |
| --- | --- | --- |
| Training | `train/loss_total`, `train/loss_distillation`, `train/loss_jepa`, `train/loss_variance`, `train/learning_rate`, `train/gradient_norm` | Every `wandb.log_interval_steps=10`; also first/final update. |
| Validation | `validation/loss_*`, `validation/teacher_kl/slot_0` through `slot_3`, aggregate teacher KL | Every configured validation interval and final validation; KL compares original distributions at `T=1`. |
| Representation | `representation/latent_std_mean`, `latent_std_min`, `latent_norm_mean`, `effective_rank`; teacher/student entropy | Every `wandb.diagnostics_interval_steps=100` on a bounded diagnostic batch. |
| Progress / speed | `progress/optimizer_step`, `simulator_transitions`, `collection_round_idx`, `update_epoch_idx`; `throughput/collection_transitions_per_second`, `training_windows_per_second`; valid/rejected window counts and collection/training duration | Progress attached to every event; collection statistics once per round; update speed at logging intervals. |
| Driving | `eval/{output_name}/{benchmark}/{policy_role}/{metric}`: collision, off-road, goal/progress, inference latency where produced by the evaluator | Only after actual simulator evaluation. Never fill unavailable metrics with zeros. |

- Prefix all entries in a family consistently. Log losses as sample-weighted means over their reporting interval or validation set; document gradient norm as pre-clip. Log scalars detached from autograd, not live graph tensors. Nonfinite training scalars are explicit errors.
- Use `progress/optimizer_step` as the default scientific x-axis for training/validation. Also log cumulative training simulator transitions for sample-efficiency charts; held-out collection and evaluation costs are separate. Reusing a window never increments simulator transitions.
- Keep W&B's log-event cursor separate from scientific counters: collection, validation, and evaluation may share an optimizer step, and evaluating an older checkpoint uses that checkpoint's step. Include a monotonically increasing event index and configure metric axes explicitly when the logger is implemented.
- Log hyperparameters, loss weights, EMA momentum, teacher checkpoint hash, effective observation layout, and validation dataset identity in the run config. Training remains the logger owner during periodic validation/evaluation; called functions must not finish its run.
- W&B is imported only when enabled; disabled runs write local JSONL metrics without SDK/network use.

### Configuration and resume

| Setting / event | Intended behavior |
| --- | --- |
| Fresh training | `wandb.enabled: true`, `mode: online`, project `pufferdrive`, group `jepa_distill`; display name defaults to local `training.run_id`. Generate a distinct W&B ID and save resolved entity/project/ID. |
| Disabled / offline | `enabled: false` or `mode: disabled` disables logging and SDK dependency. Offline mode explicitly stores local events; document session segments and sync state instead of claiming remote resume is verified. |
| Student checkpoint | `monitoring_state` stores run identity, mode, last event cursor, and scientific counters. Online resume requires the saved run; do not silently create a different run or overwrite identity using current config. |
| Resume after interruption | Remote history may be ahead of the checkpoint. Reconcile its cursor and label replayed work; no exactly-once guarantee. Configuration/identity mismatches fail clearly. API keys never enter configs or checkpoints. |
| Standalone evaluation | Obtain identity from the **student checkpoint** and attach to that run; missing identity fails when logging is enabled. Explicitly disable W&B for local-only evaluation. An injected training monitor is reused and never closed by evaluation. |

**Future verification:** use an injected/mock SDK to check fresh versus resumed IDs, disabled mode with no SDK, train/validation events sharing a step, offline segmentation, checkpoint identity round-trip, evaluation namespaces, and owner-only cleanup. No live account is needed for these tests. Current verification checks imports, signatures, config parsing, and explicit stub failures only.
