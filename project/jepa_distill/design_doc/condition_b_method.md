# Condition B — repeated teacher collection and student training

**Pipeline diagram:** [SVG](figures/condition_b_pipeline.svg) · [PDF](figures/condition_b_pipeline.pdf) · [editable source and regeneration](figures/README.md).

**Status:** training, validation, resume, driving evaluation, and online toy monitoring implemented and verified. Updated 2026-09-19.
**First milestone:** repeat **collect fresh teacher interactions → train student → collect again**, then evaluate. Tune performance after this works.

**Implementation:** [module contracts](implementation_plan.md) · [launch commands](running.md) · [verification evidence](verification.md).

## 1. Agreed design

| Item | Decision |
| --- | --- |
| Teacher | Existing CARLA/Gigaflow self-play PPO; run below. nuPlan teacher deferred until its reported stopping issue is resolved. |
| Student | Per-agent Drive backbone, initialized from teacher actor; new MLP chunk decoder. |
| Prediction / execution | `K=4` actions predicted; execute slot zero and observe again (`H=1`). |
| JEPA | MLP predicts the normalized endpoint latent using the current latent and four executed, logged actions. |
| Target / regularization | EMA **target encoder**, no target decoder; enable variance penalty and collapse diagnostics. |
| First experiment | Faithful imitation; clean training; student controls all policy-controlled cars. Focal-student evaluation is optional later. |

### Collection and reuse schedule

| Control | Meaning | Initial setting |
| --- | --- | --- |
| `collection.num_collections` | Total fresh collection rounds, all driven by the frozen teacher | `1` |
| `collection.transitions_per_round` | Fresh agent-transitions before student updates | Required launch choice |
| `training.update_epochs` | Full shuffled passes over valid windows in each collection | `1` initially; set to the desired training epochs |
| `training.batch_size` | Agent windows per optimizer minibatch | `256` |
| `collection.max_transitions` | Total interaction budget across repeated rounds | Required launch choice |

- Like PPO's collection/update schedule: collect one batch, reuse it for `update_epochs`, then collect fresh interactions. This is supervised distillation, not a PPO policy update.
- Keep the student, optimizer, EMA target, and live simulator across rounds. The teacher remains frozen; each new round advances to new interactions instead of restarting the same seed.
- Train only on the current collection by default. Stored old shards are provenance/restart artifacts, not an accumulating replay buffer. Keep validation/test data fixed and separate.
- Default `num_collections=1`: collect once and train for `update_epochs` full passes. Increase `num_collections` to alternate collection/training repeatedly. Without early budget stops, total training passes are `num_collections × update_epochs`; no separate `total_epochs` setting.

**Selected teacher run:**

```text
experiments/baseline_run_sync_2026-08-24/
  baseline_run_sync_2026-08-24_2026-08-26_16-23-18_seed0/
    final_model.pt
    config.yaml
```

- Both files exist. Use `final_model.pt` as the initial checkpoint choice; strict weight loading remains an implementation check.
- Saved settings: discrete policy, continuous environment interface, jerk dynamics, `dt=0.3`, backbone width `1024`, actor hidden width `256` with one hidden layer.
- Therefore: `C=12` joint jerk classes; `K=4` spans **1.2 seconds**; replan every **0.3 seconds**.
- Teacher training enabled lane/boundary dropout and blind/phantom partners. For clean collection, disable these corruptions and record the overrides. Road dropout changes retained slot counts and observation width: construct teacher/student with the resolved clean layout and verify strict weight compatibility.
- Preserve feature definitions, normalization, and dynamics; derive observation width from the effective layout. Initialize only student backbone weights; never resume the teacher's PPO optimizer/trainer state.

## 2. Model and losses

`B` = sampled agent windows; `D_o` = effective observation width after clean overrides. Rows are independent agent examples, not a scene-wide attention input.

| Component | Input → output | Learning signal |
| --- | --- | --- |
| Frozen teacher | `o[t+k]` → 12 logits, `k=0..3` | None |
| Context encoder `f` | `o[t]` → `z[t]`, width 1024 | All three losses |
| Chunk decoder `d` | `z[t]` → logits `[B,4,12]` | Distillation |
| Predictor `h` | `z[t]` + logged controls `[B,4,2]` → endpoint `[B,1024]` | JEPA |
| Target encoder `g` | Clean `o[t+4]` → target `[B,1024]` | EMA of `f` only |

**Initial architecture defaults:** decoder `1024 → 256 → 48`, ReLU; predictor `1032 → 1024 → 1024 → 1024`, ReLU after hidden layers. These widths are implementation starting points, not tuned results.

| Loss | Definition / purpose |
| --- | --- |
| Distillation | Soft cross-entropy from teacher probabilities to student probabilities, averaged over batch and four slots. Start at `T=1`; expose temperature in config and apply `T²` scaling. |
| JEPA | Squared distance between unit-normalized predicted and target endpoint latents. |
| Variance | Penalize low standard deviation across the batch in each **unnormalized context-latent** dimension. |

$$
\hat z_{t+4}=h(f(o_t),a_{t:t+4}),\qquad
\bar z_{t+4}=\operatorname{stopgrad}(g(o_{t+4})),\qquad
L_J=\operatorname{mean}\|\nu(\hat z_{t+4})-\nu(\bar z_{t+4})\|_2^2
$$

$$
L_V=\frac{1}{D_z}\sum_j\max(0,\gamma-\sqrt{\operatorname{Var}_b(z_{b,j})+\epsilon}),
\qquad L=\lambda_D L_D+\lambda_J L_J+\lambda_V L_V
$$

- `nu(z)=z/max(||z||₂,epsilon)`; use explicit epsilon and population variance (`correction=0`).
- Slot `k` matches the teacher at the recorded `o[t+k]`; the student sees only `o[t]`. Future slots forecast teacher behavior, including its uncertainty.
- Predictor inputs are the **executed controls**, not the student's proposed controls. Use saved action-table normalization; retain physical units in metadata.
- After each successful optimizer update: `g ← tau*g + (1-tau)*f`, initially `tau=0.99`. No target gradients; no EMA update on a skipped optimizer step.
- Deployment uses only `f + d`. JEPA does not directly update `d`; variance is a collapse safeguard, not proof of useful features.

## 3. Data contract

**Collection default:** the frozen teacher drives every round. Sample its categorical distribution, convert through its action table, and store full logits. Mean-action evaluation is specified separately below.

| Stored data | Required rule |
| --- | --- |
| Observations and logits | Save trajectories once; index overlapping windows. Copy simulator buffers before they are overwritten. |
| Executed controls | Store the actual two controls sent to the environment; nominal discrete IDs alone are insufficient for mean execution. |
| Episode/agent identity | Track worker, scene instance, reset generation, agent slot/lifetime, and timestep. No window may cross a lifetime or reset. |
| Validity | Require five valid observations and four transitions. Reject termination/truncation crossings and missing endpoints; no padding. Count rejection reasons. |
| Manifest and splits | Record checkpoint hash, effective config, action tables, code revision, seeds, shapes, counts, and collection mode. Split complete generated episodes by disjoint seeds before indexing windows. |

- Use a live bounded `Drive` collector; stream each round into separate float32 NumPy shards and a manifest. Avoid duplicating overlapping windows. Drop/count incomplete windows at round cutoffs rather than mixing rounds.
- Do not export flattened PPO minibatches: they lack the full teacher distribution and reliable trajectory identity.
- `Drive.step()` returns outcomes of the action just sent. C autoresets and Python map resampling may replace the endpoint; masks are computed before movement. Prove boundary/eligibility alignment before accepting windows.
- Use `agent_offsets`/map grouping and reset events for scene generations. Add a minimal explicit lifetime signal only if existing state cannot establish continuity; never infer it from a reused slot alone.
- CARLA validation initially uses held-out generated episodes on the same eight maps; this is **not** an unseen-map generalization test.
- During distributed training, measure student driving through the existing PPO evaluator after training on each collection: 50 reuse epochs for the full run, 2 for the two-GPU toy. Also evaluate at early completion unless the final weights were already evaluated. Held-out losses measure imitation/representation quality; simulator metrics measure driving. See [cadence and budgets](running.md#training-and-evaluation-order).

## 4. Implementation sequence

All new paths below are under `project/jepa_distill/`. Keep existing PPO behavior unchanged; reuse shared interfaces where they already fit.

| Step | Files / work | Done when |
| --- | --- | --- |
| **1. Teacher + dataset** | `config/condition_b.yaml`, `teacher.py`, `collect.py`, `dataset.py`: resolve saved config, load frozen teacher, stream shards, validate and index windows. | Small seeded collection reproduces logits/controls; windows never cross reset/removal boundaries; invalid config fails before collection. |
| **2. Student + objectives** | `model.py`: backbone copy, decoder, predictor, target encoder, losses and EMA. | Shapes match; student backbone initially matches teacher; gradient ownership and one-step EMA arithmetic verified. |
| **3. Collection/update loop** | `train.py`: repeated teacher collection, `update_epochs` passes, AdamW, validation, collapse logging, checkpoints/resume. | Two rounds use fresh interactions and retain student/optimizer/EMA state; update counts match epochs; replaying a saved batch reproduces the next optimizer update. |
| **4. Evaluation adapter** | `evaluate.py`, `config/evaluation.yaml`: expose slot-zero policy through existing evaluation API and action conversions. | Teacher and student run identical small CARLA scenarios; student controls every policy-controlled vehicle; reports go to the student run. |
| **5. Baselines + tuning** | `tests/`, then `sweep.py` and a bounded search config after the pipeline passes. | Compare teacher, distillation+variance, and full B; identical splits/seeds/budgets. Search ranks held-out results and preserves a final test split. |

**Reuse points:**

| Existing code | Reuse / constraint |
| --- | --- |
| [`ocean/torch.py`](../../../pufferlib/ocean/torch.py): `DriveBackbone`, action conversions | Same observation packing and physical-space mean conversion; derive dimensions from resolved config. |
| [`pufferl.py`](../../../pufferlib/pufferl.py): `load_policy`, `eval(..., policy=...)` | Load teacher weights; inject student adapter rather than teaching the PPO trainer to train B. Adapter supplies expected `forward_eval`, action conversion, and value-output interface; no learned critic needed. |
| [`drive/drive.py`](../../../pufferlib/ocean/drive/drive.py): `step`, `agent_offsets`, resampling | Establish transition/reset contract; test against actual environment behavior. |
| [`evaluation_benchmarks.yaml`](../../baseline_run_sync_2026-08-24/override_config/evaluation_benchmarks.yaml) | Reuse benchmark definitions and metrics. Resolve checkpoint/config overrides explicitly before evaluation. |

- Distributed recipe: two GPUs; 8,192,000 transitions per GPU per collection, 50 reuse epochs, 15 collections, effective batch 128,000 windows per GPU. Accumulate 1,024-window microbatches; variance statistics are local to each microbatch. AdamW `lr=1e-4`, gradient clip `1`, loss weights `(1,1,0.1)`; AMP/compile disabled.
- Save student/target/predictor/decoder, optimizer, RNG/sampler, round index, epoch/minibatch progress, total interactions, and current collection identity. Exact simulator resume is distinct from replaying saved optimizer batches; document any fresh-episode restart. Never resume teacher PPO training.
- Log loss components, teacher KL by slot, latent standard deviations/norms, periodic effective rank, gradient norms, validation metrics, and throughput.
- W&B monitoring: a dedicated student run logs training, validation, representation health, progress, and driving evaluation. Preserve its identity in student checkpoints; never reuse the teacher run. See the [monitoring contract](implementation_plan.md#8-wb-monitoring) and [monitoring.py](../monitoring.py).
- Outputs: `experiments/jepa_distill/datasets/<dataset_id>/` and `experiments/jepa_distill/runs/<run_id>/`; never write student results into the teacher run.
- No C changes planned initially. If a boundary signal requires `.c/.h` edits, rebuild with `python setup.py build_ext --inplace --force` and check existing behavior.

## 5. Evaluation and search

| Check | Protocol |
| --- | --- |
| First success | End-to-end pipeline; held-out action KL and closed-loop faithful imitation. Numerical acceptance gap remains to be chosen. |
| Primary population | All policy-controlled vehicles use the student; preserve replay/heuristic controllers. Focal student among teachers is a later optional comparison. |
| Action selection | Start with mean actions for teacher and student; discrete policy + continuous environment; expectation computed in physical units. |
| Timing | First compare both at native `dt=0.3`. Existing catalog uses `dt=0.1`: report that separately as transfer, preserving episode duration when building native-timestep settings. |
| Outcomes / cost | Collision, off-road, goal/progress metrics with documented aggregation; inference latency. Report unique collected transitions, offline compute, and teacher training cost separately. |

- Keep `K=4`, `H=1`, clean training, and teacher fixed during initial tuning.
- Search learning rate, JEPA/variance weights, EMA momentum, temperature, and MLP widths in bounded stages; compare runs at equal data/update budgets.
- Keep variance enabled in both JEPA/no-JEPA comparisons to isolate JEPA. A variance-off ablation is separate; `K=1` is optional diagnostic, not the main target.
- Begin with held-out KL screening, then CARLA closed-loop ranking. Evaluate nuPlan single/multi later as transfer baselines; corruption training is deferred.

## 6. Remaining launch choices

These do not block implementing the pipeline. Resolve them before a large collection or search.

| Choice | Starting recommendation | Your adjustment |
| --- | --- | --- |
| Collection behavior | Sampled categorical actions; retain soft logits | |
| Budget | Set transitions per round, total transition/disk limits, optimizer-step limit, GPU count and search-trial cap after a bounded storage/throughput check | |
| Success threshold | Choose acceptable teacher-to-student collision/progress gap before ranking search results | |

**Next implementation task:** Step 1 — teacher loading and a small validated trajectory dataset.
