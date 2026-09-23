# Verification — 2026-09-19

## Multi-GPU integration — 2026-09-22

| Check | Evidence | Result |
|---|---|---|
| Full test suite | `OMP_NUM_THREADS=1 python -m pytest project/jepa_distill/tests -q` | **56 passed** |
| Distributed tests | CPU/Gloo: accumulation reference, rank/EMA parity, unequal window counts, RNG/sampler restoration, rank-zero ownership | **5 passed** |
| Streaming tests | Boundaries, round continuity, memmap batching, disk guards | **4 passed** |
| Online two-GPU toy | `CUDA_VISIBLE_DEVICES=0,1 scripts/train_toy_2gpu.sh --run-id toy_2gpu_online_20260922_01` (scripts under `project/jepa_distill/`) | Exit 0; 2 collections per rank, 16,384 global transitions, 8 synchronized updates |
| Driving evaluation during training | Steps 2, 4, 6, 8; 2 scenarios × 64 steps each | CSV/JSON reports written; no duplicate completion evaluation |
| W&B remote verification | [Run wanv35un](https://wandb.ai/tobieliu825/pufferdrive-jepa-distill-toy/runs/wanv35un) API | Finished; training, validation, and four driving metric events confirmed |
| Standalone streaming validation | `validate.sh --checkpoint .../final_model.pt --wandb-disabled` | Exact match to all final training-validation metrics; total loss 3.562208356528447 |
| Two-GPU completed-checkpoint resume | Same toy launcher/run ID; resume `final_model.pt` with cap 8 | Exit 0; same W&B identity, no additional updates or driving evaluations |
| Full launcher | `DRY_RUN=1 scripts/train_2gpu.sh --run-id full_2gpu` | Correct two-process command; official training **not launched** |

- Run directory: `experiments/jepa_distill/runs/toy_2gpu_online_20260922_01/`.
- Rank-local data and held-out manifest paths are recorded in the checkpoint. Memory-mapped `.npy` arrays replace full-round Python buffers; collections are retained.
- Both RTX A6000 GPUs ran the configured 1,024-window microbatch. Full collection size, 40 simulator instances, and sustained throughput were not benchmarked by the toy.
- The toy proves execution and reporting, not driving quality: its final reported score was 0 and offroad rate 0.84375 after only eight updates.
- Real resume verification used a completed checkpoint; it did not benchmark long-run recovery or compare an interrupted multi-GPU training trajectory bit-for-bit.
- NCCL emitted a nonfatal device-selection warning; the job and resume both completed successfully.

## Periodic driving integration — 2026-09-20

- Full JEPA suite: **47 passed**, including **9 periodic-driving tests**.
- Verified interval/completion scheduling, final deduplication, resume state, shared monitor identity, native runtime settings, and RNG/module-mode restoration on failure.
- Explicit evaluation horizons now override training lengths and bypass conversion of the catalog's old duration.
- Evaluator calls were mocked for these integration tests. No new native simulator/GPU run was launched; the earlier standalone simulator result below predates this integration.

## Earlier runtime verification

| Check | Evidence | Result |
|---|---|---|
| Unit + orchestration tests | `OMP_NUM_THREADS=1 python -m pytest project/jepa_distill/tests -q` | 38 passed |
| Real teacher loading | Saved baseline checkpoint, clean 1,292-feature observations | Strict load; all teacher parameters frozen |
| Model reconstruction | `ConditionBModel(student.export_metadata())` + saved state | Passed without constructing a simulator |
| Repeated real collection | Two CPU workers, 64 agent slots, 2 × 4,096 transitions | 3,840 valid windows per round; 64 truncated transitions excluded per round |
| GPU training launcher | `scripts/train_toy.sh --run-id toy_local_check_20260919_01 --wandb-disabled` | 60 optimizer updates; 8,192 training transitions |
| Held-out validation | Separate seed, 2,048 transitions, 1,856 windows | Loss 3.873809 at step 5 → 2.928242 at step 60 |
| Standalone validation launcher | `scripts/validate.sh --checkpoint experiments/jepa_distill/runs/toy_local_check_20260919_01/final_model.pt --wandb-disabled` | All returned validation metrics exactly match training's final validation |
| Real checkpoint resume | Resume periodic step-60 checkpoint with optimizer cap 60 | Reused stored datasets; identical validation metrics; reconciled local logging history |
| W&B online toy job | [Run 2l5ioks4](https://wandb.ai/tobieliu825/pufferdrive-jepa-distill-toy/runs/2l5ioks4), launched with explicit destination approval | Finished; remote API confirmed 60 updates, 8,192 transitions, final validation loss 2.928242 |
| Online standalone validation | `scripts/validate.sh --checkpoint experiments/jepa_distill/runs/toy_e2e_20260919_01/final_model.pt` | Passed; 1,856 windows, identical metrics; resumed the same W&B run and synced event 88 |
| Closed-loop driving evaluation | `scripts/evaluate.sh --config project/jepa_distill/config/evaluation_toy.yaml --checkpoint experiments/jepa_distill/runs/toy_local_check_20260919_01/final_model.pt --wandb-disabled` | Passed: 2 scenarios × 64 steps, two workers; CSV and JSON written |

## Test coverage

- Loss reference calculations, gradient recipients, frozen teacher/target, and EMA arithmetic.
- Boundary rejection, actual controls, masks, stable vector slots, finite data, disk limits, and split metadata.
- Two collection rounds retain the learner and optimizer; interrupted batch replay reproduces exact student weights and window order.
- Validation leaves learner state and RNG unchanged; singleton remainder is merged without losing windows.
- Dataset identity rejects different teacher weights, feature layout, split, timestep, or action selection. Logging uses an independent event cursor and preserves student run identity.

## Local artifacts

- Online toy: `experiments/jepa_distill/runs/toy_e2e_20260919_01/`; checkpoint, local metrics, `result.json`, and `standalone_validation.json`. W&B synced no model or trajectory artifacts.
- Run: `experiments/jepa_distill/runs/toy_local_check_20260919_01/`.
- Metrics: `metrics.jsonl`; validation report: `standalone_validation.json`; student checkpoint: `final_model.pt`.
- Data: `experiments/jepa_distill/datasets/toy_local_check_20260919_01/`.
- Driving results: `eval/carla_toy_carla_dt03/standalone/` beneath the run folder.

## Scope

- Verified on one GPU with two CPU simulation workers. DDP, AMP, and compilation are not enabled in the student trainer.
- Small-run loss reduction verifies optimization and data flow; it does not establish driving quality or sample-efficiency superiority.
- This executor requires approved host GPU access for shell launchers because its filesystem sandbox hides NVIDIA device nodes. Ordinary GPU-capable terminal execution uses the same launchers.
