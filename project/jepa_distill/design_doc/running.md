# Run Condition B

## Launch

Run from the repository root; scripts activate `.venv`. The full job is started by the user.

| Task | Command |
|---|---|
| Two-GPU toy with W&B | `CUDA_VISIBLE_DEVICES=0,1 project/jepa_distill/scripts/train_toy_2gpu.sh --run-id YOUR_TOY_RUN` |
| Full two-GPU training | `CUDA_VISIBLE_DEVICES=0,1 project/jepa_distill/scripts/train_2gpu.sh --run-id YOUR_FULL_RUN` |
| Resume | Same launcher and run ID, plus `--set training.resume_checkpoint=experiments/jepa_distill/runs/YOUR_RUN/checkpoint.pt` |
| Single-GPU legacy toy | `project/jepa_distill/scripts/train_toy.sh` |
| Local logging only | Append `--wandb-disabled`; `metrics.jsonl` remains enabled |

## Two-GPU recipe

All collection and effective batch counts below are **per GPU**. Both ranks perform synchronized optimizer updates.

| Setting | Full | Toy |
|---|---:|---:|
| Drive instances × agent slots | 20 × 3,200 | 2 × 32 |
| Transitions per fresh collection | 8,192,000 | 4,096 |
| Maximum collections | 15 | 2 |
| Passes over each collection | 50 | 2 |
| Effective windows per optimizer update | 128,000 | 2,048 |
| Microbatch windows | 1,024 | 1,024 |
| Optimizer update cap | 48,000 | 16 |
| Retained dataset budget per GPU | 1 TiB | 1 GiB |
| W&B project | `pufferdrive` | `pufferdrive-jepa-distill-toy` |

- Full job: 16,384,000 transitions per collection and 245,760,000 across 15 collections, globally. Valid windows and partial batches determine actual updates.
- A full effective update accumulates 125 microbatches per GPU, then synchronizes gradients. EMA updates once per optimizer update.
- The variance penalty uses **local microbatch statistics**, not statistics across the 256,000-window global batch. Microbatch size is part of the scientific recipe.
- Collection writes memory-mapped arrays directly to disk; training loads only the current microbatch. Old collection files are retained for provenance/resume.
- Both ranks train over the pooled collection using disjoint shuffled samples. Equal-length rank partitions may repeat a small number of samples at the epoch tail; report this count explicitly.

## Training and evaluation order

1. Load the frozen teacher on each GPU; synchronize the student and create one W&B run on rank zero.
2. Collect a fixed held-out dataset with bounded validation workers. Each rank collects fresh training interactions using distinct seeds and its own simulator workers.
3. Shuffle the pooled windows; optimize distillation + JEPA + variance losses in accumulated microbatches. Repeat for the configured training passes.
4. Rank zero evaluates student driving **after training on each collection**: 50 passes for the full run, 2 for the toy. The other rank waits. Repeat fresh collection as configured.
5. Evaluate at completion unless the same weights were just evaluated; save the final checkpoint and driving results.

| Check | Full | Toy |
|---|---:|---:|
| Held-out loss validation | Every 50 optimizer updates | Every 2 updates |
| Driving evaluation cadence | After 50 passes: once per collection | After 2 passes: once per collection |
| Driving benchmark | `carla_fast` (teacher PPO benchmark) | `carla` |
| Driving scenarios × maximum steps | 250 × 500 | 2 × 64 |
| Evaluation simulator workers | 20 on rank zero | 2 on rank zero |

- Driving uses the existing PPO evaluator with student self-play, chunk slot zero, mean actions, native `dt=0.3`, and fixed seed 42.
- W&B/local metric namespace: `eval/training_native/<benchmark>/student/*`. Reports: `eval/<benchmark>_training_native/training/step_*` under the student run.
- Validation/evaluation interactions do not count toward training collection budgets. Evaluation restores model modes and RNG state; it does not update weights.
- Disable driving evaluation with `--set driving_evaluation.enabled=false`. The legacy single-GPU trainer uses `interval_steps` instead.
- Keep `driving_evaluation.interval_update_epochs` equal to `training.update_epochs` when changing the number of passes, to retain evaluation once per collection. Early termination still evaluates the final student; an already-evaluated final step is not repeated.

## Outputs and resume

- Student run: `experiments/jepa_distill/runs/<run_id>/` — configuration, metrics, checkpoints, result JSON, and driving reports.
- Data: `experiments/jepa_distill/datasets/<run_id>/` (or explicit `collection.dataset_id`) — separate rank collection roots and held-out data.
- Resume requires the same world size and scientific recipe. Checkpoints retain each rank's RNG and shared sampler position. Simulator memory is not checkpointed; subsequent fresh collections restart episodes.
- AMP and compilation remain disabled. Use the same two-GPU launcher for resume.

Verification evidence and runtime limits are recorded in [verification.md](verification.md).
