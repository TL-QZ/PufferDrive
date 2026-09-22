# Run Condition B

## Commands

Run from the repository checkout; launchers activate `.venv`.

| Task | Command |
|---|---|
| Bounded toy training | `project/jepa_distill/scripts/train_toy.sh` |
| Default training | `project/jepa_distill/scripts/train.sh` |
| Override collection reuse | `project/jepa_distill/scripts/train.sh --set collection.num_collections=2 --set collection.max_transitions=131072 --set training.update_epochs=2` |
| Held-out validation | `project/jepa_distill/scripts/validate.sh --checkpoint experiments/jepa_distill/runs/<run_id>/final_model.pt` |
| Disable remote logging | Append `--wandb-disabled` to a training command. Local metrics remain enabled. |

## Training order

1. Resolve the teacher's saved config and clean observation layout; load and freeze teacher weights.
2. Create the student, EMA target encoder, optimizer, and a dedicated W&B run. Collect a fixed validation set using its own seed.
3. Collect a fresh teacher-driven training round. Index complete windows within agent lifetimes; exclude boundary crossings.
4. Shuffle windows for each `training.update_epochs` pass. Optimize distillation + JEPA + variance losses; update EMA after each successful optimizer step.
5. Run periodic student driving evaluation in separate simulator episodes; repeat collection as configured. Save final checkpoint, held-out losses, and driving results.

## Driving evaluation during training

| Check | What it measures | Default / toy |
|---|---|---|
| Held-out validation | Losses and teacher KL on fixed teacher trajectories | Every 50 / 5 updates |
| Student driving evaluation | Closed-loop driving with the student controlling all policy-controlled vehicles | Every 50 / 20 updates, plus completion |
| Driving budget per evaluation | CARLA scenarios × maximum simulator steps per scenario | 16 × 500 / 2 × 64 |

- Reuses the PPO simulator evaluator and its driving metrics. Executes chunk slot zero with mean-action selection, then observes again; uses native training `dt` and fixed evaluation seed 42.
- Driving metrics go to the same student W&B run and local `metrics.jsonl`, under `eval/training_native/carla/student/*`. Detailed reports go under `eval/carla_training_native/training/step_*` in the student run. Evaluation interactions do not count toward teacher collection budgets.
- Evaluation restores training modes and random-number state. Completion skips a duplicate evaluation when that optimizer step was already evaluated.
- Set cadence with `--set driving_evaluation.interval_steps=100`; disable with `--set driving_evaluation.enabled=false`. Use standalone evaluation for larger final benchmarks.

## Defaults and outputs

| Setting | Default | Toy |
|---|---:|---:|
| Fresh training collections | 1 | 2 |
| Agent transitions per collection | 65,536 | 4,096 |
| Passes over each collection | 1 | 1 |
| Maximum optimizer steps | 1,000 | 64 |
| Held-out agent transitions | 8,192 | 2,048 |
| CPU simulator workers | 2 | 2 |
| Training GPUs | 1 | 1 |
| W&B project | `pufferdrive` | `pufferdrive-jepa-distill-toy` |

- Run artifacts: `experiments/jepa_distill/runs/<run_id>/` — resolved config, `metrics.jsonl`, periodic `checkpoint.pt`, `final_model.pt`, `result.json`.
- Trajectories: `experiments/jepa_distill/datasets/<dataset_id or run_id>/` — fixed validation plus separate training rounds.
- `--run-id NAME` chooses the output folder. Fresh runs reject a nonempty folder.
- Resume with the same run ID and `--set training.resume_checkpoint=PATH`. Stored collection batches can resume; simulator memory is not saved, so subsequent collections restart simulator episodes.
- CPU simulation uses the existing C engine and PufferLib workers. The learner uses PyTorch on one GPU. DDP, AMP, and compile are explicitly rejected until supported and tested.
- Standalone driving evaluation: `scripts/evaluate.sh --config project/jepa_distill/config/evaluation_toy.yaml --checkpoint PATH --wandb-disabled`. A transfer-timestep experiment uses a separate config/output name; the combined `transfer.enabled` switch is rejected.

## Validation status

GPU training, standalone validation, checkpoint resume, and bounded driving evaluation passed. The [online toy run](https://wandb.ai/tobieliu825/pufferdrive-jepa-distill-toy/runs/2l5ioks4) completed 60 updates over two collections; standalone validation reused its W&B identity. See [verification.md](verification.md) for evidence and supported scope.
