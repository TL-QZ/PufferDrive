# Distillation-only ablation without warm start

| Setting | Value |
|---|---|
| Config | `project/jepa_distill/config/condition_b_distill_only.yaml` |
| Student encoder | Random initialization; EMA target starts as its copy |
| Loss weights: distillation / JEPA / variance | **1 / 0 / 0** |
| Teacher | Existing pretrained PPO; drives collection and supplies action labels |
| Collection | 15 rounds; 8,192,000 agent-timesteps per GPU per round |
| Training | 50 epochs per collection; 128,000 windows/update/GPU; microbatch 1,024 |
| Evaluation | Inherited held-out validation and driving evaluation after each collection |
| Run ID | `condition_b_distill_only_no_warm_start` |

- Inherits `condition_b.yaml`; future base recipe changes also apply here.
- Fresh run (`resume_checkpoint: null`); run ID separates checkpoints and collection caches from other experiments.
- W&B stays enabled. Zero weights remove JEPA/variance contributions to training; auxiliary computations and diagnostic metrics can still run.
- This jointly removes warm start and both auxiliary losses; it does not isolate their individual effects.

## Launch

```bash
# Inspect the command first; no simulation or W&B upload.
DRY_RUN=1 CUDA_VISIBLE_DEVICES=0,1 project/jepa_distill/scripts/train_distill_only_2gpu.sh

# Start training explicitly.
CUDA_VISIBLE_DEVICES=0,1 project/jepa_distill/scripts/train_distill_only_2gpu.sh
```

Use `--run-id ANOTHER_NAME` for a separate repeat and `--set training.seed=1` to change the training seed.
