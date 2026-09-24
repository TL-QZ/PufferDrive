# Condition B: random encoder initialization

| Component | Initialization | Updates |
|---|---|---|
| Driving teacher | Existing pretrained PPO checkpoint | Frozen; drives collection and supplies distillation logits |
| Student encoder | Fresh DriveBackbone with the teacher's architecture | Gradients |
| Target encoder | Independent copy of the initial random student | EMA |
| Predictor and chunk decoder | Random, as in the original recipe | Gradients |

- Select with `model.encoder_initialization: random`; omitted or `teacher` preserves warm starting.
- Student and target initially match; neither shares parameters with the driving teacher.
- Full recipe: 8,192,000 transitions/GPU/collection, 50 passes, 15 collections, effective batch 128,000/GPU, microbatch 1,024. Student driving evaluation follows each collection's training; W&B settings inherit the full recipe with a separate group.
- Use a new run ID. Resuming restores saved weights and requires the saved initialization configuration.

## Official launch — user starts this

```bash
CUDA_VISIBLE_DEVICES=2,3 project/jepa_distill/scripts/train_random_2gpu.sh \
  --run-id condition_b_random_seed0
```

Config: `project/jepa_distill/config/condition_b_random.yaml`. Both new launchers default to GPUs **2,3**. An explicit `CUDA_VISIBLE_DEVICES` overrides that default.

## Bounded verification

```bash
CUDA_VISIBLE_DEVICES=2,3 project/jepa_distill/scripts/train_toy_random_2gpu.sh \
  --run-id YOUR_UNIQUE_RANDOM_TOY_ID
```

The toy uses the existing two-GPU toy recipe: two collections, two passes each, held-out validation, student driving evaluation, checkpoints, and online W&B in `tobieliu825/pufferdrive-jepa-distill-toy`.

## Verified 2026-09-23

| Check | Result |
|---|---|
| Regression suite | All 61 unique tests passed; the five CPU distributed tests were rerun with host sockets because the sandbox blocked Gloo |
| Two-GPU toy | `random_encoder_toy_20260923_01`, GPUs 2,3, exit 0; 8 updates, 16,384 total collected transitions |
| Student driving evaluation | CARLA, two scenarios at each of steps 4 and 8; summaries and per-episode CSVs saved |
| Checkpoint | `final_model.pt` records random initialization; CPU reconstruction restored every saved tensor exactly |
| W&B | [kzha7bgq](https://wandb.ai/tobieliu825/pufferdrive-jepa-distill-toy/runs/kzha7bgq), API-confirmed `finished`, training/validation/evaluation metrics uploaded |

- Artifacts: `experiments/jepa_distill/runs/random_encoder_toy_20260923_01/`.
- Toy process exited; host process check found no matching launch. GPUs 2,3 returned to 1 MiB / 0% utilization. The existing training on GPUs 0,1 remained active.
- Initial toy attempt caught strict CLI rejection of the new option. Declaring the default in the base config fixed it; resume comparison treats an omitted legacy value as `teacher`.
- The official launch command passed a dry run. Full-scale training has **not** been started; toy success verifies execution, not convergence or full-scale storage capacity.
