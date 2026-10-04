# Condition B CARLA chunk evaluation

`scripts/evaluate_carla.sh` evaluates a Condition B checkpoint against the synced baseline CARLA catalog at `dt=0.1`. It uses the saved teacher PufferLib arguments to reconstruct the runtime, then matches the baseline's CARLA seed 42, 1,000 scenarios, 6,000 steps, mean action selection, 300 evaluation agents, 20 workers, and continuous controls. Observation dropout and other observation perturbations are zero. The checkpoint must record training `dt=0.3` and a four-slot decoder.

The checkpoint's `execution_horizon=1` metadata stays intact. The separate `--k` option chooses how often a new chunk is requested:

| `k` | Executed slots | Replan interval |
| --- | --- | --- |
| 1 | slot 0 each step | 1 step |
| 2 | slots 0 and 1, each for 3 steps | 6 steps |
| 3 | slots 0–2, each for 3 steps | 9 steps |
| 4 | slots 0–3, each for 3 steps | 12 steps |

The evaluator caches chunk logits by agent and clears only the rows reported terminated or truncated by the vector environment. This matters at CARLA scenario boundaries, where the environment returns reset observations in the same step.

Implementation: `evaluate_carla.py` adds the chunk adapter and standalone entrypoint; `scripts/evaluate_carla.sh` activates the project virtual environment and limits CPU threads. The shared evaluator has an optional `reset_eval_state` hook at rollout start and after environment steps. Existing policies without this hook retain their behavior, and the original student evaluator remains unchanged. Mean controls use the existing physical action table conversion. Inference matches baseline BF16 on CUDA with compilation disabled. Reports are local; this launcher does not attach to the teacher's W&B run.

Run every `k` sequentially, with production bounds:

```bash
CUDA_VISIBLE_DEVICES=4 project/jepa_distill/scripts/evaluate_carla.sh \
  --checkpoint experiments/jepa_distill/runs/condition_b_jepa_distill/checkpoint.pt \
  --output-dir experiments/jepa_distill/carla_chunk_eval_20260926 \
  --device cuda:0
```

Use `--k 1` (or 2, 3, 4) for one schedule. `DRY_RUN=1` prints the protocol and validates the checkpoint without creating files. Add `--smoke` for a separate short run: 2 scenarios × 13 steps, one worker, 50 agents. Production outputs refuse an existing output directory. Each `kN/` run contains `protocol.json`, the existing CARLA episode CSV, summary JSON, and resolved configs under `eval/carla_act_kN/standalone/`.

Choose an available GPU and replace the checkpoint path when evaluating another distilled model. All four schedules use the same loaded checkpoint and run sequentially. Use a fresh output directory for each invocation. This evaluates the repository's PufferDrive simulator on CARLA maps, matching the baseline workflow.

## Verification (2026-09-26)

- `python -m pytest -q project/jepa_distill/tests/test_evaluate_carla.py project/jepa_distill/tests/test_evaluate.py`: **20 passed**. Tests cover slot timing, observations at replanning, mean-control parity, baseline configuration, invalid inputs, and the shared rollout's reset hook.
- Real checkpoint smoke on GPU 4 with BF16: **all four k values passed**, each with 2 scenarios × 13 steps, one worker, and 50 agents. Thirteen steps deliberately leaves an unfinished cached plan at reset for k=2/3/4. Every CSV and summary contains two episodes; resolved configs confirm `dt=0.1`.
- Smoke artifacts: `/tmp/carla_chunk_smoke_20260926_gpu4/smoke_k{1,2,3,4}/`. These are temporary test outputs, not benchmark scores.
- Production dry-run confirmed 1,000 scenarios × 6,000 steps, 300 agents, 20 workers, seed 42, and all four schedules. Python compilation, shell syntax, and diff whitespace checks passed. Independent review found no material correctness defects.

The full evaluations have **not** been launched. The smoke verifies loading, CUDA inference, control scheduling, resets, and report generation; it does not establish driving performance or full-run duration.

## Four-GPU launcher

Run one schedule per GPU concurrently, in the order given by `CUDA_VISIBLE_DEVICES`:

```bash
CUDA_VISIBLE_DEVICES=2,3,4,5 project/jepa_distill/scripts/evaluate_carla_4gpu.sh \
  --checkpoint experiments/jepa_distill/runs/condition_b_distill_no_warm_start/final_model.pt \
  --output-dir experiments/jepa_distill/carla_eval_no_warm_start_4gpu_20260927
```

This maps k=1/2/3/4 to GPU 2/3/4/5. Each child sees its assigned GPU as `cuda:0` and reuses `evaluate_carla.sh`. Logs are `<output-dir>/kN.log`; results are `<output-dir>/run_kN/kN/eval/carla_act_kN/standalone/`. The launcher waits for all jobs and exits unsuccessfully if any job fails. Interrupting it terminates all four job process groups, including simulator workers.

Exactly four distinct GPU indices and a fresh output directory are required. CPU threads default to one per job; each evaluation retains the baseline's 20 simulator workers. Add `--smoke` for short runs. `DRY_RUN=1` or `--dry-run` prints the four commands without loading models, using GPUs, or creating files.

Validation (2026-09-27): shell syntax and command dry-run passed. A temporary fake evaluator verified four concurrent starts, isolated GPU assignment, separate logs, failed-child exit reporting, and rejection of invalid GPU lists. No new full evaluation was launched.
