# Compiled SAC training and progression evaluation

Archived smoke validation. Settings and paths describe the historical check.

The sweep now enables `torch.compile` and runs the baseline `carla_fast`
benchmark after every outer collection/training cycle. The four grid axes
remain unchanged: epochs `[3,10]`, replay replacement ratio `[0.05,0.1]`,
initial alpha `[0.1,0.2]`, and tau `[0.005,0.01]`.

## Current run settings

| Setting | Value and reason |
| --- | --- |
| Training evaluation | `evaluation_interval_epochs=1`, `carla_fast`: 250 scenarios, 500 steps, `dt=0.1`, mean action. Frequency is the user's tuning choice; benchmark definitions match PPO. |
| Checkpoint writes | `checkpoint_interval=null`: no periodic saves; normal close saves final actor and learner state. Replay/environment continuation remains unsupported. |
| Execution | Compilation enabled, BF16 autocast, logical minibatch 128,000 and microbatch 32,000. Production workers are `auto`, resolving to 20 per rank as in PPO. |
| Fixed optimizer settings | Actor and critic LR `0.0005`, training seed `0`, LR annealing off by user request. Alpha LR remains `0.0003`. |
| Stop budgets | PPO's 10,000,000,000 global received agent-row ceiling and a three-hour wall-time target, whichever stops first. |

The old one-trillion ceiling was introduced to let the wall-time limit dominate;
it is not a SAC requirement. Two ranks receiving ten environments each with
3,200 agent slots add 64,000 received rows to `agent_steps`. Training the same
replay rows repeatedly increases learner/sample counters, not `agent_steps`.
Invalid or reset-boundary rows can count toward received rows without becoming
valid replay entries. Benchmark evaluation does not increment this training
collection counter.

The trial monitor checks after a full collection/training/evaluation cycle.
It stops when the deadline passes or another cycle of the latest measured
duration would exceed it. This avoids starting an obviously unaffordable full
replay pass. It is an estimate: startup compilation makes the first cycle
longer, and future cycle durations can change. The parent retains a 600-second
shutdown grace. No full sweep was started for this validation.

## Verification

Configuration validation generated all 16 production YAML files and verified
that only the four grid axes and unique run paths vary. Every trial has
evaluation enabled, compilation enabled and periodic checkpoints disabled.
Temporary CPU checks covered checkpoint behavior, deadline decisions and
benchmark-artifact validation; Python compilation passed. A focused code
review found no material defects.

The real two-GPU compiled check **passed** in 249.1 seconds
with external logging disabled. It completed one collection/training cycle,
three logical learner updates, and all 250 benchmark episodes. Losses and
benchmark metrics were finite. Both ranks exited without errors, final actor
and learner checkpoints exist, and trial finalization reported no missing
artifacts. The owned monitor was stopped and `nvidia-smi` showed no remaining
CUDA processes after cleanup.

| Recorded result | Value |
| --- | --- |
| Collected global agent rows | 313,600 |
| Replay samples used across both ranks | 768,000 |
| Peak allocated CUDA memory, each rank | 8.614 GiB |
| Peak reserved CUDA memory, each rank | 12.234 GiB |
| Maximum sampled GPU utilization | GPU 0: 94%; GPU 1: 100% |

The benchmark took approximately 174 seconds in this check. Evaluation time
is included in each trial's wall-time budget. These are functional results,
not a compiled/eager speed comparison or evidence of policy convergence after
three updates.

Local artifacts: [trial summary](compiled_evaluation_2026-10-11/summary.json),
[benchmark summary](compiled_evaluation_2026-10-11/evaluation_summary.json),
[episode metrics](compiled_evaluation_2026-10-11/episode_metrics.csv),
and [GPU samples](compiled_evaluation_2026-10-11/gpu_usage.csv).
These small evidence files were copied into this archive independently of the
disposable smoke run directory.

The smoke uses two training environments/workers per rank and 128,000 replay
entries, retaining the full model and production microbatch size. It does not
qualify the full 8,192,000-entry replay, 20-worker production layout or full
sweep throughput. W&B and model uploads are disabled for the smoke; production
evaluation uses the existing logger with `eval_carla_fast/<metric>` keys.

Each cycle writes aggregate metrics and episode-level results under
`learner/eval/training/carla_fast/epoch_<epoch>_step_<agent_steps>/`.
Trial finalization requires a complete finite benchmark summary at the final
epoch. Earlier evaluation failures can still leave gaps, because the reused
evaluation helper catches exceptions; finalization checks the final benchmark.

See [launch instructions](../../../docs/how_to_launch.md) and the
[historical PPO parity audit](../../ppo_parity_audit_2026-10-11.md).
