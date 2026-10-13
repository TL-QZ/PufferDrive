# Evaluate and compare the 16 SAC checkpoints

Run these commands yourself from the repository root. Preparing these scripts
does not launch evaluation. GPU IDs below are examples; choose available GPUs.
The shell wrappers activate `.venv` and limit numerical-library threads.

## 1. Preview the evaluation queue

```bash
bash project/soft_actor_critic_tuning/scripts/evaluate_carla.sh \
  --evaluation-name carla_final_v1 \
  --gpus 0,1 --max-concurrent 2 --dry-run
```

The default sweep is `sac16_grid_20261012T032408Z`. Preview prints commands and
validates checkpoint/config identities without creating environments, GPU
contexts, output folders or W&B sessions.

The protocol matches the baseline final CARLA evaluation: 1,000 scenarios,
eight towns, seed 42, 6,000 steps, `dt=0.1`, mean actions, BF16 GPU inference,
20 workers and `eval.num_agents=300`. Each evaluation uses one GPU. Final
evaluation compilation is disabled to match the recorded baseline; the
training sweep's compilation setting is unchanged.

## 2. Launch one checkpoint first

```bash
bash project/soft_actor_critic_tuning/scripts/evaluate_carla.sh \
  --evaluation-name carla_final_v1 \
  --trial-ids 1 --gpus 0 --max-concurrent 1
```

W&B reporting is enabled by default and attaches `final_eval_carla/` summary
metrics to the checkpoint's original training run. Authenticate W&B before
launching, or add `--no-wandb` for local reports only. Keep that choice the same
when resuming the session. No training or replay
allocation is involved.

The evaluation session manifest and console logs are written under:

```text
runs/sac16_grid_20261012T032408Z/evaluations/carla_final_v1/
```

Each trial's raw reports remain beside its checkpoint under
`learner/eval/carla_final_model_mean_metrics_carla_final_v1/<timestamp>/`.
The session manifest records the exact CSV, summary and resolved-config paths.

## 3. Evaluate the remaining checkpoints

```bash
bash project/soft_actor_critic_tuning/scripts/evaluate_carla.sh \
  --evaluation-name carla_final_v1 --resume \
  --gpus 0,1 --max-concurrent 2
```

This selects all 16 trials and skips completed evaluations that still validate.
To launch all 16 in a new session immediately, use the same command with a new
evaluation name and omit `--resume`. To retry selected failed trials, combine
`--resume` with `--trial-ids`, for example `--trial-ids 3,9`. If a failed trial
already produced a summary but was not recorded as completed, use a new
evaluation name; the launcher rejects ambiguous existing reports.

Keep one launcher per evaluation session. Existing sessions require explicit
`--resume`; checkpoint files and the training manifest are preserved. Ctrl-C
stops the launcher and its owned evaluation processes/workers. Review failed
trial logs before retrying.

## 4. Generate the comparison plots after evaluation

First validate the complete evaluation and preview the output paths:

```bash
bash project/soft_actor_critic_tuning/analysis/plot_carla.sh \
  --evaluation-manifest project/soft_actor_critic_tuning/runs/sac16_grid_20261012T032408Z/evaluations/carla_final_v1/manifest.json \
  --output-name sac16_carla_final_v1 --dry-run
```

Remove `--dry-run` to generate the plots. Use a new output name for each
analysis; existing output folders are not overwritten.

Outputs go to `plots/sac16_carla_final_v1/`. Each plotting view preserves the
four metric groups: infractions, goal/completion, motion/lane/comfort, and
Puffer score/components. The overview contributes four figures, each of four
parameter-effect folders contributes four, and the interaction view contributes
four: 24 PNGs in total. CSV tables record trial identities, aggregate metrics
and the matched parameter comparisons. There are no town-comparison plots.

The analysis requires all 16 complete full CARLA evaluations with matching
protocol and scenario identities. It uses the explicit session manifest and
does not pick whichever report happens to be newest. The four tuning values
are shown in [the trial table and analysis plan](evaluation_plan.md).

Preparation checks passed: launcher dry runs for the 16-trial queue and a
selected trial, a mocked launch/resume, and synthetic CPU generation of all
24 figures, three CSV tables and 32 matched parameter pairs under `/tmp`.
Invalid parameter, incomplete-grid, scenario and protocol inputs were rejected
before analysis output creation. A focused code review found no material
issues. Real CARLA execution and W&B reporting have not been run for this
workflow; the user starts those commands.
