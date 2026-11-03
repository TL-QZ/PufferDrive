# Round 4 ablation: omit critic-target entropy

Derived from Round 4, specifically the saved trial 98 configuration at
`runs/round4_nstep2_tes_sac/sac108_grid_20261102T202418Z560107829/configs/trial_98.yaml`.
The only algorithm change is `train.critic_entropy_in_target: false`.
Actor entropy regularization, learned alpha (initially 0.1), adaptive target
entropy, two-step returns, 64,000-row logical minibatches, 32,000-row
microbatches, and collapse diagnostics remain enabled as in the source run.

For a full two-step transition, the target is
`r_t + gamma * r_(t+1) + gamma^2 * E[min(Q1_target, Q2_target)]`.
Only bootstrap-endpoint entropy is removed; terminal and truncation semantics
are unchanged. Start fresh rather than loading the source run's critic/replay.

Trial 98 retains replay refresh 0.5, 4B global received rows, and evaluation
every two cycles. Trial 97 retains its inherited 0.1 / 2B / ten-cycle settings.
The seed, goal spacing, benchmark contents, and all other training settings
are preserved. Output directories and W&B group are isolated for this variant.

```bash
source .venv/bin/activate
bash project/soft_actor_critic_multi_step_return/scripts/launch_round4_no_critic_entropy.sh --trial 98 --dry-run --no-wandb
```

To train, omit `--dry-run --no-wandb` and select two GPUs with `--gpus ID,ID`.
Standalone evaluation must explicitly pass
`eval.benchmark_config=project/soft_actor_critic_multi_step_return/configs/round4_no_critic_entropy_nstep2_tes_sac/evaluation_benchmarks.yaml`.

Validation: both trial launcher dry runs and shell syntax passed. Resolved trial
98 matches the named source run exactly except critic-target entropy and four
run identity/path fields. All five source YAML files match after those intended
changes; the evaluation catalog is byte-identical. No training was launched.
