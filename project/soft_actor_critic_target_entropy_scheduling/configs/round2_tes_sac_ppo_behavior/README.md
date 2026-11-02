# Round 2 TES SAC with frozen PPO behavior

This isolated config set carries forward TES trials 97 and 98 with a replay
capacity of 2,048,000 transitions per rank. With minibatches of 128,000, each
replay pass has 16 logical updates; three passes produce 48 updates per train
cycle. Microbatches remain 32,000.

Trial IDs retain their grid mappings: both use gamma `0.999` and initial target
entropy `0.75 * ln(12) = 1.863680`; trial 97 uses replay refresh `0.1`, a 2B
agent-row budget and evaluation every 10 cycles; trial 98 uses refresh `0.5`, a
4B budget and evaluation every 2 cycles.

`target_entropy_required_count: 200` requires 200 accumulated qualifying
observations (at least four full training cycles plus eight updates).
`target_entropy_force_decay_steps: 500000000` additionally forces one target
reduction if 500 million global received agent rows have elapsed since the last
reduction. The initial timer starts at zero. Both normal and forced reductions
multiply the target by 0.9, reset the qualifying counter and restart this timer;
the moving statistics are retained. If both conditions trigger together, only
one reduction occurs. A large step-count jump also causes only one reduction.

The timeout uses `agent_steps`, summed across ranks, including masked/reset
rows. It does not count replay samples or optimizer updates. It is checked on
the first actor/alpha update after collection reaches the threshold, so a
collection cycle can overshoot 500M. The changed target applies to the next
alpha update. Critic-only updates do not advance the controller. Timer state is
saved in learner checkpoints. Track `losses/target_entropy` together with
`target_entropy/forced_decay_count`, `target_entropy/last_decay_agent_steps`,
and `target_entropy/steps_since_decay`.

For example, a normal reduction at 300M moves the forced deadline to 800M.
If no normal reductions occur, forced reductions occur near 500M, 1B, 1.5B,
and so on, subject to collection-cycle boundaries. Both trials use this rule;
trial 98 retains its 4B received-row budget and 50% replay refresh.

## PPO-only collection

The only algorithmic change from `../round2_tes_sac` is the collection policy.
`behavior_policy.mode: ppo_only` keeps PPO action probability at 1 for the entire
run, including any received-batch overshoot. Actions are sampled categorically
from frozen PPO logits. There is no PPO-to-SAC handoff and no PPO optimization.
SAC starts fresh; its actor, critics, temperature, targets, and benchmark
policy evaluation retain the native SAC behavior.

Teacher: `experiments/baseline_run_sync_2026-08-24/`
`baseline_run_sync_2026-08-24_2026-08-26_16-23-18_seed0/`
`best_models/best_trainer_state_000611.pt`, with the adjacent run `config.yaml`.
This best checkpoint is tensor-identical to `final_model.pt`. The trainer state
records 5,005,312,000 rank-local steps at epoch 611; the baseline two-GPU
launcher corresponds to approximately 10.01B global steps.

The dedicated resolver applies the original grid and per-trial overrides, then
sets `behavior_policy.total_steps` to the resolved 2B/4B budget. Representation
compatibility is validated before creating a simulator. Teacher weights and
SAC action tables are checked when training loads the teacher.

From the repository root:

```bash
bash project/soft_actor_critic_target_entropy_scheduling/scripts/launch_round2_tes_sac_ppo_behavior.sh --trial 98 --dry-run --no-wandb
```

Remove `--dry-run --no-wandb` to train with W&B; use `--trial 97` for the other
candidate, and `--gpus 2,3` to select a different GPU pair. Each invocation uses
exactly two GPUs and creates a fresh run directory under
`project/soft_actor_critic_target_entropy_scheduling/runs/round2_tes_sac_ppo_behavior_*`.
Dry runs write the resolved config and print the command without training.
Use this dedicated launcher: the ordinary SAC sweep launcher does not install
the PPO behavior provider.

W&B project: `pufferdrive_sac_ppo_behavior`; group: `round2_tes_sac_ppo_behavior`.
Check `behavior/ppo_probability`, `behavior/ppo_action_fraction`,
`behavior/ppo_insert_fraction`, and `behavior/ppo_replay_fraction` (1 after
nonempty collection/replay), plus `behavior/sac_actions` (0).
