# Round 5: Round 3 trial 98 with 25% replay replacement

Copies multi-step Round 3 trial 98, changing replay replacement from 50% to
25% and using isolated experiment paths. Round 5 uses its own trial ID **1** in a single-entry grid, with 0.25 in both
controlled and sweep settings.

Each steady-state collection cycle requests 512,000 new replay rows per rank,
versus 1,024,000 previously, for a replay capacity of 2,048,000. Initial filling
is unchanged. The learner retains 48 logical updates per cycle, doubling
updates per requested new replay row relative to Round 3 trial 98.

Preserved settings include adaptive TES, two-step returns, gamma 0.999,
three update epochs, 128,000 logical minibatch / 32,000 microbatch, initial
actor/critic LR 0.0005, initial alpha LR 0.0003, initial alpha 0.1, cosine
annealing, and the 4B received-row budget. Evaluation remains every two cycles,
now half a nominal replay turnover. Goal spacing and benchmark contents match
Round 3. Round 4's minibatch and diagnostic changes are not inherited.

Preview:

```bash
bash project/soft_actor_critic_multi_step_return/scripts/launch_round5.sh --trial 1 --dry-run --no-wandb
```

Standalone evaluation must explicitly pass
`eval.benchmark_config=project/soft_actor_critic_multi_step_return/configs/round5_nstep2_tes_sac/evaluation_benchmarks.yaml`.

Validation: trial 1 dry run, resolved-config parity against Round 3 trial 98,
and shell syntax checks passed. No training was launched.
