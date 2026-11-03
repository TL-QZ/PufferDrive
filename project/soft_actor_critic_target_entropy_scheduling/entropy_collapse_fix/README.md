# TES entropy-collapse candidates

Latest analysis: [Round 5 two-step TES collapse](ROUND5_DIAGNOSIS.md), including the new reward components, replay-refresh comparison, and missing diagnostic flag. Reproduce with `python project/soft_actor_critic_target_entropy_scheduling/entropy_collapse_fix/diagnose_round5.py` from the activated repository venv.

Read [DIAGNOSIS.md](DIAGNOSIS.md) for the run-by-run findings, including Round 2's recovery and persistent collapse in both one-step and two-step Round 3 TES. The [history plot](evidence/entropy_and_performance.png) and [numerical evidence](evidence/summary.json) come from existing local artifacts. The candidate configurations below remain one-step controls; they do not add a two-step intervention.

This folder contains four isolated, fresh-run candidates derived from the existing Round 2 and Round 3 TES SAC configs. Both variants hold the selected target entropy at `1.863680` by setting `train.target_entropy_schedule: constant` and clearing `train.target_entropy_force_decay_steps`. The stabilized variant adds only an actor learning-rate reduction. These are runnable candidates; no training has been run here and no improvement is established.

Launch from the repository root:

```bash
bash project/soft_actor_critic_target_entropy_scheduling/entropy_collapse_fix/scripts/launch.sh --round 2 --variant constant_control --trial 97 --dry-run --no-wandb
bash project/soft_actor_critic_target_entropy_scheduling/entropy_collapse_fix/scripts/launch.sh --round 3 --variant stabilized --trial 98 --gpu-groups 0,1 --dry-run --no-wandb
```

The defaults are `--variant stabilized` and `--trial 98`. `--round 2|3` is required. Remove `--dry-run` to launch training. By default, each launch gets a unique output directory under this folder's `runs/` directory; `--output-dir PATH` passes a custom location to the native sweep launcher. The launcher always uses the selected trial's existing 97/98 grid mapping and starts with `train.fresh_run: true`.

| Setting | Round 2 / Round 3 constant_control | Round 2 / Round 3 stabilized |
| --- | --- | --- |
| `train.target_entropy` initial grid value | `1.863680` for trials 97 and 98 | `1.863680` for trials 97 and 98 |
| `train.target_entropy_schedule` | `constant` | `constant` |
| `train.target_entropy_force_decay_steps` | `null` | `null` |
| Actor `train.learning_rate` | `0.0005` | `0.0001` |
| Critic `train.critic_learning_rate` | `0.0005` | `0.0005` |
| `train.alpha_learning_rate` | `0.0003` | `0.0003` |
| `train.anneal_lr` | `true` | `true` |
| `train.learn_alpha` | `true` | `true` |
| `train.update_epochs` for these grid IDs | `3` | `3` |

`train.anneal_lr` controls cosine scheduling for actor, critics, and learned alpha together. It remains enabled in both variants; the stabilized variant changes only the actor learning-rate value and leaves critic and alpha rates unchanged.

Trial identities are unchanged in all four config sets: trial 97 uses `gamma=0.999`, replay refresh `0.1`, a 2,000,000,000-row budget, and evaluation every 10 cycles; trial 98 uses `gamma=0.999`, replay refresh `0.5`, a 4,000,000,000-row budget, and evaluation every 2 cycles. Both retain `update_epochs=3`, target entropy `1.863680`, and learned alpha. Each round/variant has a distinct `run_name`, `wandb_group`, and base `train.data_dir`; the native launcher writes each selected trial under its unique output path.

| Environment | Round 2 | Round 3 |
| --- | --- | --- |
| Training goal spacing | `20.0–200.0 m` | `6.666666666666667–66.66666666666667 m` |
| Evaluation catalog | Existing baseline catalog | Copied private catalog at `configs/round3_<variant>/evaluation_benchmarks.yaml`; benchmark spacing remains `10.0 m` |

Round 2 and Round 3 retain their source environment, evaluation, replay, and launcher settings apart from the candidate controls, identity, and fresh-run setting described above.

## Verified on 2026-11-02

Eight dry runs covered both rounds, both variants, and both trial IDs. Resolved-config assertions checked budgets, evaluation cadence, grid identities, unchanged source settings, and private Round 3 benchmark catalogs. Shell syntax, invalid argument rejection, and invocation from outside the repository also passed. The existing target-entropy controller suite passed all 34 tests. These checks do not establish training stability; no candidate training was launched.

To refresh the diagnostic evidence from the repository root:

```bash
source .venv/bin/activate
python project/soft_actor_critic_target_entropy_scheduling/entropy_collapse_fix/diagnose.py
```

This overwrites only this folder's generated evidence; review the diagnosis against the new snapshot after rerunning. It reads standard TES trial-98 histories plus `round3_nstep2_tes_sac` trial 98 and does not evaluate newly trained candidate checkpoints. Each run's evaluations are bounded by its captured learner step count. If extraction encounters an incomplete JSONL write, retry after the writer finishes.
