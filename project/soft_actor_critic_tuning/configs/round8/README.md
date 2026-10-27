# Round 8: fixed-alpha SAC without critic-target entropy

Round 8 starts fresh runs for candidates 97 and 98 using the Round 7 replay
capacity, row budgets, evaluation cadence, initialization, and other training
settings. It changes two SAC mechanisms: the critic bootstrap target omits its
entropy term, and alpha is fixed at `0.0003` for 97/98 or `0.01` for 205/206. The actor loss remains
entropy-regularized with this fixed coefficient.

With `critic_entropy_in_target: false`, the critic target is

```text
y = r + d * E_a'[ min(Q1_target(s', a'), Q2_target(s', a')) ]
```

Here `d` is the stored bootstrap discount, so Round 8 keeps the existing
terminal and truncation bootstrap semantics. The entropy term
`-alpha * log pi(a'|s')` is omitted from the target. The actor still minimizes
the usual discrete SAC objective
`E_a[alpha * log pi(a|s) - min(Q1(s,a), Q2(s,a))]`, with alpha held at
the selected coefficient because `learn_alpha: false`.

The paper's critic-entropy ablation retains automatic temperature tuning
([paper, v2](https://arxiv.org/html/2509.09838v2)). Round 8 adds a separate
fixed-temperature choice so the actor's entropy coefficient stays constant.
`alpha_learning_rate: 0.0003` remains in the inherited config for comparison,
but no alpha optimizer update or alpha learning-rate schedule step occurs while
`learn_alpha` is false. `train.anneal_lr: true` still schedules the actor and
critic learning rates.

The original candidates keep their Round 3 grid IDs and Round 7 resource settings.
Trials 205/206 copy 97/98 with only alpha changed to `0.01`:

| Candidate | Fixed alpha | Replay refresh | Global row budget | Evaluation interval | Nominal cosine horizon |
|---|---:|---:|---:|---:|---:|
| 97 | 0.0003 | 0.1 | 2B | 10 cycles | 4,883 cycles |
| 98 | 0.0003 | 0.5 | 4B | 2 cycles | 1,954 cycles |
| 205 | 0.01 | 0.1 | 2B | 10 cycles | 4,883 cycles |
| 206 | 0.01 | 0.5 | 4B | 2 cycles | 1,954 cycles |

All four retain `gamma=0.999`, `target_entropy=1.863680`, three replay passes,
2,048,000 replay rows per rank, a 128,000-row logical minibatch, and a
32,000-row microbatch. Each run uses two GPUs, 20 environments per rank with
3,200 slots each, and the same evaluation setup as Round 7. The selected grid
IDs still resolve to refresh ratios 0.1 and 0.5. The controlled grid continues
to include `target_entropy` to preserve these historical IDs; with
`learn_alpha: false`, that field does not affect actor, critic, or alpha
updates. It is retained for ID compatibility and does not add candidates.

The alpha axis precedes the historical 108-combination grid, so the new
copies have IDs `97 + 108 = 205` and `98 + 108 = 206`. Only these four
candidates are selected by default.

## Launch

Activate the repository environment and launch one candidate at a time on two
GPUs (default pair `0,1`):

```bash
source .venv/bin/activate
bash project/soft_actor_critic_tuning/scripts/launch_round8.sh --trial 97
bash project/soft_actor_critic_tuning/scripts/launch_round8.sh --trial 98
bash project/soft_actor_critic_tuning/scripts/launch_round8.sh --trial 205
bash project/soft_actor_critic_tuning/scripts/launch_round8.sh --trial 206
```

Preview resolved configs and commands without training:

```bash
bash project/soft_actor_critic_tuning/scripts/launch_round8.sh --trial 97 --dry-run --no-wandb
bash project/soft_actor_critic_tuning/scripts/launch_round8.sh --trial 98 --dry-run --no-wandb
```

`--gpus ID,ID` selects two distinct nonnegative GPU IDs. Each launch creates a
new sweep under `project/soft_actor_critic_tuning/runs/round8/` by default;
each trial's learner artifacts are isolated in its own `learner/` directory.
The launcher marks runs fresh and does not restore a Round 7 learner state or
actor. Round 7 checkpoints encode different critic targets and a learned
temperature, so they are not used to initialize these experiments.

Budgets count global received agent rows and can overshoot by a collection
batch; replay refresh counts valid inserted transitions. The nominal cosine
horizons use `ceil(global_row_budget / (2 * refresh_count_per_rank))`, where
`refresh_count_per_rank = ceil(2,048,000 * replay_update_ratio)`. The first
cycle fills replay before steady-state refreshes, and invalid rows do not
count toward the refresh quota. Dry runs validate config generation and
scheduling only; no training was launched for Round 8.
