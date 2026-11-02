# Round 1: Round 7 candidates with 32-step SAC returns

This isolated round derives from `project/soft_actor_critic_tuning/configs/round7/` and retains its trial grid and settings. It selects the same deterministic candidates, 97 and 98, while setting `train.n_steps: 32`. Output paths and the W&B group are specific to this experiment. The Round 7 source configuration remains unchanged.

## Candidates

| Trial | `gamma` | Replay refresh | Global received-row budget | Evaluation interval |
|---|---:|---:|---:|---:|
| 97 | 0.999 | 0.1 | 2B | 10 collection cycles |
| 98 | 0.999 | 0.5 | 4B | 2 collection cycles |

Both retain Round 7's two-GPU setup, 2,048,000 replay entries per rank, optimizer and learning rates, target entropy, minibatch sizes, evaluation benchmark, and environment settings. The deterministic mapping remains `gamma=.999`, target entropy `1.863680`, and three replay passes; the selected replay refresh ratio differs by trial. Both start fresh with empty replay and use W&B group `sac_multi_step_return_round1_nstep32`.

The 32-step setting sums up to 32 consecutive valid rewards per starting transition. At `env.dt=0.3` seconds, that is up to 9.6 seconds of simulated time. Episode termination, timeout/map resampling, or invalid masks can shorten windows; boundaries flush valid prefixes according to the [n-step boundary contract](../../design.md#boundary-contract). A larger window follows the collection behavior policy for more steps and is not importance-corrected, as described in the [design](../../design.md#target-and-scope).

## Staging memory

Each rank has `vec.num_envs * env.num_agents = 20 * 3,200 = 64,000` stable slots and 923 observation features. The accumulator reserves an observation ring plus one endpoint per slot:

```text
64,000 * (32 + 1) * 923 * 4 bytes = 7.26 GiB per rank
```

The run uses CPU replay with pinned storage, so n-step staging also uses pinned CPU memory. This is additional to the Round 7 replay allocation and other process memory. The shared sweep launcher's host RAM preflight accounts for replay bytes but does not include this n-step staging allocation; ensure enough host RAM is available before launching.

## Launch

Run each trial as its own two-rank job. The wrapper chooses a unique timestamped output directory under `project/soft_actor_critic_multi_step_return/runs/round1_nstep32/` for each invocation. Default GPUs remain `0,1`; use `--gpus ID,ID` to select another pair.

```bash
source .venv/bin/activate
bash project/soft_actor_critic_multi_step_return/scripts/launch_round1.sh --trial 97
bash project/soft_actor_critic_multi_step_return/scripts/launch_round1.sh --trial 98
```

Preview resolved configurations and launch commands without starting training. Dry-run files are written to the requested output directory, so use fresh paths:

```bash
bash project/soft_actor_critic_multi_step_return/scripts/launch_round1.sh --trial 97 --dry-run --no-wandb --output-dir /tmp/sac_round1_nstep32_trial97
bash project/soft_actor_critic_multi_step_return/scripts/launch_round1.sh --trial 98 --dry-run --no-wandb --output-dir /tmp/sac_round1_nstep32_trial98
```

Dry-run validates configuration composition, trial mapping, and command construction. It does not reserve GPUs, check full n-step staging RAM, establish training throughput, or test policy quality. No training was started while preparing these files.

Validation on 2026-10-30: both trial dry runs passed, and the wrapper passed `bash -n`. A parsed YAML comparison against Round 7 found only `train.n_steps: 32` and the new W&B group as configuration changes; fixed settings and launcher trial mappings are identical. The wrapper also rejects an empty output path, preserving output isolation.
