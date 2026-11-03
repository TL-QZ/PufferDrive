# Multi-step returns for discrete SAC

`train.n_steps` controls how many consecutive, valid transitions contribute observed rewards to each SAC replay entry. The default is `1`, which retains the existing one-step collector. Values above `1` enable the standard SAC accumulator; PPO behavior-policy and warm-start workflows require `1`.

For example, with `n_steps=3`, rewards `[1, 2, 3]`, and `gamma=0.9`, the first replay entry contains reward `5.23` and discount `0.729`. The critic target is `5.23 + 0.729 * V_soft(successor)`. A true terminal instead sets the discount to zero.

A larger horizon gives each target more observed rewards; it does not establish that short-sighted behavior was caused by one-step SAC, whose critic can already propagate delayed rewards through bootstrapping. The [first experiment configs](configs/round1_nstep32/README.md) preserve Round 7 trials 97 and 98 with `train.n_steps: 32`. Preparing these configurations does not launch training.

Validation on 2026-10-30: **200 tests passed**, including CPU, CUDA/pinned-offload parity, checkpoint compatibility, and two-rank Gloo tests. See [validation evidence and limits](design.md#validation).

## First experiment: 32-step returns

Launch one candidate at a time on the default GPU pair `0,1`:

```bash
source .venv/bin/activate
bash project/soft_actor_critic_multi_step_return/scripts/launch_round1.sh --trial 97
bash project/soft_actor_critic_multi_step_return/scripts/launch_round1.sh --trial 98
```

Use `--gpus ID,ID` to select another pair. Add `--dry-run --no-wandb` to preview a candidate without starting training. Both candidates use 32-step returns; their inherited settings and validation evidence are listed in the [configuration README](configs/round1_nstep32/README.md).

## Reading order

For the TES collapse investigation, [Round 4](configs/round4_nstep2_tes_sac/README.md)
retains Round 3's two-step adaptive TES configuration, reduces the logical
minibatch from 128,000 to 64,000 rows per rank, and enables optional collapse
diagnostics. The 32,000-row microbatch remains unchanged. This doubles logical
updates per collection cycle from 48 to 96; preparation and dry runs do not
establish whether it improves stability.

```bash
bash project/soft_actor_critic_multi_step_return/scripts/launch_round4.sh --trial 98 --dry-run --no-wandb
```

[Round 4.1](configs/round4_1_nstep2_tes_sac/README.md) retains Round 4 and lowers
the initial actor and critic learning rates to `1e-4`, and the initial alpha
learning rate to `7e-5`. Cosine annealing remains enabled. Preview it with
`bash project/soft_actor_critic_multi_step_return/scripts/launch_round4_1.sh --trial 98 --dry-run --no-wandb`.

1. Read the [boundary contract](design.md#boundary-contract), especially terminal versus timeout and invalid masks.
2. Read the [overlapping-window example](design.md#overlapping-windows) to see exactly what reaches replay.
3. Read [validation](design.md#validation) before selecting run settings.

## Code map

| File | Responsibility |
|---|---|
| `pufferlib/sac_replay.py` | Per-agent `NStepAccumulator`, plus unchanged five-field `TensorReplay` interface |
| `pufferlib/pufferl_sac.py` | Configuration, collection, quota selection, metrics, checkpoint compatibility |
| `pufferlib/config/puffer_drive_sac.yaml` | Default `train.n_steps: 1` |
| `tests/unit_tests/test_sac_nstep.py` | Independent numerical oracle and accumulator boundaries |
| `tests/unit_tests/test_sac_nstep_collection.py` | Collector and compatibility tests |

## Provenance and limits

The target follows the reward accumulation and endpoint bootstrap used by [Stable Baselines3 SAC](https://stable-baselines3.readthedocs.io/en/master/_modules/stable_baselines3/sac/sac.html) and its [NStepReplayBuffer](https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/buffers.py). [Antonin Raffin's Isaac Sim article](https://araffin.github.io/post/tune-sac-isaac-sim/) motivated evaluating this strategy; its continuous-control results do not establish a benefit for discrete driving.

Here, aggregation happens during collection so replay retains its existing sampling and replacement behavior. SB3 reconstructs sequences when sampling its chronological buffer. Thus this is target-level correspondence, not a storage implementation port. Entropy is included only at the bootstrap endpoint when `critic_entropy_in_target=true`; intermediate entropy terms and importance corrections are not included. Longer sequences follow behavior policies that can differ from the learner's current policy, introducing uncorrected off-policy bias.
