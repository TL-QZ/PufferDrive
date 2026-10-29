# Adaptive target entropy scheduling

## Round 2: Round 7 replay capacity

The isolated [round2_tes_sac configuration](configs/round2_tes_sac/README.md)
uses 2,048,000 replay transitions per rank, matching Round 7. With 128,000-row
logical minibatches and three replay passes, each training cycle performs
`3 * ceil(2,048,000 / 128,000) = 48` actor updates and entropy-controller
observations. Each individual replay pass has 16 updates. The two ranks update
together; 32,000-row microbatches accumulate gradients and do not add optimizer
steps. The initial target remains 75% of maximum entropy.

Round 2 uses `target_entropy_required_count: 200`, requiring at least five
training cycles for a normal reduction, potentially longer because only
qualifying observations increment the counter. It also sets
`target_entropy_force_decay_steps: 500000000`: if 500M global received agent
rows pass without a reduction, the first subsequent actor/alpha update forces
one 10% reduction. Any normal or forced reduction restarts this timer and
resets the qualifying count; moving statistics persist. The timer starts at
zero and is restored from learner checkpoints. The step count is summed
across GPUs and includes masked/reset rows. Checkpoints/logs track the timer
and forced-reduction count; see the Round 2 README for metric names.
With the inherited EMA decay
of 0.999, the statistics have a characteristic timescale of roughly 1,000
observations, or 20.8 training cycles at this replay capacity.

Preview a fresh Round 2 run without training:

```bash
bash project/soft_actor_critic_target_entropy_scheduling/scripts/launch_round2_tes_sac.sh --trial 97 --dry-run --no-wandb
```

## Original TES experiment

This isolated experiment inherits the Round 6 learner, environment, replay,
network, and evaluation settings. It retains the 108-entry controlled grid and
the historical meanings of trials 97 and 98. The third controlled target
entropy coordinate and base `train.target_entropy` are `1.863680`, so both
selected trials start at `0.75 * log(12)` nats, rounded to six decimals. The two
lower target-entropy coordinates remain in the grid and keep the preceding
trial IDs intact. The `sweep.train.target_entropy` catalog uses the same third
coordinate for consistency.

The adaptive controller tracks an exponential moving mean and standard
deviation of policy entropy. An update qualifies when its moving mean differs
from the current target by less than 0.01 nats and its moving standard deviation
is at most 0.05. Qualifying updates count cumulatively: a non-qualifying update
does not clear the count. After one qualifying update
(`target_entropy_required_count: 1`), the target is multiplied by 0.9 and the
qualifying count resets to zero; the moving mean and standard deviation persist.
Controller statistics also persist across learner checkpoints. A reduced
target takes effect on the next learner update. Training logs expose the
current target, moving mean and standard deviation, qualifying-update count,
decay count, and required count as `losses/target_entropy`, `target_entropy/mean`,
`target_entropy/std`, `target_entropy/qualifying_count`,
`target_entropy/decay_count`, and `target_entropy/required_count`.

This schedule follows the exponential-window approach in Section 3.2 and
Algorithm 1 in Appendix B of [Target Entropy Annealing for Discrete Soft
Actor-Critic](https://arxiv.org/html/2112.02852v1). The explicit count of one
qualifying update is this experiment's choice; it is not identified with the
paper's target-critic update interval. If `target_entropy_required_count` is
`null`, the controller derives the count as
`update_epochs * ceil(train.batch_size / train.minibatch_size)`. For these
configs that is `3 * ceil(8,192,000 / 128,000) = 192` qualifying updates.

The existing cosine schedules remain enabled for actor, critic, and learned
alpha learning rates. Entropy scheduling changes the target used by the alpha
objective; it does not change alpha's learning-rate schedule. Both trials keep
`learn_alpha: true`.

## Launch

Activate the repository environment, then launch one candidate at a time on two
GPUs. Each launch gets a unique output directory under this project's `runs/`
folder unless `--output-dir` is supplied.

```bash
source .venv/bin/activate
bash project/soft_actor_critic_target_entropy_scheduling/scripts/launch.sh --trial 97
bash project/soft_actor_critic_target_entropy_scheduling/scripts/launch.sh --trial 98
```

Preview resolved settings and commands without training:

```bash
bash project/soft_actor_critic_target_entropy_scheduling/scripts/launch.sh --trial 97 --dry-run --no-wandb
bash project/soft_actor_critic_target_entropy_scheduling/scripts/launch.sh --trial 98 --dry-run --no-wandb
```

`--gpus ID,ID` selects two distinct nonnegative GPU IDs. `--output-dir PATH`
sets an explicit sweep directory. Trial 97 retains its 2B-row budget, 0.1
replay refresh ratio, and 10-cycle evaluation interval. Trial 98 retains its
4B-row budget, 0.5 refresh ratio, and 2-cycle evaluation interval. Runs start
fresh, keep the Round 6 W&B project, and use the distinct group
`target_entropy_scheduling`.

A dry run checks configuration resolution and launcher command construction. It
does not measure GPU fit, training throughput, or policy quality.
