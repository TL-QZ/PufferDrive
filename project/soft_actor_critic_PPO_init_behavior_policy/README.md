# SAC with frozen PPO collection: round 4_1 trials 97 and 98

This is a separate experiment from `project/soft_actor_critic_warm_start`.
The existing warm-start implementation, configurations, and runs are preserved.

## Agreed experiment

Both trials initialize the SAC actor and twin critics from scratch. The selected
PPO checkpoint supplies collection actions and stays frozen throughout:

```
experiments/ppo_sac_seed0_20261019_222638/ppo/best_models/best_trainer_state_000123.pt
```

Despite its filename, this file contains plain policy weights. Its saved PPO
configuration is `experiments/ppo_sac_seed0_20261019_222638/ppo/config.yaml`.
The source run records its epoch-123 evaluation at 2,015,232,000 received steps.
PPO initialization and optimizer history do not become SAC initialization or
optimizer history.

| New global received steps | PPO collection probability | SAC collection probability |
| --- | --- | --- |
| 0–250M | 100% | 0% |
| 250M–500M | Linear decrease from 100% to 0% | Linear increase from 0% to 100% |
| 500M–1B | 0% | 100% |

The **1B budget includes initial replay fill**. It excludes the teacher's prior
training and benchmark evaluation steps. Received steps count simulator agent
rows across both ranks, including masked/reset and quota-discarded rows; they
are not replay samples, optimizer updates, or valid inserts alone.

At collection boundaries, with `t = SAC.agent_steps`:

```
ppo_probability = clamp((500_000_000 - t) / 250_000_000, 0, 1)
```

Probability stays constant for that collection call. The collector caps the
received budget at the next 250M/500M/1B boundary. Whole vector receives can
overshoot a boundary; actual counters are logged. Changing stage does not reset
simulators or discard replay. In the first decay collection the probability may
still equal one; subsequent collection boundaries lower it.

SAC starts normal actor, critic, and temperature updates once replay is full and
continues throughout all stages. PPO never receives gradients, optimizer steps,
or mutable training-mode updates. There is no PPO training buffer, weight
transfer, behavior cloning, or PPO value bootstrap.

## Source configurations

Dedicated local trial YAMLs snapshot round 4_1. Runtime preparation reads these
local files, rather than regenerating settings from the tuning project.

| Setting | Trial 97 | Trial 98 |
| --- | --- | --- |
| Replay refresh per collection round | 0.1 | 0.5 |
| Replay capacity per rank | 8,192,000 | 8,192,000 |
| Epochs over replay per training round | 3 | 3 |
| Logical minibatch / microbatch per rank | 128,000 / 32,000 | 128,000 / 32,000 |
| SAC-only `carla_fast` evaluation interval | 10 rounds | 2 rounds |

Both retain gamma 0.999, tau 0.005, initial alpha 0.1, learned temperature with
target entropy 1.863680, source seed/model/environment settings, two ranks, and
no wall-time cutoff. This experiment does **not** inherit the fixed-temperature
override from the earlier warm-start experiment.

The changed settings are the 1B budget, frozen-teacher behavior schedule, separate
run identity/output paths, and W&B group `sac_ppo_init_behavior_round4_1`.
Replay alone requires approximately 112.9 GiB host RAM across two ranks, before
simulators, SAC networks, and teacher weights. GPU memory and throughput require
measurement. There is no PPO rollout or optimizer allocation.

## Run one trial

Each invocation selects one two-GPU job. Run the two trials separately; do not
launch both onto the same GPU pair concurrently.

1. Activate the repository environment.

   ```bash
   cd /common/users/ql236/auto_driving/PufferDrive
   source .venv/bin/activate
   ```

2. Preview trial 97 without training.

   ```bash
   bash project/soft_actor_critic_PPO_init_behavior_policy/scripts/launch.sh \
     --trial 97 --gpus 0,1 --dry-run
   ```

3. Start trial 97 in a fresh run directory. This prepares and starts training.

   ```bash
   bash project/soft_actor_critic_PPO_init_behavior_policy/scripts/launch.sh \
     --trial 97 --gpus 0,1
   ```

4. After that run completes, start trial 98.

   ```bash
   bash project/soft_actor_critic_PPO_init_behavior_policy/scripts/launch.sh \
     --trial 98 --gpus 0,1
   ```

Use `--output-dir PATH` to select a fresh output directory. A dry run writes a
resolved config and prints a direct command for that config; alternatively a
normal launcher invocation creates a fresh run as shown above. Do not invoke the
launcher again with a directory it already created. W&B is enabled by default;
`--no-wandb` disables experiment loggers and model uploads. Teacher paths and
representation compatibility are checked before training.

The entry point is `pufferlib.pufferl_sac_ppo_init_behavior_policy`. It calls native
`pufferl_sac.train()` with an optional behavior provider. Native SAC checkpoints
and benchmark evaluation continue to contain/use the SAC actor alone. Exact
interrupted-run resume is not provided because native checkpoints omit replay,
pending teacher-source labels, and simulator state. Use a fresh run after a
failed or stopped launch.

## Collection, replay, and learning

During mixed collection, choose PPO or SAC independently for each received
agent's action. Sample from the selected actor's categorical distribution; do not
average logits or physical controls. Teacher and SAC action tables are checked
for exact equality. Replay stores the actual action index sent through the
shared action-to-control conversion. Pure PPO and pure SAC phases only evaluate
the selected actor; mixed batches evaluate both actor networks. PPO's unused
value head is never evaluated.

The source of a replay transition is the actor that issued its previous action,
not the actor selected for the next send. Stable agent-slot IDs, masks, terminal
zero discounts, and pure-truncation final observations retain native semantics.
In multi-agent simulation, a source label identifies one action's actor, not the
provenance of an entire joint trajectory.

All SAC Bellman targets use the current SAC policy with target critics. PPO is
only the behavior policy. The one-step native SAC update needs no additional
behavior importance weight. Teacher data do not guarantee accurate Q estimates
for actions rarely taken by PPO; SAC-only evaluations are necessary to assess
learning.

Replay uses native cyclic replacement and uniform shuffled training passes.
Collection probability and replay composition differ: older PPO data remain
while new SAC transitions replace them. After the final pending PPO transition
is handled, one full traversal removes all remaining teacher entries: ten full
refresh rounds for trial 97 or two for trial 98. Both require 16,384,000 valid
inserts globally. Received rows can exceed that number. No permanent teacher
buffer or extra source-biased replay sampling is introduced.

## Metrics and interpretation

`behavior/ppo_probability` is the collection probability. The provider also logs
`behavior/ppo_action_fraction`, `behavior/ppo_insert_fraction`,
`behavior/ppo_replay_fraction`, and teacher replay count, plus stage and collection
start counters. Counts/fractions aggregate across ranks. At beta zero, pending
teacher actions can still produce a final teacher transition.

Training `environment/*` metrics describe mixed behavior while PPO is active;
they cannot establish SAC performance. Scheduled `carla_fast` evaluations and
native final evaluation run the SAC actor alone. Compare those driving metrics
at matched new received steps, and report wall time and teacher pretraining cost
separately. The schedule is fixed; it does not imply SAC has surpassed PPO at 500M.

Configuration and unit checks establish setup correctness, not learning quality.
See `VALIDATION.md` for the checks actually run before launch.
