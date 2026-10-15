# SAC round three: update frequency and replay refresh

Status: configuration preparation only; no training launched by this setup.
Round three uses the small, single-GPU round-two resources. For each of the
three selected round-two candidates, sweep `update_epochs = [3, 15, 50, 100]`
and `replay_update_ratio = [0.1, 0.5, 0.75]`: **36 selected trials**.
Each trial has a **three-hour wall-time target** and a **500-million received
agent-row ceiling**, whichever stops it first. The latest 500-million budget
is retained; round two itself has a one-billion ceiling.

## 1. Candidate settings held fixed

These selections reflect the user's round-two analysis, not newly verified
performance claims. Every trial uses **tau = 0.005** and initial alpha 0.1.
Gamma and entropy stay paired within each candidate.

| Source round-two candidate | Gamma | Target entropy (nats) | Fraction of maximum | Round-three trial IDs |
| --- | ---: | ---: | ---: | --- |
| 17 | 0.95 | 0.621227 | H25 | 1–12 |
| 26 | 0.99 | 1.242453 | H50 | 49–60 |
| 31 | 0.999 | 1.863680 | H75 | 97–108 |

Round-three IDs identify new experiments, not the old round-two IDs. Each
candidate starts from scratch; no round-two checkpoint is resumed.

## 2. Sweep intuition and learning cadence

`update_epochs` is the number of shuffled passes over the entire replay buffer
before the next collection cycle. `replay_update_ratio` is the fraction of
replay slots replaced with new transitions after the initial full replay fill;
it is not the fraction sampled for training. Every training pass uses the
whole replay, including retained and newly collected transitions.

With replay capacity 512,000 and logical minibatch 128,000, each pass has four
optimizer updates. The sweep tests how much learning to do between collections
and how fresh the replay should be for that learning.

| Update epochs | Optimizer updates per cycle | Question tested |
| ---: | ---: | --- |
| 3 | 12 | Round-two baseline: frequent returns to collection. |
| 15 | 60 | Does five times more replay training improve learning? |
| 50 | 200 | Can substantially more reuse help before stale data hurts? |
| 100 | 400 | Stress-test heavy reuse and long intervals between collection. |

| Replay update ratio | New transitions inserted per refresh | Old slots retained | Question tested |
| ---: | ---: | ---: | --- |
| 0.1 | 51,200 | 90% | Round-two baseline: retain most experience. |
| 0.5 | 256,000 | 50% | Does refreshing half the replay support more updates? |
| 0.75 | 384,000 | 25% | Does fresher replay improve stability at high epoch counts? |

A useful nominal reuse measure is sampled training rows divided by newly
inserted rows: `update_epochs / replay_update_ratio`, after warmup. For example,
3/0.1 gives 30, 15/0.5 also gives 30, and 100/0.1 gives 1,000. Equal ratios do
not imply identical behavior: collection frequency, experience age, and the
number of target/temperature updates between collections differ. Received
agent rows can exceed inserted transitions, so this measure is not an exact
optimizer-update-to-environment-step ratio.

Tau remains 0.005 **per optimizer update**. More epochs also mean more target
and temperature updates per collection cycle. Benchmark evaluation remains
every training cycle, so higher epoch counts change evaluation frequency per
optimizer update and per hour.

## 3. Trial-ID map

Replay ratio varies fastest, then epoch count. Each row below compares the
same cadence across all three candidate policies.

| Update epochs | Replay update ratio | Source 17 (H25) | Source 26 (H50) | Source 31 (H75) |
| ---: | ---: | ---: | ---: | ---: |
| 3 | 0.1 | 1 | 49 | 97 |
| 3 | 0.5 | 2 | 50 | 98 |
| 3 | 0.75 | 3 | 51 | 99 |
| 15 | 0.1 | 4 | 52 | 100 |
| 15 | 0.5 | 5 | 53 | 101 |
| 15 | 0.75 | 6 | 54 | 102 |
| 50 | 0.1 | 7 | 55 | 103 |
| 50 | 0.5 | 8 | 56 | 104 |
| 50 | 0.75 | 9 | 57 | 105 |
| 100 | 0.1 | 10 | 58 | 106 |
| 100 | 0.5 | 11 | 59 | 107 |
| 100 | 0.75 | 12 | 60 | 108 |

The existing Cartesian-grid launcher enumerates gamma, entropy, epochs, then
ratio (108 combinations). `launcher.trial_ids` selects only the **36 paired
candidate combinations** above. The other 72 are marked skipped in the
manifest and receive no resolved config or training process. This avoids
launching the six unrequested gamma/entropy pairings. Explicit `--trial-ids`
overrides that selection, so use the round-three IDs in this table.

Trials **1, 49 and 97** repeat the round-two cadence (3 epochs, 0.1 refresh)
for candidates 17, 26 and 31 respectively.

## 4. Resources and stopping

| Setting | Round-three value |
| --- | ---: |
| GPUs per trial | 1 |
| Replay transitions | 512,000 |
| Logical minibatch | 128,000 |
| Maximum microbatch | 32,000 |
| Received-agent-row ceiling | 500,000,000 |
| Wall-time target | 3 hours |
| Shutdown grace | 600 seconds |
| Default concurrent trials | 8 |
| Environment instances per trial | 20 |
| Agent slots per instance | 3,200 |

Environment, network, optimizer, compilation, CPU replay, and evaluation
settings retain the round-two configuration. Training seed remains 0; W&B
group is `vanilla_sac_round3`. Round-one and round-two configs are unchanged.

The child checks time between complete cycles and can stop early when another
cycle is unlikely to fit. The parent can terminate a process at three hours
plus the 600-second grace. A long cycle, especially with 100 epochs, may
consume the grace; a hard termination cannot guarantee final artifacts. The
three-hour setting is therefore a wall-time target, not a promise of exactly
three hours of learning. The step ceiling, nonfinite-loss checks, errors, and
explicit cancellation can also end a run.

## 5. Preview, launch, and compare

From the repository root, preview the 36 selected trials without training:

```bash
bash project/soft_actor_critic_tuning/scripts/launch_sweep.sh \
  --config-dir project/soft_actor_critic_tuning/configs/round3 --dry-run
```

Launch when ready (defaults to GPUs 0–7, eight concurrent trials):

```bash
bash project/soft_actor_critic_tuning/scripts/launch_sweep.sh \
  --config-dir project/soft_actor_critic_tuning/configs/round3
```

For four available GPUs, append `--gpu-groups '0;1;2;3' --max-concurrent 4`.
To run the three baseline-cadence controls, append `--trial-ids 1,49,97`.
Smoke mode uses trial 1 with reduced resources and a short time limit.

Default artifacts: `runs/round3/sac108_grid_<timestamp>/`, with selected
folders `trial_01` through `trial_12`, `trial_49` through `trial_60`, and
`trial_97` through `trial_108`. The `sac108` prefix reflects enumeration;
**only 36 trials run**.

Within each candidate, compare epoch counts at fixed refresh, then refresh
ratios at fixed epochs. Judge sustained return, score, pufferscore, progress,
goals and infractions; use Q, entropy and alpha as supporting diagnostics.
Compare both matched received-agent steps (sample efficiency) and matched
wall time (compute efficiency), since these settings change collection and
training throughput. One seed per configuration does not establish robustness.
