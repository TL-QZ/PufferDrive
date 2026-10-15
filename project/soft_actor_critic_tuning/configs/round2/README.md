# SAC round two: entropy, discount, and target-critic sweep

Status: configuration prepared; training has not been launched by this setup.

## 1. Objective and hypotheses

Find a SAC configuration that learns useful driving and sustains it, starting
from round-one trial 6. That run delayed the return collapse, but did not achieve
reliable driving progress. Its alpha and soft Q values increased while evaluated
return deteriorated. The original artifacts remain in
[`runs/sac16_grid_20261012T032408Z/trial_06/`](../../runs/sac16_grid_20261012T032408Z/trial_06/).

This experiment tests three controls together:

- **Target entropy:** allow more selective policies instead of requiring almost
  maximal entropy. The original 98% target is retained as a control.
- **Gamma:** test whether shorter discounted horizons improve useful learning
  by reducing dependence on distant bootstrapped values and entropy bonuses.
  Shorter horizons can also undervalue distant goals.
- **Tau:** test both slower and faster target-critic tracking. The previous
  0.005–0.01 interval spanned only 2x; the new range spans 50x. Faster tracking
  is not inherently better because it also propagates online-critic errors.

Soft Q includes future entropy bonuses. Rising mean Q alone does not prove
estimation error, and flattening Q alone is not the experiment's success criterion.
The ranges are diagnostic choices, not established optima for driving. Relevant
background: [SAC targets](https://spinningup.openai.com/en/latest/algorithms/sac.html)
and [target entropy in discrete SAC](https://arxiv.org/abs/2112.02852).

## 2. Exact grid and control

The launcher takes the Cartesian product of the `controlled_exp` value lists:
**4 x 4 x 4 = 64 trials**, one training seed per combination.

| Parameter | Four values |
| --- | --- |
| `train.tau` | `0.001`, `0.005`, `0.01`, `0.05` |
| `train.gamma` | `0.95`, `0.98`, `0.99`, `0.999` |
| `train.target_entropy` | `0.621227`, `1.242453`, `1.863680`, `2.435209` |

### Trial-ID map

Each cell below is a **trial ID**. Tau varies slowest, gamma next, and entropy
fastest. `H25` means 25% of maximum policy entropy, not a 25% random-action
probability; the column headers also give the absolute target entropy in nats.

| Tau | Gamma | H25: 0.621227 | H50: 1.242453 | H75: 1.863680 | H98: 2.435209 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.001 | 0.95 | 1 | 2 | 3 | 4 |
| 0.001 | 0.98 | 5 | 6 | 7 | 8 |
| 0.001 | 0.99 | 9 | 10 | 11 | 12 |
| 0.001 | 0.999 | 13 | 14 | 15 | 16 |
| 0.005 | 0.95 | 17 | 18 | 19 | 20 |
| 0.005 | 0.98 | 21 | 22 | 23 | 24 |
| 0.005 | 0.99 | 25 | 26 | 27 | 28 |
| 0.005 | 0.999 | 29 | 30 | 31 | 32 |
| 0.01 | 0.95 | 33 | 34 | 35 | 36 |
| 0.01 | 0.98 | 37 | 38 | 39 | 40 |
| 0.01 | 0.99 | 41 | 42 | 43 | 44 |
| 0.01 | 0.999 | 45 | 46 | 47 | **48 — control** |
| 0.05 | 0.95 | 49 | 50 | 51 | 52 |
| 0.05 | 0.98 | 53 | 54 | 55 | 56 |
| 0.05 | 0.99 | 57 | 58 | 59 | 60 |
| 0.05 | 0.999 | 61 | 62 | 63 | 64 |

Read across a row to compare entropy targets with tau and gamma held fixed.
Within a tau block, read down an entropy column to compare gamma values.
For completed or running experiments, use that run's saved `manifest.json`
as the authoritative mapping if the configuration is edited later.

Quick comparisons around control trial 48:

| Parameter varied | Trial IDs | Settings held fixed |
| --- | --- | --- |
| Target entropy | **45, 46, 47, 48** | Tau 0.01; gamma 0.999 |
| Gamma | **36, 40, 44, 48** | Tau 0.01; target entropy 2.435209 |
| Tau | **16, 32, 48, 64** | Gamma 0.999; target entropy 2.435209 |

### Tau: how quickly the target critic follows the online critic

The previous sweep tested only `0.005` and `0.01`, a **2x range**. This sweep
covers **50x**, retaining both previous values and adding substantially slower
and faster tracking. The old search was narrow; that does not establish that
its tau values were too small.

After every logical learner update:

```text
target <- (1 - tau) * target + tau * online
```

| Tau | Updates until the initial target's weight halves | Interpretation |
| --- | ---: | --- |
| `0.001` | 693 | Strong smoothing; tests whether slow target changes stabilize learning, with a risk of stale targets. |
| `0.005` | 138 | Slower of the two round-one settings. |
| `0.01` | 69 | Round-one trial 6's setting; retained in the control. |
| `0.05` | 14 | Fast tracking; tests whether target lag is limiting learning, but propagates online-critic errors faster. |

The half-life is `log(0.5) / log(1 - tau)` learner updates. It measures decay
of the initial target parameters' contribution, not the accuracy of Q estimates.
Some references use the complementary coefficient `polyak = 1 - tau`.
[SAC target-network reference](https://spinningup.openai.com/en/latest/algorithms/sac.html).

**Replay size changes the per-cycle meaning.** The current 512,000-entry replay,
128,000-row minibatch, and three passes give **12 updates per cycle**, versus
192 in round-one trial 6. At `tau=0.01`, the initial target contribution decays
by about **11.4% per cycle**, versus **85.5% previously**. Thus the same tau
has the same per-update smoothing but slower tracking measured in cycles.
These figures use the current resource settings, replacing the earlier
2,048,000-entry / 48-update-per-cycle example discussed during planning.

### Gamma: how far into the future the critic values outcomes

At training `dt=0.3`, the approximate continuing-task discount timescale is
`1 / (1 - gamma)` steps:

| Gamma | Discount timescale (steps) | Simulated time | Interpretation |
| --- | ---: | ---: | --- |
| `0.95` | 20 | 6 seconds | Deliberately short-horizon diagnostic: less dependence on distant bootstrapped values, but may undervalue distant goals. |
| `0.98` | 50 | 15 seconds | Intermediate horizon for learning useful near-term driving. |
| `0.99` | 100 | 30 seconds | Longer horizon while reducing dependence on distant predictions relative to the original setting. |
| `0.999` | 1,000 | 300 seconds | Original setting; retained in the control. |

These are **discount timescales, not episode lengths or hard cutoffs**.
Termination shortens the actual horizon. Gamma discounts both future task
rewards and future entropy bonuses, so lowering it changes the objective as
well as sensitivity to bootstrapping errors. Smaller is not automatically better.

### Target entropy: how selective the actor is allowed to become

The actor has 12 jerk action classes, giving maximum entropy `log(12)`.
The config expects **absolute nats**, not a fraction:

| Fraction of maximum entropy | `train.target_entropy` (nats) | Interpretation |
| --- | ---: | --- |
| 25% | `0.621227` | Low entropy pressure; allows substantially more selective policies, with a risk of reducing exploration too early. |
| 50% | `1.242453` | Moderate entropy pressure; tests a balance between selectivity and exploration. |
| 75% | `1.863680` | Higher entropy pressure, while relaxing the original near-uniform target. |
| 98% | `2.435209` | Original entropy target, rounded to six decimals; retained in the control. |

The values equal `[0.25, 0.50, 0.75, 0.98] * log(12)`, rounded to six decimal
places. These percentages are **not random-action probabilities**. Automatic
temperature tuning raises alpha when measured entropy is below the target and
lowers it when entropy is above the target; realized entropy need not equal
the target during learning.

Retaining 98% lets us test whether reducing entropy pressure actually helps.
We avoid zero in this first grid because removing exploration pressure too
early may let the actor commit to an insufficiently trained critic's preferred
actions. Research identifies failure modes at both high and low fixed targets;
these exact four values are diagnostic choices, not proven driving optima.
[Target Entropy Annealing for Discrete SAC](https://arxiv.org/abs/2112.02852).

### Reduced-resource control

Tau varies slowest, gamma next, and entropy fastest. IDs are one-based.
**Trial 48 is the reduced-resource control:** tau `0.01`, gamma `0.999`, target
entropy `2.435209`. It uses round-one trial 6's learning settings with the new
single-GPU resource budget. It is not a bitwise reproduction of the original
run: rank count, replay size, aggregate optimizer batch, and training budget differ.
Each trial starts from scratch; no collapsed checkpoint is resumed or warmstarted.

## 3. Fixed settings and resource budget

| Setting | Round two |
| --- | --- |
| GPUs per trial / default concurrent trials | 1 / 8 |
| Replay capacity (`train.batch_size`) | 512,000 transitions |
| Logical minibatch / maximum microbatch | 128,000 / 32,000 |
| `update_epochs` / `replay_update_ratio` | 3 / 0.1 |
| `initial_alpha` / `alpha_learning_rate` | 0.1 / 0.0003 |
| Actor LR / critic LR | 0.0005 / 0.0005 |
| Optimizer / gradient clipping / weight decay | AdamW / 0.5 / 0.01 |
| Training seed / vector seed | 0 / 42 |
| Precision / compilation / replay placement | BF16 / enabled / CPU |
| LR annealing | disabled |
| `total_timesteps` | 1,000,000,000 received agent rows per trial |
| Wall-time target / shutdown grace | 3 hours / 600 seconds per trial |
| Environment workers / agent slots per worker | 20 / 3,200 |
| Scheduled evaluation | `carla_fast` after every collection/training cycle |

Network, reward, environment, and evaluation settings remain inherited from the
round-one snapshot. Training `dt=0.3`, evaluation `dt=0.1`, and evaluation's mean
physical-action selection remain fixed throughout this comparison.

The learner makes 4 logical updates per replay pass and **12 per cycle**.
After initial replay fill, each cycle inserts 51,200 valid transitions and
samples 1,536,000 replay rows: approximately 30 replay sample uses per new valid
transition. Tau and alpha update once per logical learner update.

CPU replay is approximately **3.5 GiB per trial** with 923 observation features.
Eight concurrent trials need about **31.1 GiB for replay plus 10% headroom**;
simulator, Python, checkpoint, and other process memory is additional. Eight
trials use 160 environment workers. Model and microbatch size are unchanged,
so peak GPU memory should not be assumed to shrink with replay capacity.

The nominal budget is **192 GPU-hours**, or approximately **24 hours at eight
concurrent trials**, plus shutdown overhead. Trials may stop earlier at the
step ceiling or to avoid starting a cycle beyond the wall-time target. The
1-billion-step ceiling is not a guarantee that every trial reaches that many
steps. Compare performance against both agent steps and elapsed time.

## 4. Preview and launch

Run from the PufferDrive repository root. The wrapper activates `.venv` and
limits numerical-library threads. The default launcher configuration remains
round one; **round two must be selected explicitly**.

Preview all 64 configurations without starting training:

```bash
bash project/soft_actor_critic_tuning/scripts/launch_sweep.sh \
  --config-dir project/soft_actor_critic_tuning/configs/round2 --dry-run
```

Launch the full grid on GPUs 0–7, one GPU per trial:

```bash
bash project/soft_actor_critic_tuning/scripts/launch_sweep.sh \
  --config-dir project/soft_actor_critic_tuning/configs/round2 \
  --gpu-groups '0;1;2;3;4;5;6;7' --max-concurrent 8
```

Use available GPU IDs for the intended machine. Semicolons separate trials;
`'0,1'` instead assigns two GPUs to one trial. `--gpu-pairs` remains a legacy
alias. W&B uses the existing authenticated session; `--no-wandb` disables
external logging if desired.

To launch only the reduced-resource control:

```bash
bash project/soft_actor_critic_tuning/scripts/launch_sweep.sh \
  --config-dir project/soft_actor_critic_tuning/configs/round2 \
  --trial-ids 48 --gpu-groups '0' --max-concurrent 1
```

The launcher retains its historical filename `run_sac16_grid.py`, but derives
trial count from the selected config. It supports both the original 16-trial
round one and this 64-trial round two. It reads `controlled_exp`, not `sweep`
search distributions or `max_runs`.

## 5. Artifacts and interpretation

Round-two runs are isolated under:

```text
project/soft_actor_critic_tuning/runs/round2/sac64_grid_<UTC timestamp>/
  manifest.json
  configs/trial_01.yaml ... trial_64.yaml
  trial_XX/
    metrics.jsonl
    summary.json
    rank_0_runtime.json
    learner/
      config.yaml
      final_model.pt
      trainer_state.pt
      eval/training/carla_fast/epoch_.../evaluation_summary.json
```

The manifest records config directory, axes, GPU assignments, and trial status.
W&B project: `pufferdrive_vanilla_sac_tuning`; group: **`vanilla_sac_round2`**.
The four YAMLs here are an independent snapshot; [`../round1/`](../round1/)
retains the original settings. Loading never falls back to another round's files.

Evaluate results using these criteria:

- **Useful driving:** improve making-progress rate, progress ratio, and goals
  reached alongside episode return, compared with trial 48. Near-zero return
  caused by staying still does not qualify.
- **Sustained learning:** inspect the full progression and late-run behavior,
  rather than selecting an isolated return peak. Check speed and infractions
  to distinguish useful movement from unsafe motion.
- **Learner interpretation:** inspect `losses/alpha`, `losses/entropy`,
  `losses/q1`, `losses/q2`, `losses/target`, and `losses/critic_loss`. Q1/Q2 are
  mean value predictions, not loss values. Compare entropy with each trial's
  own target rather than expecting a common entropy across the grid.
- **Comparable evidence:** every `carla_fast` benchmark uses 250 scenarios and
  500 steps. Use its per-cycle summaries for continuous progress evidence;
  online training episode metrics can be absent in some cycles and describe
  collection before the corresponding learner updates.
- **Limits:** this is a one-seed screening experiment. Promising settings need
  repeated seeds and longer evaluation before a robust-performance claim.

Periodic model checkpoints remain disabled; normal completion saves final
actor and learner checkpoints. Per-cycle evaluation summaries remain available,
but earlier peak policies are not automatically retained. No success or
performance improvement is claimed before these experiments run.

### Preparation validation

The launcher tests passed (10 tests, 8 subtests). A full dry run verified all
64 unique parameter combinations, fixed trial-6 settings, single-GPU commands,
and control trial 48. Round-one YAML contents were verified unchanged.
The generated [preview manifest](../../runs/round2/sac64_grid_20261014T025617Z/manifest.json)
and its sibling `configs/` folder are local ignored artifacts, not training
results. GPU training, peak memory, throughput, and policy improvement have
not been tested for this experiment.
