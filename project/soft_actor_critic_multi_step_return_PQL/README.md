# Peng Q(lambda) discrete SAC

The implementation adds sequence replay and a detached backward lambda-return
scan to the existing SAC learner. It shares SAC's discrete soft bootstrap,
critic/actor losses, optimizers, alpha/TES updates, and target-network updates.
The existing PPO GAE kernel is a numerical test oracle; its rollout layout and
PPO-specific correction inputs are not used in the SAC production path.

See [PLAN.md](PLAN.md) for the target equations and implementation design, and
[the validation report](reports/implementation_validation.md) for completed
checks. No official training run was launched.

## PQL_official_round1

The paired tuning round is in [`configs/PQL_official_round1`](configs/PQL_official_round1).
Trial **98** uses **50%** FIFO replacement and trial **01** uses **25%**.
Both use the Round 4.1 trial-98 settings as a shared baseline, with the Round 2
replacement rate and code name retained for trial 01. This aligns the original
recipes so replacement rate is the only differing training setting.

Common settings:

- 20 environments × 800 agent slots = 16,000 slots per rank; replay capacity
  2,048,000 transitions gives nominal room for 128 transitions per slot.
- PQL horizon 128, lambda 0.95, raw `n_steps=1`, gamma 0.999, and critic-target
  entropy enabled. Episode boundaries and gaps can shorten target suffixes.
- Initial actor/critic learning rates `1e-4`; alpha learning rate `7e-5`.
  Existing cosine learning-rate annealing remains enabled for all three.
- Adaptive target entropy starts at 1.863680, EMA decay 0.999, mean threshold
  0.01, **std threshold 0.025**, **2,000 cumulative qualifying updates**, decay
  factor 0.9, and a forced reduction after 500M global received rows without
  a reduction.
- Both trials use a 4B global-row budget, three replay epochs, 64,000-row logical
  batches, 32,000-row microbatches, evaluation every two collection cycles, and
  collapse diagnostics enabled.

The different refresh rates imply different update-to-new-data ratios and
evaluation spacing in received rows despite identical update/evaluation settings.
With a full replay, each cycle replaces 1,024,000 rows per rank for 98 and
512,000 for 01. The 25% variant therefore performs about twice as many learner
updates per new replay row; adaptive entropy qualifying counts remain measured
in learner updates, while the forced-decay timer uses global received rows.

Preview either trial without training:

```bash
bash project/soft_actor_critic_multi_step_return_PQL/scripts/launch_PQL_official_round1.sh --trial 98 --dry-run --output-dir /tmp/PQL_official_round1_98
bash project/soft_actor_critic_multi_step_return_PQL/scripts/launch_PQL_official_round1.sh --trial 01 --dry-run --output-dir /tmp/PQL_official_round1_01
```

For a training launch, remove `--dry-run` and the preview `--output-dir`, and
specify the intended two GPUs with `--gpus ID,ID`. No training is launched as
part of configuration preparation.

## PQL_official_round2: original goal spacing

[`configs/PQL_official_round2`](configs/PQL_official_round2) copies the Round 1
pair with original goal spacing restored: **20–200 m in training** and
**30 m in evaluation**, compared with 6.67–66.67 m and 10 m in Round 1.
Trial 98 remains 50% replacement; trial 01 remains 25%. All learner settings,
including 800 agents per environment, PQL horizon 128, raw `n_steps=1`, learning
rates, and entropy schedule, are unchanged. PQL still constructs multi-step
targets from linked replay rows.

Use `scripts/launch_PQL_official_round2.sh --trial 98 --dry-run` or
`--trial 01 --dry-run` (with the full project path as in the Round 1 commands).
Run names, output directories, W&B group, and benchmark path are isolated.
Because evaluation spacing also changes, compare the rounds' scores with that
benchmark difference in mind, or evaluate both checkpoints on one shared catalog.

## Original source-derived sets

This project also contains two isolated PQL derivatives of existing discrete SAC
recipes. Both keep the source recipe's selected grid trials and settings, while
using raw one-step replay rows (`train.n_steps: 1`) and `train.return_estimator:
pql` with horizon 128, lambda 0.95, and endpoint-soft entropy. The source configs
omit `critic_entropy_in_target`; PQL configs pin the inherited learner default
`true` explicitly.

| PQL set | Source recipe | Supported trial selectors | Preserved trial budgets |
|---|---|---|---|
| `round4_1_pql` | `soft_actor_critic_multi_step_return/configs/round4_1_nstep2_tes_sac` | 97, 98 | 2B and 4B global received rows |
| `round2_enhance_pql` | `soft_actor_critic_multi_step_return/configs/round2_enhance` | 01 or 1 | 4B global received rows |

Each set has its own W&B group, learner data path, launcher output path, and a
byte-for-byte copy of its source evaluation benchmark catalog. The grid ratios,
learning rates, optimizer, environment, evaluation cadence, and other recipe
settings remain inherited from the source. The source `SAC_run_config.yaml`
keeps the selected trial numbering and trial-specific budget/evaluation maps.

## Preview commands

Activate the repository environment, then preview every supported selector.
These commands write only sweep manifests and resolved configs under `/tmp`;
`--dry-run` does not start the learner.

```bash
source .venv/bin/activate
bash project/soft_actor_critic_multi_step_return_PQL/scripts/launch_round4_1_pql.sh --trial 97 --dry-run --output-dir /tmp/pufferdrive_pql_preview/round4_1_trial97
bash project/soft_actor_critic_multi_step_return_PQL/scripts/launch_round4_1_pql.sh --trial 98 --dry-run --output-dir /tmp/pufferdrive_pql_preview/round4_1_trial98
bash project/soft_actor_critic_multi_step_return_PQL/scripts/launch_round2_enhance_pql.sh --trial 01 --dry-run --output-dir /tmp/pufferdrive_pql_preview/round2_trial01_alias
bash project/soft_actor_critic_multi_step_return_PQL/scripts/launch_round2_enhance_pql.sh --trial 1 --dry-run --output-dir /tmp/pufferdrive_pql_preview/round2_trial1
```

A command without `--dry-run` starts training. For an intentional official
launch, run one supported trial at a time, for example:

```bash
bash project/soft_actor_critic_multi_step_return_PQL/scripts/launch_round4_1_pql.sh --trial 98
```

## Horizon and memory limits

`pql_horizon: 128` is a maximum segment length; it does not guarantee that each
loss row has 128 observed rewards. Episode boundaries, invalid rows, replay
residency, and segment packing can shorten a target. The source topology has
2,048,000 replay rows across 64,000 environment agent slots, or about 32 rows
per slot on average. Measure available sequence lengths and effective target
suffixes before treating 128 as a practical horizon.

The source observation has 923 float32 features. Observation-plus-successor
storage alone is estimated at 14.084 GiB per rank for 2,048,000 rows; this
excludes scalar/link arrays, temporary plans, transfers, and network activations.
CPU-offloaded replay still consumes host memory. Retaining 128 rows for every
slot would require 8,192,000 rows and about 56.335 GiB per rank for observation
pairs alone. Treat these as planning estimates and measure actual resource use
in a short feasibility run before any long training run.

## Validation and provenance

The focused tests compare each merged PQL config against its source recipe,
allowing only the PQL controls and isolated identities/catalog path to differ.
They also dry-run each source/PQL launcher pair for every supported selector
and compare the generated per-trial configs. Run them with:

```bash
source .venv/bin/activate
python -m unittest discover -s project/soft_actor_critic_multi_step_return_PQL/tests -v
```

The exact config sources are documented in each `SAC_run_config.yaml` and the
table above. Benchmark contents are cloned from those source directories; only
the PQL configs' `eval.benchmark_config` path points into this project.

## Runtime coverage metrics

`replay/pql_rank0/` reports rank-zero sequence lengths, per-row target suffix
lengths, boundary cuts, replay age, and retained tail weight. Distribution
summaries describe the last replay epoch's plan; gap/overwrite/stale-reference
event counts accumulate until logging, then reset. Replay age is measured in
accepted rank-local insertion rows, not seconds or per-agent environment steps.
Use the suffix fractions at 32, 64, and 128 to inspect the realized horizon.
