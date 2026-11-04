# Peng Q(lambda) discrete SAC

The implementation adds sequence replay and a detached backward lambda-return
scan to the existing SAC learner. It shares SAC's discrete soft bootstrap,
critic/actor losses, optimizers, alpha/TES updates, and target-network updates.
The existing PPO GAE kernel is a numerical test oracle; its rollout layout and
PPO-specific correction inputs are not used in the SAC production path.

See [PLAN.md](PLAN.md) for the target equations and implementation design, and
[the validation report](reports/implementation_validation.md) for completed
checks. No official training run was launched.

This project contains two isolated PQL derivatives of existing discrete SAC
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
