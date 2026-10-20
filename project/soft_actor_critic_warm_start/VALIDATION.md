# Warm-start validation

Validation date: 2026-10-19. These checks establish reduced-scale behavior, not full-scale learning performance.

## What now works

The coordinator executes native PPO collection and updates, refreshes SAC replay from completed transitions, trains native SAC critics against the updated PPO actor, and switches to native SAC actor learning without resetting the vector environment. Fixed temperature is an explicit native SAC option; the default remains automatic temperature tuning.

The dedicated [`config/warm_start.yaml`](config/warm_start.yaml) records the original baseline and resolved trial 97 settings with hybrid overrides. Policy and environment dictionaries match exactly. The full profile retains the 10G PPO cosine schedule, switches near 2G collected global agent rows, and continues for 2G additional rows (approximately 4G cumulative). The source baseline and tuning configs were not modified.

The dedicated-config migration was checked against a snapshot of the previous resolver: every retained PPO/SAC field matches exactly. Only unused sweep/profile catalogs were removed. Five config tests cover the defaults, edited budgets/critic learning rate, YAML and CLI seeds, environment validation, and smoke isolation. The launcher still writes a resolved YAML for each run.

## Evidence

Final combined regression run: **78 passed in 15.12 seconds**. The two existing distributed tests passed separately in 34.73 seconds with localhost socket access. The component results below overlap the combined run; do not add them as independent test totals.

| Check | Result |
| --- | --- |
| Native SAC models/replay/seeding/learner/final-observation regression suite | 42 passed, 1 skipped |
| Native PPO observer and new SAC interface tests | 10 passed |
| Existing SAC two-process Gloo tests | 2 passed; required localhost socket access outside the sandbox |
| Actor transfer with real Drive/DriveSAC modules | Matching probabilities, action ordering, copied non-aliased actor weights, and gradients on synthetic observations |
| Paired native PPO versus hybrid critic-pretraining test | Two complete cycles; all PPO state tensors and sampled actions match exactly on CPU with deterministic synthetic vector observations |

The paired test also verifies that critic updates actually occurred. It is stronger than checking only copied actor weights, but does not establish GPU/compiled numerical parity or an identical full driving training curve.

## Real-simulator CPU smoke runs

Both use the real Drive simulator and reduced native model/buffer settings, with benchmark evaluation, BF16, and compilation disabled.

| Quantity | One rank | Two ranks |
| --- | ---: | ---: |
| PPO cycles | 2 | 2 |
| Global steps at handoff | 128 | 256 |
| Global steps at completion | 192 | 384 |
| Critic optimizer updates per rank before switch | 6 | 6 |
| Total SAC learner updates per rank at completion | 30 | 30 |
| Maximum actor probability difference at transfer | 0 | 0 |
| Final actor/online-critic/target-critic tensors | All finite | All finite |
| Temperature | Fixed approximately 0.1 (FP32 representation) | Fixed approximately 0.1 |

Temporary artifacts from these checks:

- `/tmp/sac_ppo_warm_start_smoke_20261019b`
- `/tmp/sac_ppo_warm_start_ddp_20261019`
- `/tmp/sac_ppo_warm_start_smoke_auto_20261019`

The final single-rank run uses `batch_size: auto`, as the full baseline does, and resolves it to 64 rows in the reduced profile. It completed after the final checkpoint/RNG and default-batch fixes. Independent review reported no remaining material findings.

To rerun the focused warm-start checks:

```bash
source .venv/bin/activate
OMP_NUM_THREADS=1 python -m pytest -q \
  tests/unit_tests/test_ppo_transition_observer.py \
  tests/unit_tests/test_sac_warm_start_interfaces.py \
  tests/unit_tests/test_sac_ppo_warm_start.py \
  tests/unit_tests/test_warm_start_actor_parity.py \
  project/soft_actor_critic_warm_start/tests/test_configuration.py
```

The boundary sidecar references an immutable `sac/boundary_trainer_state.pt` and `sac/boundary_actor.pt`; later SAC saves must not overwrite these. Native learner checkpoints omit replay and live environment state, so the hybrid explicitly rejects resume rather than promising equivalent continuation.

The first smoke exposed a pre-existing native PPO edge: a one-row partial minibatch makes unbiased advantage standard deviation undefined. The reduced smoke profile uses a whole-rollout PPO minibatch to avoid this. Native PPO loss behavior was not changed.

## Limits

No accessible NVIDIA driver was available on this host. GPU/NCCL behavior, compiled/BF16 parity, full replay allocation, peak host/VRAM usage, throughput, and driving-score improvement are not validated. The full replay estimate is approximately 112.92 GiB host RAM across two ranks, excluding PPO storage, model/optimizer state, simulator allocations, and working memory.

Next validation stage is a reduced two-GPU run, followed by the matched-step continued-PPO comparison. Do not interpret a completed CPU smoke or finite critic loss as evidence of improved driving.
