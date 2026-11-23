# Implementation validation and first launch

Validated on 2026-11-14 (America/New_York); launch timestamps use UTC.

## Passed checks

- Full project suite: **21 passed in 3.00 seconds**. Covers frozen model loading, return indexing, implemented PQL targets, selection, resume integrity, native transition parity, final observations, and reset after a held boundary.
- Mandatory production rebuild completed successfully. Private extension built separately. No C source warnings; the toolchain emitted a serial LTO compilation notice.
- Native parity tests compare production/private transitions bitwise through a short timeout. Full-length checkpoint gates additionally validate exact reference-state reconstruction before branching.
- Independent source review finished without remaining material findings after corrections to action buffers, metadata dimensions, native bounds, provenance, scene counts, and held-scene reset.

## Final GPU execution gates

Each gate used one reference episode, one probing agent/frame, all 12 actions and 2 paired continuation seeds. Both completed successfully with 24 finite reconstructed training targets and 12 action-summary rows.

| Model | GPU | Reference frames | Gate elapsed seconds* | Output |
|---|---:|---:|---:|---|
| TES | 6 | 2560 | 75 | `runs/tes_gate_final_20261114` |
| PQL | 7 | 147 | 21 | `runs/pql_gate_final_20261114` |

*From manifest creation to completion; excludes checkpoint initialization before manifest creation. These are gate measurements, not estimates of large-scale throughput. TES GPU memory was observed at approximately 523 MiB during the initial gate; this is not a peak-memory measurement or a guarantee for denser scenes.

The original gate exposed 24 horizon-censored TES branches, and 11 terminal plus 13 scene-truncated PQL branches. This demonstrates why termination and missing-tail information must accompany return comparisons. These checks do not establish critic accuracy or policy quality.

## First experiment launched

Output: `runs/first_20261114`. Frozen completed 4-billion-step checkpoints; two reference scene initializations per model, up to 12 probing situations, 12 actions and 8 continuations: at most 1152 branches per model. Scene copies are batched in groups of 24.

Sessions: `q_probe_20261114_tes` on GPU 6 and `q_probe_20261114_pql` on GPU 7. Both sessions were verified live and their progress files advanced after launch. See `launch.json`, each model's `manifest.json`, `progress.json`, and `console.log`. Successful collection automatically generates action summaries.

Training remains unchanged. The experiment disables partner blindness and phantom braking to enforce all-policy control; this deviation is recorded in each manifest. See [EXPERIMENT.md](EXPERIMENT.md) for interpretation and commands.

## Directory reorganization — 2026-11-21

Shared implementation moved to `development/`; each model now contains `scripts/`, `records/`, and `plots/`. Historical manifests and launch commands retain their original paths as provenance.

- 75 original collected-data and execution-history files remained byte-for-byte identical.
- All nine PQL figures regenerated through the model-local analysis entry point.
- Plot directories contain images and READMEs only; supporting CSV/JSON/notes are in records/analysis/. Local Markdown links resolve.
- Relocated CPU/private-binding suite: 21 tests passed in 9.11 seconds.
- Launcher dry-run produced two model commands and created no output directory. No new experiment or training was launched.

## Per-rollout evaluation metrics — 2026-11-21

New collections save all 40 evaluation `metrics_mean` fields per probing-agent continuation; see [metric definitions and storage](METRICS.md).

- Full development suite: **28 passed in 5.70 seconds**. Native metric tests also passed after adding both zero-progress and positive-progress regression cases (2 passed in 4.50 seconds).
- Exact metric-key parity checked against the saved final TES and PQL `carla_fast` evaluation summaries.
- Native tests verify nonzero-prefix exclusion, interval reward/goal/distance agreement, finite values, interval safety flags, score gating, schema completeness, malformed input rejection, and readout fingerprint stability.
- Two 32-step CPU scenes verify bitwise observation, reward, termination, pose, and speed parity with evaluation telemetry disabled versus enabled. A fake-batch collector test verifies immediate capture when one branch ends before another. JSON/NPZ alignment and mixed legacy/new CSV summaries are covered.
- Production and private extensions rebuilt successfully. No C source warnings; the toolchain emitted its serial LTO scheduling notice. Production simulator source is unchanged.
- All 120 files in the existing first experiment remain byte-for-byte identical. No new collection, GPU experiment, or training was launched.
