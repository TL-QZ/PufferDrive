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
