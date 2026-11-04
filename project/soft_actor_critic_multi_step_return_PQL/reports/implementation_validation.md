# Discrete SAC + Peng Q(lambda): implementation validation

Date: 2026-11-03. No official training run was launched.

## Implementation and reuse

`pufferlib/sac_returns.py` implements the detached float32 backward return scan.
`pufferlib/sac_sequence_replay.py` composes the existing TensorReplay payload
storage with chronological FIFO replacement and generation-checked temporal
links. Every resident row is loss-bearing once per replay epoch. Segments stop
at episode boundaries, collection gaps, overwritten history, horizon limits,
and logical-batch boundaries.

`pufferlib/pufferl_sac.py` shares the discrete soft value computation with fixed-n
SAC. It freezes a logical batch's scalar targets before optimizer microbatches
and reuses the existing critic, actor, temperature, adaptive TES, distributed
gradient accumulation, and target-network update loop. A timeout ends the trace
but retains its true final-observation bootstrap. True terminals have zero
discount and do not require a successor network evaluation.

PPO's GAE kernel is reused only as a numerical oracle through an h+1 shifted
adapter. SAC uses neither PPO advantage normalization nor V-trace ratios.
The target is the lambda mixture of reward-only n-step returns with a soft
bootstrap at each return's endpoint; it is not a full intermediate-entropy
policy-evaluation return.

## Verification

- Independent review found no remaining material issues. Its focused replay,
  return, collector, learner, and startup run passed 43 tests, with one CUDA
  test skipped in that CPU-only execution.
- Final GPU check on physical GPU 4 passed 9 tests in 13.17 seconds: return
  algebra and CUDA parity plus eager FP32 and compiled BF16 synthetic learner
  updates. These use tiny networks and synthetic replay, not a Drive environment.
- An independent FIFO oracle passed 200 randomized insertion/overwrite cycles,
  including interleaved agents, gaps, terminals, truncations, and epoch coverage.
- Configuration tests compare merged and generated trials against each source,
  verify byte-identical benchmark catalogs, and dry-run all supported selectors.
  Both launcher scripts pass `bash -n`; whitespace checks pass.

Final combined non-Gloo regression: **285 passed, 6 skipped, 5 deselected,
34 subtests passed in 20.75 seconds**. The five deselected Gloo tests were
verified separately: four existing SAC distributed tests passed, and the new
PQL two-rank test passed in 10.21 seconds. The PQL test uses unequal segment
layouts across ranks and checks parameter agreement, global sampled-row
accounting, averaged alpha gradients, adaptive TES, and Polyak updates.

Reproduce the combined non-Gloo run from the repository root:

```bash
source .venv/bin/activate
CUDA_VISIBLE_DEVICES='' python -m pytest tests/unit_tests/test_sac*.py tests/unit_tests/test_pufferl_sac_startup.py tests/unit_tests/test_ppo_init_behavior_policy.py project/soft_actor_critic_tuning/tests/test_run_sac16_grid.py project/soft_actor_critic_multi_step_return_PQL/tests -k 'not gloo' -q --disable-warnings --maxfail=1
```

Run `test_sac_distributed.py` and `test_sac_pql_distributed.py` separately where
local Gloo sockets are permitted. Run `test_sac_pql_cuda.py` and
`test_sac_pql_returns.py` with an explicitly selected available CUDA device.

## Synthetic replay planning measurement

A CPU, one-thread index benchmark used 2,048,000 rows, 64,000 agent slots,
32 transitions per agent, and **one observation feature**. Insertion took
0.569 seconds; epoch planning took 2.713 seconds. Peak process RSS was
966.39 MiB, including the PyTorch process baseline. The plan contained 32
logical batches of 64,000 rows, with maximum chain length 32 and mean per-row
target suffix length 16.5. No suffix reached 64 or 128.

This measures index planning at the configured row count. It does not measure
full 923-feature observation storage, pinned-memory pressure, environment
throughput, real CUDA transfers, or policy improvement. The launcher estimates
payload, sequence metadata, epoch plans, and scratch storage with 10% headroom;
GPU network activations are excluded. Full-scale runtime feasibility remains
unmeasured until a driving run is performed.

## Prepared launch sets

- `configs/round4_1_pql`: source `round4_1_nstep2_tes_sac`, trials 97 and 98.
- `configs/round2_enhance_pql`: source `round2_enhance`, trial 01 (alias 1).

Both use raw `n_steps=1`, `return_estimator=pql`, `pql_horizon=128`,
`pql_lambda=0.95`, and `soft_each_return_endpoint`. Source budgets, learning
rates, refresh ratios, environments, and evaluation settings remain inherited.
Launch commands and safe previews are in [the README](../README.md).
