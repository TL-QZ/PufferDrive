# Source audit (before measurements)

These are code-derived workload facts, not measured speedups.

## Matched architecture and environment

The saved November 6 PPO and October 29 SAC trial-98 configs have identical `policy` dictionaries. Their environment dictionaries differ only in `capture_final_observations` (SAC true for correct timeout bootstrapping). PPO uses two independently owned Drive backbones; SAC owns actor, Q1, Q2 and two target-Q backbones. Both have float32 parameters and use configured BF16 autocast for network operations.

## Sample exposure

PPO `batch_size=auto` resolves to total vector agent slots times horizon. In the baseline this is 20*3200*128 = 8,192,000 rows per rank. Masks and advantage filtering reduce the trained subset; three passes apply to the retained subset. Count actual `_ppo_loss` rows rather than assuming three times received rows.

SAC's 2,048,000-row replay is filled initially, then half refreshed with valid inserts. Each training cycle processes all replay rows in each of three shuffled passes. The steady-state exposure is six sampled rows per inserted valid row. Received rows include masked/reset/quota-discarded rows, so this is not six samples per received row. Count both `global_step` and `valid_inserts` deltas.

## Network work per sampled row

- PPO: actor and value forwards, then gradients through both. Rollout inference also computes both actor and value.
- One-step SAC critic update: next-observation actor and two target-critic forwards without gradients, followed by two online-critic forwards/backwards.
- SAC actor update: actor forward/backward and two online-critic forwards without gradients. These critic evaluations use parameters after the critic optimizer step; pre-step Q values cannot be reused without changing the algorithm.

Thus SAC performs eight backbone forwards and three backward paths versus PPO's two forwards and two backward paths per training exposure. Forward/backward counts are not FLOP or elapsed-time ratios; pooling, memory traffic, compilation and DDP affect their costs.

## Replay overhead candidates

`pufferlib/pufferl_sac.py::_train_sac_transition` fetches `replay.get_batch(micro_indices)[0]` during actor training. `TensorReplay.get_batch` allocates and gathers all five fields, then the caller discards actions, rewards, next observations and discounts. Observation-only gathering preserves semantics. Retaining current observations through the critic optimizer step can additionally avoid the second host gather/H2D, but must bound memory and preserve the newly updated critic forwards.

Pinned batches are newly allocated per call. Reusing staging storage requires stream-lifetime discipline: a CPU write must not race an outstanding asynchronous H2D. Explicit event synchronization or double buffering is required unless the measured implementation establishes completion first.

Both learners use `.item()` inside their training loops, so GPU scalar synchronization is not unique to SAC. SAC has additional independent optimizer and target-network work. Measure rather than attribute the entire gap to scalar extraction.

## Measurement traps

- `Profile` samples one cycle in five and multiplies elapsed time by five. It calls CUDA synchronization at profile boundaries; sampled timings perturb overlap and are estimates.
- PPO logs `epoch` using a distributed sum, so two-rank logged epochs are twice the real cycle count. SAC logs its cycle directly.
- The new PPO uses A6000 hardware; the historical SAC reference used A5000 hardware. Raw historical speed ratios confound hardware.
- New PPO evaluation every two cycles means about 32.768M received rows between benchmarks. SAC evaluation every two half-refresh cycles means 4.096M valid inserts between benchmarks, plus masked/reset rows. Identical epoch intervals do not imply identical evaluation overhead per row.
- Cold/random-policy probes do not represent late-training filtering or vehicle activity. Report exact filtering and received/inserted/sample counts; use learner-only paired data when testing implementation parity.
