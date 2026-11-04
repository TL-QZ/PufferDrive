# PQL as an alternative to fixed n-step SAC

Status: implemented; final validation is recorded in
[`reports/implementation_validation.md`](reports/implementation_validation.md).
The follow-up request authorizes the learner,
sequence replay, tests, and two config/launcher sets based on `round4_1` and
`round2_enhance`. Official training runs are reserved for the user. The numerical
contract below is retained; the source-trial anchor is historical context.

The implemented option is a Peng-style lambda mixture of the existing SAC n-step
targets. It preserves reward-only accumulation and soft endpoint bootstraps.
It is an alternative learner mode alongside the existing fixed n-step mode,
not a replacement for that mode or a request for another standalone algorithm.

## 1. Target definition and experimental anchor

The original planning anchor was this saved trial:

`../soft_actor_critic_multi_step_return/runs/round4_nstep2_tes_sac/sac108_grid_20261102T202418Z560107829/configs/trial_98.yaml`

The implementation request supersedes that single-trial packaging choice:
derive `configs/round4_1_pql` from the current
`../soft_actor_critic_multi_step_return/configs/round4_1_nstep2_tes_sac` (trials
97 and 98), and `configs/round2_enhance_pql` from `configs/round2_enhance` in that
same source project (trial 01). Preserve each source's learner/environment
settings independently, including their different budgets and refresh ratios;
change only the return estimator controls, identities and catalog paths.

The first PQL candidate uses maximum sequence length H=128 and lambda=0.95.
128 denotes transitions in a segment, not an episode length, recurrent BPTT,
or a guarantee that every target observes 128 rewards. Episode, validity,
replay-residency, and segment boundaries can shorten the available sequence.

| Setting | Preserve from the saved trial 98 |
|---|---|
| Actor, critics, optimizers, seed | Same networks and optimizer settings; train seed 0 |
| Discount and learning rates | gamma 0.999; actor/critic LR 0.0005; alpha LR 0.0003 |
| Temperature | Learned alpha, initial 0.1; adaptive TES starting at 1.863680 |
| TES controller | EMA 0.999; mean/std thresholds 0.01/0.05; decay 0.9; required count 200; forced decay at 500M received rows |
| Replay and learning budget | 2,048,000 transition rows per rank; refresh 0.5; three replay epochs |
| Logical/microbatch | 64,000 loss-bearing transition rows / at most 32,000 rows |
| Target critic | tau 0.005; one update per logical learner update |
| Run budget and evaluation | 4B global received agent rows; carla_fast every two collection cycles |
| Environment and diagnostics | Same goal spacing, benchmark contents, environment settings, and collapse diagnostics |
| Hardware topology | Two ranks; 20 environments x 3,200 agent slots per rank; CPU-offloaded replay |

Keep critic-target entropy enabled. The previously prepared no-critic-entropy
experiment is separate. Clone the benchmark catalog into the isolated PQL config
directory and change only its path; preserve its contents. Do not reuse a
checkpoint from the source run: initial experiments start fresh.

### Exact mathematical contract

Let r_t be the reward after action a_t at observation s_t. Define the bootstrap
using the current actor, current detached alpha, and target twin critics:

```text
Vbar(s) = sum_a pi(a|s) * [min(Qbar1(s,a), Qbar2(s,a)) - alpha * log pi(a|s)]
Y_t^(n) = sum_(k=0)^(n-1) gamma^k * r_(t+k) + gamma^n * Vbar(s_(t+n))
```

For h available transitions before the next boundary:

```text
G_t^(lambda,h) = (1-lambda) * sum_(n=1)^(h-1) lambda^(n-1) * Y_t^(n)
                + lambda^(h-1) * Y_t^(h)
```

Every constituent return has the same entropy convention as our current
fixed-n implementation. There is no explicit intermediate sampled-action
`-alpha*log pi(a_t|s_t)` bonus and no importance-sampling correction.

For implementation, let d_t be 0 for a true terminal and gamma otherwise.
Let c_t be 1 only when the following transition belongs to the same valid
sequence AND is present in the sampled segment. Compute backward:

```text
if c_t == 0:
    G_t = r_t + d_t * Vbar(next_observation_t)
else:
    G_t = r_t + d_t * [(1-lambda)*Vbar(next_observation_t) + lambda*G_(t+1)]

critic_loss = mean_valid_rows(sum_j (Q_j(s_t,a_t) - stop_gradient(G_t))^2)
```

The boundary branch avoids reading a nonexistent continuation. A true terminal
does not require evaluating its placeholder next observation. All target
construction is detached. Actor loss, alpha updates, TES, and target-network
updates retain their existing definitions and logical-update cadence.

The GAE relationship is algebraic: on an uninterrupted segment, define
`delta_t = r_t + gamma*Vbar(s_(t+1)) - Vbar(s_t)` and `A_t = G_t - Vbar(s_t)`.
Then `A_t = delta_t + gamma*lambda*A_(t+1)`. PQL uses G as a Q-regression target;
the SAC actor does not consume A. This is a practical entropy-modified PQL
option, not a claim of unbiased full maximum-entropy policy evaluation.

## 2. Replay, sequence sampling, and boundary semantics

### Proposed storage: raw transitions with temporal links

Retain one copy of each raw transition rather than one materialized window per
start. Add a separate sequence replay class/mode; leave TensorReplay and the
existing NStepAccumulator path behavior unchanged.

Store observations, actions, raw rewards, true successor observations,
one-step discounts, explicit boundary metadata, and flat integer sequence links.
Use slot generation counters to detect overwritten links. Keep per-agent last
row/generation metadata to join consecutive valid outcomes; no 128-observation
staging ring per agent is needed. Collection remains keyed by stable env_ids,
never by receive-batch position. Copy simulator-owned arrays before reuse.

Prefer chronological FIFO replacement in this new mode: the existing
once-shuffled slot eviction order would fragment temporal chains. Generation
checks at sampling prevent linking an old predecessor to newly overwritten
storage. Build links only across valid uninterrupted transitions; a false
mask, terminal, or reset clears the per-agent continuation. Replay-overwrite
cuts are legitimate sampling boundaries; malformed topology is an error.
An otherwise valid transition discarded by an insertion quota also breaks
the retained chain: never link across an unrecorded reward/action. Process
all received IDs before selecting insertion rows so these gaps are visible.

This changes storage, eviction, and sample grouping relative to historical
n-step runs. Report that difference; do not label a historical-vs-PQL result
as a pure lambda ablation.

### Sample and train all valid positions, not 128 copies per starting row

At each replay epoch, build disjoint contiguous segments of at most H raw
transitions from resident chains. Shuffle segments and pack them into logical
batches. Train all valid positions of each segment exactly once per epoch.
Targets near a segment end have shorter observed suffixes. Bootstrap at cuts
from the stored successor; do not discard shorter tails to favor long episodes.
This deliberately defines a finite-segment return, like rollout-based GAE:
H is the sampled segment length limit, not 128 future steps for every loss row.
Physical storage chunk boundaries alone must not cut a sequence. Sampling
boundaries are explicit algorithmic truncations and must be counted in metrics.
A different design giving every loss row up to 128 successors would require
overlapping context/lookahead and extra target-forward work; it is not the
initial design selected here.

Preserve exactly 64,000 loss rows per full logical update and at most 32,000
per microbatch. Split segments at logical-batch packing boundaries if necessary
and mark these as artificial bootstrap cuts. Never duplicate/drop rows to fill a batch,
and never treat padding as a transition or a loss row. Prefer flat packed rows
with explicit segment boundaries. Actor updates use the same valid rows.
Microbatch boundaries do not cut targets: compute the logical batch's detached
scalar targets before slicing it into gradient microbatches (Section 4.3).

Compute endpoint values and backward targets per segment using a fixed actor,
alpha, and target-critic snapshot for that logical update. Batch network
forwards over real rows; the backward scan is bounded by H. Accumulate critic
gradients across microbatches before stepping, as today. Do not cache detached
returns permanently in replay: alpha, policy, and target critics evolve.

At full capacity this retains 32 logical updates per replay epoch and 96 per
collection cycle. Keep DDP logical-update counts aligned across ranks even if
their segment shapes differ. Weight gradients and metrics by valid loss rows,
not number of segments. Preserve the existing actor/critic update ordering.

| Boundary | Required target and linking behavior |
|---|---|
| True terminal | Include final reward, d=0, no continuation; terminal wins if both flags are set |
| Timeout/map reset | Include final valid reward; d=gamma; bootstrap from captured pre-reset observation; no reset crossing |
| Invalid/masked gap | Exclude invalid row; cut prior chain and use its last valid stored successor |
| Segment end or missing resident successor | Bootstrap from that row's next observation; no invented next action/reward |
| Collection ends / budget ends | Already completed raw rows remain eligible; pending actions without outcomes are not inserted |

Collection pauses alone do not break links if the same agent trajectory
continues. Thus behavior policies may change within a stored sequence. This
is deliberate uncorrected off-policy learning, and its age must be measured.

## 3. Memory and horizon feasibility gate

With the source topology, nominal capacity per agent slot is
`2,048,000 / 64,000 = 32` transitions. Active-agent fractions and episode lengths
change the actual distribution. This ratio is not a prediction that every
sequence has length 32, but it rules out assuming uniformly retained 128-step
histories. Packing and episode boundaries shorten target suffixes further.

Planning estimates for 923 float32 observation features, per rank:

| Allocation | Observation memory only |
|---|---:|
| 2,048,000 raw rows, each with observation and next observation | 14.084 GiB |
| Naive 128-step per-agent staging, including successor (64,000 slots) | 28.388 GiB additional |
| Naive 128-observation window for each replay row | 901.367 GiB |

Derive observation size from the actual resolved environment before allocation;
923 is an estimate from the existing experiment topology, not a hardcoded new
constant. Add scalar arrays, links/generations, pending-action storage, index
plans, minibatch transfers, network activations, pinning, and both ranks to the
preflight. CPU offload does not remove the host-memory cost.

The source feature count is 923: 10 ego + 17 reward-conditioning + 9 goal
features + 16*9 partner + 49*9 lane + 30*9 boundary + 4*7 control + 4 count
features. The lane/boundary counts include the source observation dropout.

Before claiming a 128-step experiment, measure available-chain lengths and
effective target-suffix lengths separately, including median/p90/max, fraction
of targets reaching 32/64/128, and cut causes. Also log the retained mixture
tail weight lambda^(h-1): at lambda=.95, a 32-step target places about 20% weight
on its final available bootstrap, whereas a 128-step target places about 0.15%.

If useful long sequences are scarce, record that as a design blocker for the
128-step objective rather than silently increasing replay. Uniformly retaining
128 rows for all 64,000 slots would need about 8,192,000 raw rows/rank, before
allowance for trajectory churn; observation pairs alone would be 56.335 GiB/rank.
Changing capacity, selected collection agents, or retention policy changes the
experiment and needs an explicitly revised plan. No such change is selected
here. A short synthetic/collection pilot after implementation must settle this
gate before any long run.

## 4. Future implementation stages and validation

Public estimator controls:

```yaml
train:
  return_estimator: pql   # default remains n_step for existing configs
  n_steps: 1             # raw insertion; do not compose the old accumulator
  pql_horizon: 128
  pql_lambda: 0.95
  critic_entropy_in_target: true
```

Validate mode names, finite lambda in [0,1], positive integer horizon excluding
bool, and incompatible combinations before creating replay/env resources.
Initially support standard SAC collection only; reject PPO behavior providers
and warm-start replay. Legacy configs/checkpoints remain fixed-n by default.
Persist estimator, horizon, lambda, and entropy semantics in learner metadata;
reject incompatible resume. Initial PQL trials are fresh, and learner-only
checkpointing must not imply replay/trajectory restoration.

1. **Freeze the mathematical contract.** Add an independent explicit-mixture
   oracle before wiring into training. Cover h=1, lambda=0, lambda=1, gamma=0,
   changing alpha across learner updates, terminal and timeout tails. With
   lambda=1, each available suffix must equal the existing fixed-n target for
   that same suffix; with lambda=0 or H=1 it must equal one-step SAC numerically.
2. **Implement bounded sequence replay.** New storage/sampling path in
   `pufferlib/sac_replay.py` (or a dedicated sibling module if it becomes large).
   Test stable-ID interleaving, copied observation ownership, overwritten-link
   generations, FIFO wraparound, all boundary types, and short tails. Every
   resident valid row must appear exactly once in an epoch's loss-row plan.
3. **Integrate collector and learner.** Extend `pufferlib/pufferl_sac.py` mode
   selection, raw collection links, packed sampling, and detached backward scan.
   Preserve actor/alpha/TES cadence, DDP weighting, gradient accumulation,
   quotas, budgets, diagnostics, and checkpoint validation. Keep old-mode RNG
   behavior unchanged. CPU synthetic gradient and two-rank tests precede CUDA.
4. **Prepare isolated trial configs and resource checks.** Extend SAC defaults
   and startup validation only as required; add isolated config/launcher/docs in
   this PQL project. Extend the sweep preflight for sequence storage and peak
   memory. Compare generated configs to the exact source trial; allow only
   return-estimator controls and run identity/catalog paths. Validate benchmark
   content parity, launcher construction, and scheduled/standalone evaluation.
5. **Verify and run a separately authorized pilot.** Run focused new tests and
   existing SAC regression, then CUDA/pinned-memory/compile checks and a bounded
   two-rank collection/learning pilot. Measure memory, throughput, horizon
   coverage, row accounting, and nonfinite values. Long training requires a
   later instruction and a resolved horizon-feasibility gate.

No C changes are planned. Tests should exercise boundary behavior and numerical
oracles, not merely restate the implementation. Padded diagnostic values, if
any, must never affect TES, loss means, or collective reductions.

### 4.1. GAE reuse decision, grounded in current source

Reuse the backward-return mathematics and existing SAC training machinery.
Do not call PPO's `_compute_advantages()` or modify the PPO kernel for the
first implementation. The small new return scan is not the main engineering
cost; preserving temporal replay links and training accounting is.

Current source locations:

| Component | Existing behavior | PQL decision |
|---|---|---|
| `pufferlib/pufferl.py::_compute_advantages` | Uses PPO rollout values/masks, calls advantage kernel, returns `advantages + values` | Mathematical comparison only; no PPO-buffer dependency |
| `pufferlib/pufferl.py::compute_puff_advantage` | Dispatches to compiled kernel; can transfer tensors to CPU and back when CUDA extension is unavailable | Use as a small CPU test oracle, not the production PQL path |
| `pufferlib/extensions/pufferlib.cpp::puff_advantage_row` and `extensions/cuda/pufferlib.cu::puff_advantage_row_cuda` | Dense float32 `[segments, horizon]`, backward loop from `horizon-2`, rewards/dones indexed at `t+1`, one mask gates bootstrap and trace | Leave unchanged; PQL needs transition-aligned inputs and separate boundary decisions |
| `pufferlib/pufferl_sac.py::_compute_targets` | Discrete expectation over current actor and minimum target critics, optionally including entropy | Extract the shared soft-bootstrap computation; preserve fixed-n output and call contract |
| SAC `_critic_loss`, `_policy_loss`, optimizer/TES/target updates | Selected-action twin-Q regression and discrete SAC policy optimization | Reuse directly |

For an isolated segment of h transitions, an adapter to the GAE kernel is
mathematically possible: allocate h+1 state slots, put r_t and terminal_t in
slot t+1, use all-one ratios and clips, and set values to Vbar(s_0)...Vbar(s_h).
Then `advantages[:, :h] + values[:, :h]` equals our PQL targets. At a true
terminal the final value can be zero; at a timeout/cut it must be the true
successor's soft value. Each cut must end a separate kernel row, since setting
its done flag would incorrectly remove a nonterminal bootstrap. Variable
lengths require length grouping or additional masked scratch storage.
The test adapter must supply contiguous float32 2D inputs explicitly: the
existing CPU/CUDA validation code discards the return value of `.contiguous()`
and therefore does not repair strided inputs. Record this adjacent issue without
changing the PPO extensions in this work. The legacy `tests/test_c_advantage.cu`
targets a different kernel and is not a substitute for testing the registered op.

That adapter introduces shifted arrays, an extra state slot, baseline values
that cancel algebraically, ratio buffers, and grouping/packing work. It also
couples SAC to PPO extension availability. It is useful for equivalence tests,
but is not selected for production merely to avoid a short backward scan.
Do not import PPO advantage normalization, priority filtering, value clipping,
or V-trace importance ratios into the SAC objectives.

Planning validation (2026-11-03): an ad hoc CPU check passed 72 combinations
of segment length {1,3,7}, gamma {0,0.9,0.999}, lambda {0,0.5,0.95,1} and
terminal/nonterminal tails. For every suffix, the explicit mixture, direct
backward recursion and existing compiled CPU GAE kernel through the shifted
adapter agreed within float32 tolerance. This verifies the algebra/adapter,
not an implemented PQL learner, replay integration or CUDA performance.

### 4.2. File ownership and implemented interfaces

The interfaces below are now implemented. Keep new estimator
and sequence-storage concepts in dedicated modules; avoid duplicating the SAC
optimizer loop or creating a general-purpose return-estimator framework.

| File | Planned change |
|---|---|
| New `pufferlib/sac_returns.py` | Pure detached `compute_pql_targets(rewards, discounts, next_values, segment_offsets, trace_lambda)` returning float32 `[B]` |
| New `pufferlib/sac_sequence_replay.py` | `SequenceReplay` composing `TensorReplay` for the five existing payload fields; owns FIFO insertion, generations, temporal links, per-agent tail state, and epoch segment planning |
| `pufferlib/pufferl_sac.py` | Config validation and replay construction, collector metadata routing, shared bootstrap-value helper, PQL batch preparation, checkpoint metadata and diagnostics |
| `tests/unit_tests/test_sac_pql_returns.py`, `test_sac_sequence_replay.py`, `test_sac_pql_collection.py`, `test_sac_pql_learner.py` | New numerical, topology, collector and learner acceptance tests; extend existing distributed/startup/resume suites where relevant |
| This project's configs/scripts plus existing sweep preflight | Isolated source-trial-derived launch setup; estimator-aware memory estimates and config parity checks |

Reuse `TensorReplay` allocation, copied storage and `get_batch()` transfers by
composition. The sequence wrapper controls its writes so payload replacement
cannot bypass generation/link maintenance. Existing fixed-n users continue to
use `TensorReplay` and `NStepAccumulator` unchanged. Do not add a second copy of
observation storage in the wrapper.

Sequence metadata uses flat integer arrays: predecessor/successor row indices
and expected generations, generation per occupied row, stable agent identity,
episode/boundary metadata, insertion age, and per-agent last row/generation.
On overwrite, invalidate reciprocal links only when their generations match;
sampling also verifies generations. Enumerate resident chain heads in stable
order, visit each row exactly once, split at H, then shuffle segments using the
seeded training RNG. Bound traversal by resident row count and segment scans
by H; duplicate visits, cycles, or cross-agent links are errors. Expected stale
links caused by eviction are cuts, not malformed new transitions.

The collector reuses `observe_transition()`, `_consume_final_observation_infos()`
and `record_actions()`. Carry compacted stable IDs and terminal/truncation flags
alongside raw payloads. Process invalid IDs and quota-discarded IDs before
linking the next retained transition. Terminal and truncation rows may link
from their predecessors, but cannot link forward across reset. A collection
pause alone does not clear a valid per-agent tail.

Validate PQL settings before both environment and replay construction, including
the public `train()` startup path and direct `PuffeSAC` construction. PQL rejects
`n_steps != 1`, nonempty injected replay, PPO behavior providers/factories and
PPO warm-start workflows. Persist mode, H, lambda and an explicit entropy
convention identifier such as `soft_each_return_endpoint` in checkpoint config.
Legacy missing mode resolves to fixed-n; mismatched estimator settings reject
resume. Reuse current learner-only checkpoint machinery: replay, links and
pending actions restart empty, and no exact trajectory resume is promised.

### 4.3. One logical learner update, with microbatch-invariant targets

The epoch plan is flat replay indices `[B]` in segment order plus integer
`segment_offsets[S+1]`; adjacent offsets delimit S nonempty segments. There
are no padding transitions. B counts loss rows, not sequences or context rows.
Only logical-batch packing can introduce an additional cut; changing
`max_minibatch_size` must not change G for a fixed logical batch plan.
Check this within floating-point tolerance. Batch-size-dependent or stochastic
model layers would need an explicit target-forward policy to preserve this
contract; verify the selected actor/critic architecture before integration.

1. **Prepare scalar bootstrap inputs.** Fetch raw rewards/discounts and forward
   real nonterminal successor observations in chunks no larger than the
   microbatch limit. Actor logits and twin-Q outputs are `[chunk_rows, A]`;
   reduce over A to `next_values[chunk_rows]`. Terminal rows require no network
   evaluation of placeholder successors; set their unused scalar value to
   zero. Hold the actor, alpha and target critic fixed throughout this logical
   target pass by performing no optimizer or target updates. All arithmetic
   feeding the scan is detached float32.
2. **Scan all segments backward.** Start each segment tail with `r+d*Vnext`;
   for internal rows use `r+d*((1-lambda)*Vnext+lambda*Gnext)`. Build flat index
   groups by distance from segment end on CPU when constructing the batch
   plan. At depth zero compute all tails together; at each subsequent depth
   gather the already computed successors and scatter current targets. This
   uses at most H batched tensor iterations and O(B) scalar/index storage,
   without a Python loop per transition or padded observations. Never gather
   an absent successor and mask it afterward.
3. **Reuse critic gradient accumulation.** Fetch current observations/actions
   in existing microbatches, slice the precomputed G vector, call `_critic_loss`
   and weight each backward by `micro_count / logical_batch_count`. Preserve
   DDP `no_sync`, clipping after accumulation and one critic optimizer step.
4. **Reuse the actor and temperature updates.** Train the actor on exactly the
   same valid loss rows after the critic step. Keep entropy aggregation, alpha
   optimization and TES cadence unchanged. PQL does not supply an advantage
   tensor to the actor.
5. **Reuse final update bookkeeping.** Polyak-update target parameters/buffers
   once, increment existing row/update counters, and log sequence telemetry.
   Discard scalar targets after the logical update; recompute for the next
   update with the new actor/alpha/target critics.

Keep a single shared optimizer/update loop. The mode branch selects the epoch
index plan and target preparation; fixed-n retains its current random-row plan
and per-microbatch `_compute_targets()` calls. Do not create extra random draws
or target forwards in the default mode. The PQL prepass stores only scalar
targets/inputs and indices for a logical batch, not B observation activations.

The bounded PyTorch scan is the initial correctness implementation. Profile
network forward, CPU planning, transfers and scan separately on realistic
segment distributions. Verify eager/compiled network modes; do not assume the
ragged scan supports full-graph capture or hide CPU transfers behind the PPO
fallback. A specialized compiled scan is a later optimization only if profiling
shows a material bottleneck. No CPU/CUDA extension edits are required initially.

### 4.4. Delivery order and acceptance gates

Each stage is reviewable before starting its dependent stage. The follow-up
implementation request authorizes stages 1-4. Runtime evidence and unavailable
hardware checks must be reported separately; no official runs are launched.

1. **Numerical core and shared bootstrap extraction.** Compare the new scan to
   an independent explicit n-step-mixture oracle for every suffix, and to the
   existing CPU GAE kernel through the h+1 adapter on small contiguous examples.
   Cover unequal segment lengths, h=1, lambda=0/1, gamma=0, terminal/timeout
   tails, negative rewards, changing alpha between updates, and no target
   gradients. Construct twin critics whose ordering swaps across states to
   verify minimum-before-mixture semantics. Preserve fixed-n target tests.
2. **Sequence storage and collection.** Test interleaved stable IDs, shared
   observation-buffer reuse, FIFO wraparound, generation mismatches, invalid
   gaps, quota discards, true terminal versus simultaneous truncation,
   pre-reset final observations, and continuation across collection calls.
   Every resident valid row must appear exactly once in the epoch loss plan.
   Reuse scenarios from `test_sac_nstep_collection.py` and
   `test_sac_final_observations.py`, with raw-row/link expectations.
3. **Learner integration and distributed accounting.** Extend the tiny-network
   fixtures in `test_sac_learner.py` and CPU/Gloo harness in
   `test_sac_distributed.py`. Compare full versus microbatched updates for the
   same segments, including a sequence crossing a microbatch boundary. Compare
   lambda=0 to one-step SAC using identical ordered loss rows (different replay
   samplers need not produce identical full training runs). Verify different
   segment shapes across ranks still give equal logical update counts, correct
   gradient/entropy weighting, once-only row coverage and unchanged TES/target
   cadence. Extend startup and checkpoint compatibility checks.
4. **Experiment packaging and regression.** Prepare the two requested isolated
   config/launcher sets from `round4_1_nstep2_tes_sac` and `round2_enhance` with
   their original trial selectors (97/98 and 01). Reuse the launcher's CLI conventions and
   `project/soft_actor_critic_tuning/scripts/run_sac16_grid.py::_memory_preflight`
   with explicit sequence metadata/scalar scratch estimates. Run the new tests
   and existing SAC learner, replay, n-step, final-observation, entropy,
   diagnostics, startup, loading and distributed suites. Verify config diff,
   benchmark byte parity, dry-run output and unchanged legacy-mode RNG behavior.
5. **Runtime feasibility gate.** After separately authorizing a bounded pilot,
   verify CUDA, CPU-offloaded/pinned replay and configured compilation mode.
   Record actual RAM/VRAM, throughput, chain versus suffix length distributions,
   cuts by cause, replay age and nonfinite values. Keep the existing 32-row/slot
   nominal-capacity limitation explicit. No long training until this gate is
   resolved and the user requests it.

## 5. Experiment interpretation, artifacts, and references

The prepared experiments use PQL H=128/lambda=.95 in both requested source
families. Preserve trial IDs and source meaning; do not add a controlled grid
axis that renumbers them. Trial selection and official launch belong to the user.

For causal interpretation, a future control using the SAME sequence replay,
eviction, and loss-row sampler but the existing two-step target would separate
return estimation from replay changes. This is a validation/comparison control,
not a second requested algorithm. Where it has two available steps, compare
against the existing accumulator's two-step target; report shortened tails.
Do not launch this control or a lambda sweep as part of plan preparation.

Evaluate peak and late retention together at comparable global received rows.
Use puffer_score, progress, speed, goals, infractions, and DNF, plus entropy,
alpha, TES state, critic/TD error and gradient diagnostics. Add sequence-length,
target-horizon, cut-cause, replay-age, and loss-row/sample-count telemetry.
Compare actual update counts and collection freshness; nominal resource parity
does not make historical aggregated replay identical to raw sequence replay.

Artifacts stay under this project's own `configs/`, `scripts/`, `runs/`, and
`reports/` directories. The implementation also adds shared learner modules and
focused tests under `pufferlib/` and `tests/unit_tests/`. Dry-run manifests are
configuration evidence, not evidence of completed training or GPU feasibility.

Primary reading:

- [Kozuno et al., Revisiting Peng's Q(lambda), ICML 2021](https://proceedings.mlr.press/v139/kozuno21a.html): uncorrected off-policy lambda returns and conditional convergence results.
- [Supplement, Appendices I and J](https://proceedings.mlr.press/v139/kozuno21a/kozuno21a-supp.pdf#page=22): entropy modifications, recursion, and sequence-based actor-critic implementation. Our exact operator is specified above rather than inferred from the paper's prose.
- [Authors' implementation](https://github.com/robintyh1/icml2021-pengqlambda): reference for partial-trajectory replay, not a drop-in implementation for our discrete/multi-agent collector.
- [Schulman et al., GAE](https://arxiv.org/pdf/1506.02438): lambda-weighted TD-error sums and their return interpretation.
- [Existing local n-step design](../soft_actor_critic_multi_step_return/design.md): current return convention, stable-ID collection, and boundary contract.
