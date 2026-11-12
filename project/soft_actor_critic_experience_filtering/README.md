# SAC experience filtering: TD-error threshold design

Status: corrected the filter scope to train the whole learner on one retained
subset. The earlier `experience_filter_20261112T012430Z` runs use critic-only
filtering; they are not results for the corrected algorithm. New configs and
output identities use `actor_critic`. Live processes and their resolved configs
are not changed by this source update.

## Launch and monitoring

| Variant | Return target | Logical batch per rank | Microbatch cap | Default GPUs |
| --- | --- | ---: | ---: | --- |
| TES | One-step soft SAC | 128,000 | 64,000 | 0,1 |
| PQL | Horizon 128, lambda 0.95 | 64,000 | 64,000 | 2,3 |

Both use 2,048,000 replay rows per rank on GPU, one update epoch, one-third FIFO
replacement, and 4B global received rows. Filtering uses beta 0.25 and scale 0.01.

Prepare resolved configs and commands without starting training:

```bash
bash project/soft_actor_critic_experience_filtering/launch_tes.sh --dry-run
bash project/soft_actor_critic_experience_filtering/launch_pql.sh --dry-run
```

When ready to train, omit `--dry-run` and use a fresh output directory. TES
defaults to GPUs `0,1`; PQL defaults to `2,3`. Override with `--gpu-groups 4,5`.
W&B is online by default; `--no-wandb` disables it. The scripts activate `.venv`.
Each run writes its resolved config, command, console log, and `metrics.jsonl`
through the production trial launcher. Generated runs/preflight outputs stay local.

Plot `filter/retained_fraction` and `filter/filtered_fraction` in W&B. They are
fractions from zero to one, computed from global row counts for the latest
training cycle, and sum to one when candidates were scored. Cumulative rates
are logged separately. In `actor_critic` mode, retained rows feed both critic and
actor optimization, and their entropy feeds alpha/TES. Repeated visits to a
replay row count as repeated exposures.

Print the latest rates from a local trial (substitute the actual run directory):

```bash
source .venv/bin/activate
python project/soft_actor_critic_experience_filtering/monitor.py RUN_DIRECTORY/trial_98/metrics.jsonl
```

The monitor accepts multiple metric paths, so TES and PQL can be displayed
together. It skips an unfinished final JSONL line while a live run writes it.

The filter settings are in each variant's `SAC_tunable_config.yaml`, under
`train.experience_filter`. Set `enabled: false` for the original learner control;
set `threshold_scale: 0` with filtering enabled for the all-rows overhead control.
Use separate output directories and run identities for each control. These are
fresh experiments; enabled and disabled learner checkpoints are not interchangeable.
The explicit legacy mode `apply_to: critic` retains the earlier behavior for
reproducibility. Its checkpoints cannot resume into `actor_critic` mode.

## 1. Experiment and scope

Select one subset of each candidate batch and train the critic, actor, and
entropy temperature on that subset. Keep replay insertion, replacement,
candidate ordering, return definitions, and evaluation protocol unchanged.
Targets and TD-error scores must still be computed for all candidates to decide
which rows to retain. Raw max-Q and actor-advantage scoring remain separate
future experiments.

This is inspired by [Prioritized Experience Replay](https://arxiv.org/html/1511.05952),
but uses hard exclusion for the current update instead of positive sampling
probabilities for every transition. It deliberately changes the critic training
distribution for both critic and actor; importance weighting cannot recover the excluded contributions.
High residual is a proxy for learning opportunity, not a guarantee: noise and
bootstrap errors can also produce large residuals.

## 2. Score and moving threshold

For candidate transition i and each online critic j, compute before optimization:

```text
delta[i, j] = target[i] - Q_j(observation[i], recorded_action[i])
score[i] = max(abs(delta[i, 1]), abs(delta[i, 2]))
current_max = MAX(score across all candidates and all learner ranks)
ema_max = current_max                                  # first scoring batch
ema_max = beta * current_max + (1 - beta) * ema_max     # subsequent batches
threshold = threshold_scale * ema_max
keep[i] = score[i] >= threshold
```

Use absolute errors from both critics, not just their minimum: either critic
being wrong is sufficient to retain a row. Targets and scores are detached.
Calculate residuals and reductions in float32, matching the existing loss path.

`beta` weights the NEW maximum. This matches the repository PPO convention in
`pufferlib/pufferl.py::_train_ppo_transition`, and corrects the opposite convention
used in the initial conversation equation. This is an EMA of batch maxima, not
an EMA of the mean TD error. The threshold uses the newly updated EMA.

The scoring unit is one original logical critic batch, before microbatching and
before any optimizer step. Advance the EMA exactly once per unit. All candidates
contribute to its maximum, including excluded rows. Microbatch size must not
change scores, threshold cadence, or selected rows beyond numerical tolerance.
This cadence differs from PPO's rollout-level scoring; copy its formula, not
its entire scheduling mechanism.

Prepared settings (experimental starting points, not tuned defaults):

```yaml
train:
  experience_filter:
    enabled: true
    score: twin_max_abs_td_error
    apply_to: actor_critic
    threshold_mode: ewma_max
    ewma_beta: 0.25
    threshold_scale: 0.01
```

Reject unsupported modes and nonfinite values at startup. Require
`0 < ewma_beta <= 1` and `threshold_scale >= 0`; reject booleans for numeric
settings. Scale zero is the all-rows control. Equality passes: all-zero scores
with a zero threshold retain all rows. With a positive historical threshold,
an all-zero batch can retain none. Never silently relax the threshold.

## 3. Target and update contract

1. Build the original candidate logical batch using the existing replay order.
   For PQL, preserve the original sequence segments, horizon, and boundaries.
2. Compute and temporarily cache the SAME scalar targets used for training.
   One-step/fixed-n SAC uses the configured reward, discount, and soft bootstrap;
   PQL computes lambda returns on the intact candidate sequences before masking.
   Do not replace a PQL residual with a one-step TD error.
3. Score all candidates with the current online critics under no-grad, update
   the global EMA, then compact retained indices and their aligned targets in
   original order. No resampling, ranking, padding of statistical samples, or
   changes to replay storage.
4. Fit both critics on retained rows using the cached targets. Then train the
   actor on those SAME retained observations, using the updated critic values.
   Each original nonempty logical batch still produces one optimizer step per
   enabled subsystem; this change does not repack rows across logical batches.
5. Compute globally retained-row-weighted entropy for alpha and TES. Perform the
   Polyak update after the successful learner update. If no rank retains a row,
   skip critic, actor, alpha, TES observations, and Polyak updates on every rank.
   Continue counting original candidate visits separately from actual updates.

Target caching lasts only for the current logical batch; it is not a frozen
cycle-target or frozen actor-Q experiment. Replay is unchanged, so every resident
row is eligible for rescoring on its next scheduled epoch traversal. A row may
still expire through normal FIFO replacement without a critic update; log this
as a possible consequence rather than promising every row is trained.

The explicit legacy `critic` mode keeps its original full-candidate actor and
alpha/TES updates and its target update even when the critic selection is empty.
This compatibility behavior does not apply to the corrected `actor_critic` mode.

## 4. Distributed reduction and empty ranks

Use one shared EMA: all-reduce the candidate maximum with MAX, and retained
counts with SUM. Let world size be W and total retained count be K. The intended
critic objective is the global retained-row mean:

```text
L = SUM_retained[(Q1 - target)^2 + (Q2 - target)^2] / K
```

DDP averages rank gradients. Therefore each real local microbatch contributes
`W / K * sum(per_row_loss)` before backward. Do not average local retained means:
that would overweight ranks retaining fewer examples. Report losses and Q/target
means with the same global row weighting. Apply this same normalization to actor
gradients. Compute alpha/TES entropy from the sum of retained entropies divided
by K, not the mean of rank means or the original candidate count.

Unlike the existing PPO/profiling filters, do not trim every rank to the minimum
retained count. One empty rank must not cancel useful critic training elsewhere.
The initial implementation should use synchronized microbatch rounds:

```text
round_count = MAX_rank(ceil(local_retained_count / microbatch_capacity))
```

All ranks execute the same round count, using DDP no-sync except on the final
round. A rank with no real rows left executes a graph-connected zero-loss forward
on one valid original candidate observation through the relevant critic or actor. This is
communication participation only: it has zero gradient contribution, is excluded
from sample/loss counters, and never becomes a retained training example. Use a
zero sum of finite critic outputs, never an empty-tensor mean. Every rank executes
the optimizer step when K > 0, even if its local retained count is zero.

This strategy requires verifying that both actor and critic have no stateful forward updates
(e.g. batch-normalization statistics) or stochastic effects that would make the
extra participation forward change training state. If this condition fails,
resolve the reduction architecture before implementation; do not silently accept
state changes. Avoid introducing explicit manual gradient reduction in v1.

Assume the existing synchronized candidate schedule: each rank has a nonempty
original batch and the same logical-update count. Check this contract in preflight.
Collectively detect nonfinite scores before MAX/backward so all ranks abort with
the same diagnostic rather than leaving peers in a collective. K = 0 is a shared
branch; all ranks skip the whole learner update in `actor_critic` mode.

## 5. Isolation, persistence, and measurements

The filter implementation lives in this folder, with copied source configs and
catalogs and focused tests. Production SAC opts into it through
`train.experience_filter`; the existing production runner handles both variants.
Reuse production replay, targets, actor loss, and checkpoint machinery. Do not
import the profiling fixed-target filter as the learner: it changes target
lifetime and filtering scope. Disabled filtering preserves the baseline path.

Prepare BOTH TES and PQL from `project/sac_gpu_4b_comparison/configs/{tes,pql}`
at source revision `573f6d2bc66abe9a2599bcaa6bbb7d4199f9153e`. Each remains a
separate comparison against its own unfiltered baseline. Preserve its 4B global
received-row budget, trial 98, evaluation cadence, optimizer settings, replay
settings, and local catalog contents. Only filtering and experiment identity
change. No source experiment configs are edited.

Checkpoint the EMA initialization flag/value, scoring-batch count, config/schema
identity, and cumulative counters through the learner checkpoint lifecycle.
Restore the shared state identically on every rank and reject incompatible
filter settings. Enabled-filter resume requires that state; a weights-only
start is explicitly a fresh experiment. Disabled filtering retains the original
code path and old-checkpoint compatibility.

| Implemented measurement | Interpretation |
| --- | --- |
| `filter/{candidate,retained,filtered}_count` and `{retained,filtered}_fraction` | Global counts and row-weighted fractions for the latest cycle; `cumulative_` counterparts track the run |
| `filter/{current_max,ema_max,threshold}` | Shared values from the last logical scoring batch |
| `filter/actor_rows` and `filter/update_steps` | Actual global actor exposure and completed critic optimizer updates (shared learner updates in the prepared recipes); cumulative counterparts distinguish candidate visits from training |
| `performance/{filter_target,filter_score,filter_select}` and existing performance logs | Target preparation, scoring, selection, and baseline learner/collection timing; include all costs in comparisons |
| Existing evaluation, entropy, and retained-row critic loss/Q metrics | Unchanged evaluation protocol; critic summaries use the global retained-row denominator |

Per-rank retention distributions, TD-error quantiles, and retained/rejected error
means are follow-up diagnostics, not emitted by this implementation. Legacy
critic-only resumes must not fabricate historical actor/update counters that
were never recorded; their metric-history scope must be explicit.
For the prepared actor-enabled recipes, `filter/actor_rows` must equal
`filter/retained_count` in `actor_critic` mode. This is a direct monitoring check
that actor training follows the same selected subset.

No speedup is assumed: all candidates still need targets and scoring forwards,
retained rows need another critic forward, while actor work now follows the
same retained subset. Full-candidate target/scoring costs remain.
DDP imbalance and variable compiled shapes may add overhead. Record compile
warmup separately and include steady-state compaction costs. Do not invent
retention floors or duplicate selected samples to stabilize batch size.

The learning-rate scheduler retains the baseline cycle cadence. A custom
threshold scale above one can skip every initial critic step and trigger
PyTorch's scheduler-before-first-optimizer warning; no empty optimizer step is
inserted to suppress it. The prepared scale of 0.01 retains at least the maximum
on the first scored batch, including an all-zero first batch.

## 6. Validation and experiment gates

1. **Math/unit tests:** exact score/EMA/mask fixtures, first-batch initialization,
   ties, beta=1, scale=0, all-zero/empty selection, nonfinite and invalid config
   rejection, target-index alignment, and checkpoint round trips.
2. **Single-process parity:** disabled path matches baseline; scale-zero enabled
   path matches losses/updates within declared numerical tolerance. Vary physical
   microbatch size. Verify identical PQL targets before/after row selection and
   selected actor inputs, retained-weighted alpha/TES entropy, replay order,
   and all-subsystem skipping on globally empty selections.
3. **Two-rank CPU/Gloo:** compare against a concatenated single-process retained
   batch for uneven counts, one empty rank, unequal microbatch counts, and globally
   empty selection. Test shared EMA, nonfinite collective failure, synchronized
   optimizer state, and resume equivalence. This does not establish CUDA safety.
4. **Bounded GPU probe after authorization:** verify NCCL completion, memory,
   compiled-shape behavior, state-free participation forwards, finite updates,
   and measured scoring-versus-backward cost. No official training at this gate.
5. **Controlled policy experiment after authorization:** same source settings,
   seeds, replay exposure, and evaluation protocol for disabled control,
   scale-zero overhead control, and EMA filtering. Fix beta at 0.25 initially;
   begin with scale 0.01 and select any further scales from observed retention.
   A matched-retention random-drop control can then separate score usefulness
   from the effect of fewer learner examples. Report multiple seeds before making
   a policy-quality claim.

## Source anchors checked for this design

| Source | Relevant behavior |
| --- | --- |
| `pufferlib/pufferl.py::_train_ppo_transition` | EMA of absolute-advantage maxima; beta weights new maximum; local thresholds and minimum-count trimming |
| `pufferlib/pufferl_sac.py::_critic_loss` | Sum of two MSE losses on recorded actions |
| `pufferlib/pufferl_sac.py::_train_sac_transition` | Candidate epochs, microbatch weighting, actor/alpha/TES and Polyak update order |
| `pufferlib/pufferl_sac.py::_prepare_pql_targets` | Intact segment-based lambda targets before gradient microbatches |
| `project/PPO_SAC_Profiling/sac_variants.py` | Prior cycle-level advantage filtering; useful formula precedent, different target/selection contract |

## Shared-subset validation (2026-11-12)

The focused filter/controller/learner suite passes **17 tests**, including exact
retained actor inputs, retained-weighted alpha/TES entropy, all-rows TES/PQL
parity, microbatch invariance, globally empty selection, frozen actor updates,
legacy resume compatibility, and a production two-rank Gloo update compared
with a concatenated selected-row reference. Four config/launcher tests also pass.

The existing SAC learner, PQL, TES, loading, artifact, startup, warm-start,
diagnostic, and distributed regression suites pass: **110 tests**, including
five CPU/Gloo distributed tests. Independent production-code review found no
unresolved correctness issue. Shell syntax and launcher dry-run tests pass.

CUDA/NCCL execution, compiled-shape behavior, training speed, and policy quality
have not been validated for this correction. The existing critic-only jobs
were not restarted; new `actor_critic` runs need fresh output directories.

## Historical validation: critic-only mode (2026-11-11)

The following results describe the original implementation. They do not certify
the corrected `actor_critic` implementation; its validation is recorded separately.

| Check | Result |
| --- | --- |
| New experiment-folder suite | 18 passed, including TES/PQL all-rows parity, nonzero PQL microbatch invariance, target alignment, persisted checkpoint state, global count metrics, two-rank gradient oracles, a real filtered n-step distributed update, and both launchers |
| Existing SAC learner/PQL/TES/loading/artifact/startup/warm-start/diagnostic suites | 105 passed |
| Existing SAC and PQL distributed suites | 5 passed on CPU/Gloo |
| Syntax, shell, resolved config keys/estimators, config/catalog parity, and source-artifact audit | Passed; 20 new source/config/documentation files, generated outputs ignored |

Run the focused suite from the repository root:

```bash
source .venv/bin/activate
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' PYTHONPATH=. pytest -q project/soft_actor_critic_experience_filtering/tests
```

`PYTHONPATH=.` makes the local `project` namespace importable to the pytest
entrypoint. Local dry-run evidence is saved in
`preflight/launch_jc6tb6ol/{tes,pql}/manifest.json`, with resolved trial configs
under each `configs/trial_98.yaml`. These artifacts describe prepared commands;
they are not evidence of a live run or W&B synchronization.

Independent implementation review found no material correctness issue. The
filter-specific full-loop distributed test exercises the n-step route; PQL has
single-process filtered parity/target tests plus the existing unfiltered PQL
distributed suite. A filtered PQL CUDA/NCCL probe remains a launch-validation gate.
