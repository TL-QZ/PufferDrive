# Round 5: useful driving followed by persistent entropy collapse

**Finding:** the strongest supported mechanism remains weak entropy regularization,
followed by coupled actor/critic instability and a saturated policy. Round 5 adds
an important contributor: reducing replay refresh to 0.25 doubles optimization
exposure per new transition compared with Round 3's 0.5. This is a hypothesis
about the trigger, not a demonstrated causal attribution.

**The new reward metrics were recorded, but the collapse diagnostics were not enabled.**
The saved learner config omits `train.log_collapse_diagnostics`; its default is
false. No logit-range, action-probability, action-value-gap, gradient-norm or TD-error
diagnostic keys occur anywhere in the captured metrics. Therefore this run cannot
settle whether a critic outlier or an actor step initiated the failure.

## Snapshot and reproducibility

Run: `project/soft_actor_critic_multi_step_return/runs/round5_nstep2_tes_sac/`
`sac108_grid_20261102T224156Z029676942/trial_01` (trial **01**, not 98).
Snapshot on November 3, 2026: 581 learner records through 791.904M received rows;
290 evaluations through epoch 580 / 790.560M rows. Metrics advanced during this
investigation; these are captured endpoints, not a completed run.

Reproduce from the repository root:

```bash
source .venv/bin/activate
python project/soft_actor_critic_target_entropy_scheduling/entropy_collapse_fix/diagnose_round5.py
```

This updates only `evidence/round5/`: full training/evaluation CSVs, summary,
SHA-256 provenance (including saved benchmark configs), and the plot below.
The older diagnosis and its evidence remain separate.

![Round 5 entropy, temperature, losses, evaluations and rewards](evidence/round5/collapse.png)

## Failure timeline

| Epoch | Received rows | Entropy | Alpha | Critic loss | Eval score |
|---:|---:|---:|---:|---:|---:|
| 466 | 633.536M | 1.77385 | .000699 | .001647 | .29489 |
| 467 | 634.816M | 1.74725 | .000689 | .003845 | — |
| 468 | 636.096M | 1.63697 | .000687 | .016273 | .28997 |
| 469 | 637.408M | .40298 | .000707 | .021772 | — |
| 470 | 638.752M | 0 | .000756 | .086490 | 0 |
| 472 | 641.376M | 0 | .000826 | .107864 | 0 |
| 581 | 791.904M | 0 | .004580 | .036091 | Not scheduled |

The best saved evaluation was epoch 338: score **.59073**, speed **4.005 m/s**,
progress **.40055**, off-road **.03240**, collision **.01920**. Round 5 learned useful
driving before failing, unlike the weak pre-collapse Round 3 two-step run.

Critic loss rises almost tenfold from 466 to 468, while average entropy remains
1.64. It reaches approximately 65.5 times the epoch-466 loss at 472. This localizes
an instability before exact-zero entropy; cycle averages cannot order individual
actor and critic updates. Replay-action mean Q falls from about 2.14 to negative
values afterward. Mean Q is not an action gap, and matching mean Q and mean target
does not rule out large individual TD errors.

All **112** captured learner records from 470 through 581 have exactly zero
entropy. Alpha increases about **6.1 times** after the first zero without recovery.
All 56 evaluations from 470 through 580 have the same vector: score 0, goals 0,
speed .97719 m/s, off-road .52288, collision .20936. This is consistent with
persistent saturated behavior; it does not prove every observation selects the
same action or prove floating-point underflow.

## Why the refresh change matters

Saved Round 3 and Round 5 learner configs match in environment, policy, gamma,
learning rates, target schedule and n-step horizon. The substantive train-config
change is `replay_update_ratio: 0.5 -> 0.25`; run/output/catalog paths also differ.
These are separate runs across different code revisions, not a controlled causal
experiment, even though the training configs closely match.

Per rank, replay capacity is 2,048,000 and the logical minibatch is 128,000.
Three passes give `3 * 2,048,000 / 128,000 = 48` learner updates per cycle in both
runs. Round 5 inserts 512,000 new valid transitions per rank per steady-state cycle
instead of 1,024,000. Consequently:

- Training exposure per new transition rises from `3 / .5 = 6` to `3 / .25 = 12`.
- Replay retains older behavior longer in learner cycles, which can increase
  actor/critic mismatch and intermediate behavior-policy mismatch for two-step returns.
- Temperature and target-network updates also happen more often per new transition;
  the change does not isolate actor optimization alone.

Both runs collapse near 640M received rows: Round 3 at 643.936M / epoch 238;
Round 5 at 638.752M / epoch 470. Similar row counts with different update counts
are not evidence of a fixed universal failure threshold. Round 5's pre-collapse
alpha is much smaller (~.00069 versus ~.004), consistent with more optimizer
updates and a long period of entropy above target. Replicates are needed to
attribute this difference specifically to replay refresh.

## What the extra reward metrics establish

The training component logs are means of episode-summed rewards, not per-timestep
rewards, and are sparse across
learner records. They are not the minibatch rewards or TD targets that caused the
optimizer step; training uses randomized reward coefficients and evaluation uses
its saved benchmark settings, so their magnitudes should not be directly equated.

At epoch 580, training episode return is -2.40982. Comfort contributes -1.60691
(about two thirds of that net negative return), off-road -.45668, collision
-.17896, and goal +.00624; velocity reward is zero. At the best evaluation, goal
reward is +17.02336 and comfort only -.06286. The collapsed policy loses goal
achievement and generates repeated comfort/safety penalties. This supports a
behavioral failure rather than a score-only artifact, but does **not** demonstrate
that the comfort coefficient triggered the collapse. Turning it off based on this
post-collapse decomposition would be premature.

## Mechanism and next experiment

The source actor objective is `sum_a p(a|s) * (alpha * log p(a|s) - min(Q1,Q2))`
([pufferl_sac.py](../../../pufferlib/pufferl_sac.py), `_policy_loss`). Small alpha
makes action-value differences more decisive. Once probabilities saturate, the
probability-weighted actor gradients can become tiny or zero; increasing alpha
then need not revive exploration. This matches the observed persistence, although
raw logits are required to establish the numerical mechanism.

The alpha loss has the correct feedback direction: `log_alpha * (H - target_H)`.
Alpha rises as entropy falls. The target remains **1.677312** across the failure;
its only recorded forced decay was at 500.992M rows, about 137.76M rows earlier.
There is no target-decay event at the collapse. The target is a soft objective,
not a guaranteed entropy floor. Learning rates remain near their initial values;
this is not an optimizer stopping because its rate reached zero.

The instrumentation gate is in `_train_sac_transition` at
[pufferl_sac.py](../../../pufferlib/pufferl_sac.py:1735), with metric definitions in
[sac_diagnostics.py](../../../pufferlib/sac_diagnostics.py:33).
For the next **fresh** run, retain the Round 5 setup and explicitly enable
`train.log_collapse_diagnostics=true`. Verify the first metrics record contains
`actor/grad_norm_max`, `policy/logit_range_max`,
`policy/q_action_gap_over_alpha_max`, and `critic/abs_td_error_max` before spending
a full training budget. These interval summaries will improve localization but
still cannot reconstruct the exact ordering of individual minibatches.

For a prevention ablation, reduce only actor LR from .0005 to .0001 while retaining
`n_steps=2`, refresh .25, and the private benchmark catalog. Treat constant target
entropy as a separate control. Neither intervention has been validated here, and
the older one-step candidate configurations should not silently substitute for
this two-step run. No trainer/config changes or GPU jobs were made by this analysis.
