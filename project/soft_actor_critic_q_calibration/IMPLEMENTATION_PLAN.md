# Frozen self-play action probes

## Questions and measurements

From the same CARLA scene, do online twin critics and their implemented training targets correctly predict relative returns of alternative first actions? All vehicles remain controlled by one frozen policy. A branch forces only the probing agent's first action, then returns it to sampled self-play.

The eight visible entries are discounted reward return, standard soft return, online Q1, online Q2, min-Q, the checkpoint-specific training-formulation target, safety/ending flags, and goals reached. Safety includes collision, at-fault collision, off-road, and red-light events. Goal count also determines whether any goal was reached. Metadata and censoring are mandatory, not optional metrics.

Standard soft return includes sampled future-action entropy, excludes the forced initial action's entropy, and is separate from the actual PQL recurrence. PQL targets must use the existing implementation and checkpoint lambda/horizon; no algorithm fixes are part of this experiment.

## First experiment

- Models: matching full TES and PQL checkpoints under `project/sac_gpu_4b_comparison`; exact paths, progress, configuration, and hashes are written into the run manifest.
- Per model: two independent CARLA scene initializations, three probing times, two active probing agents per time, all twelve actions, eight continuation seeds: at most 1,152 branches.
- Reference episodes retain the saved dt (expected 0.3 seconds), scenario horizon (expected 2,560 frames), early-reset semantics, reward randomization, scene density, and sampled action policy.
- Explicit all-policy-control override: set partner-blindness and phantom-braking probabilities and trigger probabilities to zero in the experiment copy. Both checkpoints trained with probabilities 0.02 and triggers 0.03; these scripted behavior overrides conflict with the requested all-agent policy control. Record the deviation without changing source training configs.
- Adapt probing times to each reference episode's actual lifetime: select three spread-out distinct eligible pre-action frames, near 10%, 30%, and 50% of the eligible span. Select agents without using Q or future outcomes. Report shortages explicitly rather than sampling across a reset or fabricating probes.
- End each probe trajectory at its true agent terminal or the original scene boundary. Preserve timeout/cap censoring. No zero-tail claim for incomplete returns.

## Implementation approach

Keep new source, tests, documentation, and generated outputs inside this project. Preserve the user's brainstorm and learning notes. Do not restore the deleted experiment or alter training checkpoints/configuration.

Use deterministic initialization and recorded joint-action prefixes to reconstruct scenes if no verified snapshot API exists. Verify reconstruction before forcing any action. Batch scene copies for policy inference; use independently seeded continuation streams, paired across alternative first actions. Make task identities independent of batch size. Persist branch records incrementally and support restarting completed task groups without repeating them.

Use isolated native instrumentation only where required for per-agent event metadata and pre-reset boundary capture. Preserve training-mode dynamics and randomization. If instrumented code prevents automatic reset, verify parity through the first episode boundary against the production extension. Do not switch on evaluation mode merely to hold ended scenes: that changes training semantics.

Keep compact scene/ego starting metadata and reference joint-action prefixes so later scenario categorization does not require redoing the reference rollout. Avoid storing full observations for every continuation step. Record source hashes and deviations from saved configuration.

## Validation gates

1. Checkpoint/configuration audit and mathematical unit tests: reward/entropy indexing, terminal versus timeout, PQL parity, action indexing, and no cross-reset continuation.
2. Native/production transition parity, deterministic scene reconstruction, and branch seed reproducibility.
3. Execution gate for each model: one probe, twelve first actions, two repeats (24 branches), with runtime and memory evidence.
4. Independent implementation review; fix material issues before the first experiment.
5. Launch both model experiments in durable sessions with manifests, progress, logs, and incremental data; verify live progress before reporting launch success.

CPU checks are preflight, not GPU throughput or calibration evidence. Initial summaries report uncertainty and missing tails. Agreement with target-critic bootstraps is consistency evidence, not independent ground truth. Probes from one scene are correlated, and two initializations are not a population-level evaluation.
