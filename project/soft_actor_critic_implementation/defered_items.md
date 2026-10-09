# Deferred items

These capabilities are deferred. Learner checkpoints remain usable: replay is refilled and environments restart after restore.

## 1. D4 — Replay restoration

**Purpose:** preserve the dataset of old experience across restarts.

Today, the learner's weights and counters are restored, but replay is empty. The next collection fills it again before normal replay refresh/training proceeds.

**Example:** suppose replay holds 10,000 transitions, including rare situations collected earlier. With D4, restarting reloads those same 10,000 records. With the current behavior, those records are gone and newly collected experience replaces the dataset.

This requires saving the five transition fields, occupancy and relevant storage metadata, plus replay-private RNG state if the storage implementation owns any. The loader would restore that data instead of requiring replay to remain empty. Checkpoints become larger and take longer to write/read.

The environment can still restart fresh. Stored transitions are complete independent records; they do not require the simulator to return to the scenes that produced them. New collection can refresh the restored dataset normally.

**If omitted:** learner restore remains valid, but it rebuilds the training dataset and incurs refill cost. Restoring global RNG alone cannot recover missing replay records.

**Status: deferred.** Replay will be refilled after learner restore.

## 2. D5 — Environment and collection continuation

**Purpose:** continue the live trajectories instead of resetting the environment.

Imagine the learner saved a pending observation/action pair for agent slot 7. To continue correctly, the next received reward and observation must be the outcome of that exact action in that exact world. Restoring just the pending arrays while resetting Drive would join unrelated transitions.

D5 therefore needs a consistent snapshot of:

- Simulator state: active scenarios/maps, agent states, episode progress and environment RNG.
- Learner collection state: pending observations/actions, flags and slot correspondence.
- Vector-worker state: outstanding actions or completed-but-unconsumed results, captured at a coordinated checkpoint boundary.

The existing [PPO checkpoint](../../pufferlib/pufferl.py#L878) does not save this live environment state either. D5 goes beyond its current learner-resume behavior.

**If omitted:** the simulator restarts and pending state is cleared, as SAC already does. You can still restore the learner, and you can independently add D4 to retain replay.

For full training continuation, D4 and D5 must work together with the existing learner/RNG checkpoint. Even then, bit-for-bit reproducibility also depends on deterministic execution.

**Status: deferred.** The environment will restart and pending collection state will be cleared after learner restore.
