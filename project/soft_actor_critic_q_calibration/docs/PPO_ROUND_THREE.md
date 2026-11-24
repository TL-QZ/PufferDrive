# PPO value targets in the third calibration run

The feed-forward PPO driving checkpoint supplies 12 action probabilities and a scalar value for the probing car. At a fixed observation the value is identical for all first actions. To compare actions, each first action is forced once and all subsequent joint actions are sampled from the frozen policy. Each recorded branch has its own value-training target and its own driving outcomes.

The reference dataset uses 2,560 self-play frames on each of eight CARLA maps. The selected states cover six nearest-agent bins, six road-boundary bins and four visible-goal bins starting at 12 m. The calibration environment uses the same 800-agent capacity and disabled phantom braking and partner blindness as the SAC probes. The saved PPO training config used a 3,200-agent capacity and enabled those two perturbations. The geometry manifest records the overrides and checkpoint identity.

## Value target

The forced action occupies row 0 of a fresh 128-row PPO learner segment. With the frozen policy, the importance ratio is one. The target uses at most the next 127 transitions. For each transition from state t, the temporal-difference residual uses its reward and the next state's value. GAE sums residuals backward using gamma 0.999 and lambda 0.95. The target at the initial state equals the initial value plus that unnormalized advantage. The last row of the training kernel has zero advantage. An earlier terminal masks all later residuals. For a timeout without an ego terminal, the trainer adds gamma times the *previous* state's value to the reward and sets the done mask. The probe follows that same timeout convention.

This is a reconstructed frozen-checkpoint target under a specified learner-segment alignment. A historical training example could occupy a different segment offset. Longer driving returns and metrics follow the probing car until it ends, the scene ends, or 2,560 steps elapse. Within a state, subtracting the same baseline from all 12 targets leaves both Pearson correlation and pairwise ranking unchanged.

The probe records eligibility after each simulated transition. A terminal destination can already be ineligible. That transition’s reward still enters the preceding action’s target, and its done flag closes the recurrence before the destination value can contribute.

The entropy coefficient acts in the PPO policy loss. The PPO panels use the ordinary discounted reward return. The comparison table includes that same reward-only measure for SAC and separately identifies SAC's sampled soft return.

## Interpreting the comparison

The 16 bin definitions and quotas are shared across checkpoints. Each checkpoint generates its own self-play states, so the cross-policy summaries compare distributions of situations. Each target uses some of the rewards that also enter the observed episode return. Treat the resulting associations as evidence about the target, not an independent test of an action-quality prediction made before the rollout.

- [PPO checkpoint inventory](../../soft_actor_critic_4b_runs_analysis/manifest.json)
- [PPO transition collection and value target](../../../pufferlib/pufferl.py)
- [Native GAE recurrence](../../../pufferlib/extensions/pufferlib.cpp)
- [Completed round-three comparison](../runs/third_20261122/TARGET_COMPARISON.md)
