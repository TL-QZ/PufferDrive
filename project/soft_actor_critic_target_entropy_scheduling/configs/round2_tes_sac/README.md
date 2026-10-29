# Round 2 TES SAC

This isolated config set carries forward TES trials 97 and 98 with a replay
capacity of 2,048,000 transitions per rank. With minibatches of 128,000, each
replay pass has 16 logical updates; three passes produce 48 updates per train
cycle. Microbatches remain 32,000.

Trial IDs retain their grid mappings: both use gamma `0.999` and initial target
entropy `0.75 * ln(12) = 1.863680`; trial 97 uses replay refresh `0.1`, a 2B
agent-row budget and evaluation every 10 cycles; trial 98 uses refresh `0.5`, a
4B budget and evaluation every 2 cycles.

`target_entropy_required_count: 200` requires 200 accumulated qualifying
observations (at least four full training cycles plus eight updates).
`target_entropy_force_decay_steps: 500000000` additionally forces one target
reduction if 500 million global received agent rows have elapsed since the last
reduction. The initial timer starts at zero. Both normal and forced reductions
multiply the target by 0.9, reset the qualifying counter and restart this timer;
the moving statistics are retained. If both conditions trigger together, only
one reduction occurs. A large step-count jump also causes only one reduction.

The timeout uses `agent_steps`, summed across ranks, including masked/reset
rows. It does not count replay samples or optimizer updates. It is checked on
the first actor/alpha update after collection reaches the threshold, so a
collection cycle can overshoot 500M. The changed target applies to the next
alpha update. Critic-only updates do not advance the controller. Timer state is
saved in learner checkpoints. Track `losses/target_entropy` together with
`target_entropy/forced_decay_count`, `target_entropy/last_decay_agent_steps`,
and `target_entropy/steps_since_decay`.

For example, a normal reduction at 300M moves the forced deadline to 800M.
If no normal reductions occur, forced reductions occur near 500M, 1B, 1.5B,
and so on, subject to collection-cycle boundaries. Both trials use this rule;
trial 98 retains its 4B received-row budget and 50% replay refresh.

Run `scripts/launch_round2_tes_sac.sh --trial 97` or `--trial 98`. The launcher
uses this config directory, W&B group `round2_tes_sac`, and a unique
`runs/round2_tes_sac_<timestamp>_<pid>` output directory.
