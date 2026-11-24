# Round four · How long does a target’s action preference agree with driving outcomes?

For each frozen PQL, TES and PPO checkpoint, a starting state identifies one car at one reference frame in stochastic CARLA self-play. We aim for 20 distinct starting states per distance bin. Each of the 12 initial jerk choices is forced once in each of four stochastic repeats. Within a repeat, the 12 choices share continuation seeds. The four repeats use distinct seeds. This gives 48 branches per selected state.

## Starting goals

The goal axis follows route order. We check whether the current waypoint is already fulfilled using simulator coordinates and the probing car’s own goal-hit rule. If fulfilled, we look ahead in route order for the next outstanding waypoint. This calculation selects the probe and labels its bin prior to taking the forced action. The other two axes describe the nearest observed agent and observed road boundary. The goal edges in meters are 0, 2, 5, 12, 30, 100, 300 and infinity. A bin’s population reflects the corrected geometry.

## Exact horizons and early endings

A step counts once the simulator has applied an action and produced its reward. The forced action is step one. A branch always records its exact observed length, ending reason, and full endpoint measurements. It also records the metrics through steps 8, 32, 128, 1,280 and 2,560 whenever it reaches those steps. The transition ending the branch counts. A car ending on step 100 contributes to the 8-step and 32-step views and the full endpoint view. Its 128-step and longer measurements are missing. The saved endpoint can later be assigned to any desired range.

The critic training target follows its checkpoint’s original definition and stays fixed across these horizon comparisons. Each outcome uses only the chosen prefix. Its score, average speed and rates come from the native interval metrics. The reward-only discounted return is common across all checkpoints. SAC also has a sampled soft return with entropy from the continuation actions.

## Comparing actions

For a given start, action and horizon, average the target and the outcome over the same surviving repeats of that action. Compare the resulting action means within each state. Retain every underlying branch and the number of eligible repeats. The full endpoint comparison includes early terminations. Exact-horizon comparisons condition on reaching the horizon, so improved long-horizon outcomes can reflect which actions survived.

Use Pearson r for linear association, Spearman’s rho for association between average ranks, Kendall’s tau-b for ordinal association with tie correction, and directional pairwise ranking accuracy. Signed correlations use the recorded metric, so negative correlation with total infractions means higher targets accompany fewer infractions. Pairwise accuracy favors fewer infractions. Equal outcomes do not contribute to accuracy, and equal targets on distinguishable outcomes receive half credit. Report the number of states that can be scored alongside the averages.

[Round-four outputs](../runs/fourth_20261124/README.md)
