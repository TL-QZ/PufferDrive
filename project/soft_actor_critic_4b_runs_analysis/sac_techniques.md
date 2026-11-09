## Techniques and changes across all seven SAC runs

The table describes **what was configured**, not a claim that one technique caused a score change. Run names match the results above; exact run IDs and source paths are in [setup_inventory.md](setup_inventory.md). The low-LR run is included here even though it has no full-evaluation result in this analysis.

| SAC run | Return / replay technique | Target entropy | Optimization and data collection | Goal spacing: training → evaluation | Main comparison |
|---|---|---|---|---|---|
| **TES round 2** | One-step soft SAC target | Adaptive TES, standard trigger | Higher LR; minibatch 128K; replay refresh 0.5; 3,200 agents | Original: 20–200 m → 30 m | Reference SAC configuration: adaptive entropy scheduling without multi-step returns |
| **n-step 4 round 2** | Fixed 4-step return | **Constant target entropy; no TES** | Higher LR; minibatch 128K; replay refresh 0.5; 3,200 agents | One-third: 6.67–66.67 m → 10 m | Versus TES round 2: adds 4-step returns, removes adaptive TES, and shortens goals together |
| **n-step 2 TES round 4.1** | Fixed 2-step return | Adaptive TES, standard trigger | Lower LR; minibatch 64K; replay refresh 0.5; 3,200 agents | One-third: 6.67–66.67 m → 10 m | Versus n-step 4: shorter return horizon, adaptive TES, lower LR, and smaller minibatch |
| **PQL round 4.1** | PQL target: horizon 128, λ=0.95; raw replay remains one-step | Adaptive TES, standard trigger | Lower LR; minibatch 64K; replay refresh 0.5; 3,200 agents | One-third: 6.67–66.67 m → 10 m | Closest return-method comparison with n-step 2 TES: fixed 2-step versus PQL target |
| **PQL official round 1** | Same horizon-128, λ=0.95 PQL target | Adaptive TES, slower trigger | Lower LR; minibatch 64K; replay refresh 0.5; **800 agents** | One-third: 6.67–66.67 m → 10 m | Versus PQL round 4.1: slower entropy adaptation and fewer collection agents together |
| **Low-LR n-step 4 TES (trial 01)** | Fixed 4-step return | Adaptive TES, slower trigger | Lower LR; minibatch 64K; **replay refresh 0.25**; 3,200 agents | One-third: 6.67–66.67 m → 10 m | Versus original n-step 4: adds TES, lowers LR/minibatch, and reduces replay refresh; learning curves only in this analysis |
| **PQL official round 2** | Same horizon-128, λ=0.95 PQL target | Adaptive TES, slower trigger | Lower LR; minibatch 64K; replay refresh 0.5; 800 agents | **Original restored: 20–200 m → 30 m** | Closest goal-spacing comparison with PQL official round 1; round 2 also has learner-only recovery |

### Exact settings behind the labels

| Setting | Higher LR / standard trigger | Lower LR / slower trigger |
|---|---|---|
| Actor / critic / alpha learning rates | 5e-4 / 5e-4 / 3e-4 | 1e-4 / 1e-4 / 7e-5 |
| Adaptive TES entropy standard-deviation threshold | 0.05 | 0.025 |
| Adaptive TES required qualifying updates | 200 | 2,000 |

Learning-rate and TES labels are separate choices: n-step 2 TES and PQL round 4.1 use **lower LR with the standard TES trigger**. Every adaptive TES run starts at target entropy 1.86368, uses EMA decay 0.999, mean-change threshold 0.01, entropy-decay factor 0.9, and a forced decay after 500M received-agent steps without a decay. The constant-entropy n-step-4 run keeps its target at 1.86368 while alpha itself is still learned.

Agent counts denote the saved `env.num_agents` per environment instance, not the combined count across workers and GPUs. Minibatch and replay-capacity settings are per learner rank. TES means target-entropy scheduling; its standard-deviation threshold refers to the controller's entropy statistic.

All seven SAC runs retain the entropy term in the critic target, gamma=0.999, tau=0.005, replay capacity 2,048,000 per rank, and three update epochs. None of these seven is a critic-target-entropy removal ablation. Replay refresh controls the fraction replaced per collection cycle; 0.25 versus 0.5 is not a change to replay capacity. PQL combines linked one-step transitions into a separate λ-return target; it is not fixed 128-step insertion into replay.

### How to compare the effects

| Question | Closest runs to compare | Interpretation limit |
|---|---|---|
| Does PQL help relative to a fixed multi-step target? | n-step 2 TES round 4.1 ↔ PQL round 4.1 | Similar listed optimizer/TES/goals settings, but separate training histories and recovery; not repeated-seed evidence |
| Does the slower TES trigger with fewer agents help PQL? | PQL round 4.1 ↔ PQL official round 1 | Both trigger and collection-agent count change; their individual effects cannot be separated |
| Does restoring original goal spacing help PQL? | PQL official round 1 ↔ PQL official round 2 | Both training and evaluation goal spacing change; round 2 includes recovery, so goal counts and score changes are not a pure training-only effect |
| Does the lower-LR, slower-TES, lower-refresh package help 4-step SAC? | n-step 4 round 2 ↔ low-LR n-step 4 TES | Several settings change together; compare learning curves at matched received steps, since the low-LR run is excluded from full evaluation here |

For every comparison, read **total distance traveled and distance per infraction together**, then score, progress and safety rates. Goal counts depend directly on goal spacing. Lower collision rates can also accompany less movement, so neither safety nor distance alone establishes improvement. The n-step-2, PQL round-4.1 and PQL official round-2 histories include learner-only recovery with fresh replay and environments. These are single runs per setup; the available results support comparisons of configurations, not isolated causal effects.
