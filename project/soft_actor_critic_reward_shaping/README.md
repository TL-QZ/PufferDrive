# Potential-based reward shaping for SAC

Research consolidated on 2026-11-02. Documentation only: no reward, learner,
configuration, or training changes have been made for this investigation.

The advisor proposed adding route-progress feedback to help learning from
sparse, delayed goals, potentially alongside Peng-style Q(lambda) returns.
This is potential-based reward shaping (PBRS), a method with established
policy-invariance results. Those results justify the reward transformation
under their assumptions; they do not guarantee faster convergence or better
driving performance with neural networks.

## 1. Proposed reward and the distance sign

For reward r_t from action a_t in state s_t, define

```text
shaped_reward_t = r_t + F_t
F_t = kappa * [gamma * Phi(s_(t+1)) - Phi(s_t)]
```

Use the same discount gamma as the learner. A fixed kappa can be absorbed into
the potential; it sets the practical shaping scale.

If D(s) is nonnegative remaining route distance, the intended progress
heuristic for kappa > 0 is **Phi(s) = -D(s)**, giving

```text
F_t = kappa * [D(s_t) - gamma * D(s_(t+1))]
```

Example: gamma=0.999, distance 100 m -> 98 m:

```text
negative-distance potential: F_t =  2.098 * kappa
positive-distance potential: F_t = -2.098 * kappa
```

Positive distance is not intrinsically outside the PBRS theorem. It supplies
the opposite immediate heuristic from the intended encouragement of progress.
Policy invariance and helpful learning signals are different questions.

Discounted potential differences are not identical to raw progress:

```text
D_t - gamma*D_(t+1) = (D_t - D_(t+1)) + (1-gamma)*D_(t+1)
```

Thus standing still at positive distance can yield a positive shaping term.
The discounted trajectory sum and boundary treatment establish invariance,
not the sign of each individual bonus. Do not replace the formula with
`max(0, distance_progress)` and retain the same theoretical claim.

With distance measured in meters, kappa has units of reward per meter.
Alternatively, use Phi=-D/D_ref for a fixed positive reference distance D_ref;
then kappa is in reward units. No scale is selected yet. Inspect the actual
per-transition reward distribution and route distances before choosing it.

## 2. Why the objective is preserved, and where it can fail

For an uninterrupted T-transition trajectory, the shaping sum telescopes:

```text
sum_(t=0)^(T-1) gamma^t * F_t
    = -kappa*Phi(s_0) + kappa*gamma^T*Phi(s_T)
```

For a bounded potential in an infinite discounted task, the final term tends
to zero. For a terminated episodic return, setting all terminal potentials to
zero removes it. The resulting return differs by a starting-state quantity
that does not depend on the chosen actions. This is the central invariance
argument of Ng, Harada & Russell [1]. Finite episodes require particular care
with terminal potentials, as analyzed by Grzes [3].

For a fixed policy under consistent boundaries:

```text
Q_shaped^pi(s,a) = Q_original^pi(s,a) - kappa*Phi(s)
```

All action values at a state shift equally. Shaping changes the learning
problem's intermediate feedback without changing which action is optimal
in the idealized setting. It can help a learner discover useful trajectories
before sparse goal rewards have propagated through many updates.

The same telescoping reasoning applies to a fixed-temperature
entropy-regularized objective: shaping adds a policy-independent starting
state shift, while entropy terms themselves are unchanged. This is an
algebraic observation, not a guarantee about our learned-alpha/TES training
dynamics, approximation errors, exploration, or optimizer stability.

Maintain one well-defined potential on the task state and use consistent
discounts and boundary semantics. Clipping the shaping difference, omitting
selected transitions, or changing its scale during training need not retain
the same telescoping guarantee. A bounded/clipped state potential itself can
still be valid if its difference is applied consistently.

## 3. Compatibility with current n-step targets and planned PQL

Our current n-step convention sums observed rewards and adds a soft value at
the endpoint, without explicit intermediate entropy bonuses. See the
[existing n-step design](../soft_actor_critic_multi_step_return/design.md).

Apply shaping to raw one-step transitions before n-step aggregation or the
lambda-return calculation. For an uninterrupted n-step sequence:

```text
Y_shaped_t^(n) = sum_(k=0)^(n-1) gamma^k * shaped_reward_(t+k)
                + gamma^n * Vbar_shaped_soft(s_(t+n))

Vbar_shaped_soft(s)
    = sum_a pi(a|s) * [min_j Qbar_shaped_j(s,a) - alpha*log pi(a|s)]
```

Critics learn the shaped return. Keep the existing actor entropy, alpha,
and target-entropy scheduling definitions; no additional entropy term is
introduced by PBRS.

The [planned PQL option](../soft_actor_critic_multi_step_return_PQL/PLAN.md)
can use the shaped reward in its finite-segment recursion:

```text
G_shaped_t = shaped_reward_t
             + gamma * [(1-lambda)*Vbar_shaped_soft(s_(t+1))
                        + lambda*G_shaped_(t+1)]
```

At a sampled segment end, use the one-step bootstrap instead of an unavailable
continuation. At a true terminal, omit bootstrap. These return boundaries
must follow the same terminal/truncation contract as the underlying reward.

An algebraic consistency check explains why the two methods combine. If the
bootstrap critics have the corresponding state shift,

```text
Vbar_shaped_soft(s) = Vbar_original_soft(s) - kappa*Phi(s)
```

then the endpoint potential cancels against the shaped bootstrap:

```text
Y_shaped_t^(n) = Y_original_t^(n) - kappa*Phi(s_t)
```

Every n-step target has the same shift, so a lambda mixture whose weights sum
to one retains it. This identity assumes corresponding shifted value
estimates; it is not an assertion that independently trained neural critics
will satisfy the relation throughout training. Do not subtract Phi a second
time from a critic already trained on shaped rewards.

| Mechanism | What it contributes |
|---|---|
| PBRS | Domain knowledge through intermediate route-progress feedback |
| Fixed n-step returns | Several observed rewards in each critic target |
| PQL lambda returns | A mixture of short and longer observed-return targets |

PBRS can be tested with the current n-step learner independently of PQL. It
does not require sequence replay or resolve PQL's off-policy mismatch.

## 4. Driving-specific decisions before an implementation plan

1. **Choose a coherent route potential.** Distance to the next waypoint can
   jump from nearly zero to a large distance when that waypoint advances. A
   negative-distance potential then produces a negative shaping jump. Include
   the waypoint/route identity in the state definition and apply the same
   potential consistently across that transition. Do not silently suppress
   the jump while claiming PBRS invariance. Remaining distance along a fixed
   route to its final destination could be smoother if that quantity exists
   and is well-defined. Its availability in the current simulator has not
   been audited in this investigation.
2. **Specify true terminals and truncations separately.** For true terminals,
   use zero terminal potential, including failure terminals as well as goals.
   For time limits or collection cuts where learning bootstraps, retain the
   true final-state potential and consistent shaped value. Do not evaluate
   the potential on an auto-reset observation from another episode. An
   equivalent boundary-aware reward uses
   `r_t + kappa*(d_t*Phi(next_state_t)-Phi(state_t))`, with d_t=0 at a true
   terminal and gamma otherwise. A sampled segment cut is not a true terminal.
3. **Define scale and state validity.** Confirm route-distance units, finite
   values, route selection, and behavior for unavailable/unreachable routes.
   Decide these semantics before enabling shaping; do not silently substitute
   arbitrary distances. Fix the potential and kappa within an initial run.
   Route progress is privileged training information; it need not become a
   new actor input merely to compute the reward.
4. **Verify the reward transform independently.** Use numerical telescoping
   checks, terminal/timeout cases, waypoint advancement, and comparisons of
   one-step, n-step, and lambda targets under explicitly shifted critics.
   Check that kappa=0 reproduces the existing reward path. Test raw rewards
   before aggregation and keep invalid/reset-only rows out of reward sums.
5. **Keep the first experiment interpretable.** Compare the existing promising
   trial with and without shaping while preserving other settings. Treat
   PQL+shaping as a separate combination experiment. Log original reward,
   shaping reward, and their sum separately, and evaluate using the same
   original benchmark metrics. Track progress, goal achievement, safety, and
   late performance retention rather than shaped return alone. No experiment
   or launch is authorized by these suggestions.

Theorem-backed invariance does not establish practical convergence in this
partially observed, shared-policy multi-agent training system. The multi-agent
result in [4] is relevant support under its stochastic-game assumptions, not
proof that every implementation or learning trajectory preserves behavior.

## 5. Literature and reading order

| Reference | Main contribution | How to use it here |
|---|---|---|
| [1] Ng, Harada & Russell (1999), *Policy invariance under reward transformations: Theory and application to reward shaping*, ICML | Foundational potential-difference transformation and policy-invariance result; distance/subgoal heuristics | Main theoretical citation for the advisor's proposal |
| [2] Wiewiora (2003), *Potential-Based Shaping and Q-Value Initialization are Equivalent*, JAIR 19:205-208 | Equivalence to suitable Q-table initialization under specified conditions | Intuition for how prior knowledge changes learning; not automatic equivalence for neural function approximation |
| [3] Grzes (2017), *Reward Shaping in Episodic Reinforcement Learning*, AAMAS | Terminal-potential and episodic-boundary analysis | Read before specifying goal/collision/timeout behavior |
| [4] Lu, Schwartz & Givigi (2011), *Policy Invariance under Reward Transformations for General-Sum Stochastic Games*, JAIR 41:397-406 | Potential-based shaping preserves Nash equilibria under the stated stochastic-game framework | Multi-agent theoretical context |
| [5] Muller & Kudenko (2025), *Improving the Effectiveness of Potential-Based Reward Shaping in Reinforcement Learning*, extended arXiv paper; accepted as AAMAS extended abstract | Dependence of effectiveness on potential offsets, initial Q-values, and external rewards | Practical reasons a valid potential may guide learning poorly |

Primary sources:

- [1: Author-hosted paper](https://people.eecs.berkeley.edu/~russell/papers/icml99-shaping.pdf)
- [2: Paper](https://people.eecs.berkeley.edu/~pabbeel/cs287-fa09/readings/potential_based_shaping_equals_initialization.pdf)
- [3: AAMAS proceedings paper](https://aamas.csc.liv.ac.uk/Proceedings/aamas2017/pdfs/p565.pdf)
- [4: Author manuscript](https://arxiv.org/abs/1401.3907) — journal publication is 2011; arXiv posting is 2014.
- [5: Author manuscript](https://arxiv.org/abs/2502.01307)

Read [1] for the core argument, [3] for boundary correctness, then [5] for
practical potential design. Sections 1 and 3 above include our own algebraic
examples and proposed application to this project's SAC/PQL conventions;
they are not empirical results from these papers or from a new driving run.
