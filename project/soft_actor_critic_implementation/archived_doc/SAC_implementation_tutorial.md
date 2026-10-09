# SAC implementation: follow one transition through the learner

**Read this beside [`pufferlib/pufferl_sac.py`](../../pufferlib/pufferl_sac.py). Start with `evaluate()`, then `train()` and `_train_sac_transition()`.** These own collection, cycle bookkeeping, and full-replay minibatch epochs, respectively. The names follow the PPO implementation: `policy` is the actor, `global_step` counts received agent-slot rows, and `epoch` counts collection/training cycles.

This tutorial explains how the operations fit together, including why each exists. We will use two actions and small tensors so every reduction can be inspected. Those examples are invented arithmetic examples, not Drive measurements or Drive's actual observation layout. Drive's default discrete actor has 12 categories; the learner uses the same operations with a wider action dimension.

The public trainer takes supplied models, replay and vecenv. Its run lifecycle now follows PPO conventions, while full Drive startup still needs component construction and pre-reset observation transport. See the [implementation plan](planning.md) for that boundary and the [source walkthrough](trace_ppo_and_plan_sac.md) for the call flow and validation record.

## 1. Components and responsibilities

### Start with the operation PPO already makes familiar

In PPO, collection produces observations, chosen actions, rewards, old log probabilities and value predictions. Training then revisits the saved observations and actions. It does not run the simulator again to compute each gradient.

SAC also revisits stored experience. The key difference is what it predicts and how it uses those predictions. PPO's value head produces one scalar `V(s)`: the estimated return from a state under the policy. A discrete SAC critic produces a vector `Q(s, :)`: one estimated return for each possible first action.

For a two-action example:

```text
observation s

policy(s) -> probabilities [0.25, 0.75]
             "Choose action 0 with probability 0.25; action 1 with 0.75."

critic(s) -> Q1 [2.0, 5.0]
             Q2 [3.0, 4.0]
             "Estimate the return of each first action, then following the policy."
```

Probabilities and Q values answer different questions. A probability describes the policy's choice. A Q value estimates the consequence of choosing an action. A high Q value does not itself execute that action; the actor still supplies the distribution used for collection.

### Why are there two critics?

A learned Q estimate can be too optimistic. If the policy learns to prefer actions precisely because their estimates are accidentally high, this estimation error can reinforce itself. SAC uses two critics and takes their elementwise minimum when constructing targets and improving the policy.

With the values above:

```text
Q1                 [2.0, 5.0]
Q2                 [3.0, 4.0]
minimum(Q1, Q2)     [2.0, 4.0]
```

The minimum is taken for each action separately. It does not pick one critic for the whole batch, and it does not choose the minimum-valued action. Both critics are trained against the same target using their own predictions. This reduces one source of overestimation; it does not make either prediction ground truth.

In the implementation, `self.critic` is one supplied module containing both critics. Its forward returns `(q1, q2)`. The singular name follows the model naming convention; the pair of returned tensors is the twin-Q contract.

### Online critics and target critics have different jobs

`self.critic` changes through gradient descent. `self.target_critic` is a complete frozen copy used to construct the number that the online critics should predict. If every online change immediately changed its own target by the same amount, regression would chase a rapidly moving quantity. Slowly updating the target copy reduces that movement.

The target includes the critic encoders, not just the output layers. Otherwise a supposedly slow target head could still receive features from a rapidly changing online encoder. `_train_sac_transition()` applies Polyak interpolation to the complete copy after each learner update; lesson 4 shows the arithmetic.

| Object in the file | What it supplies | How it changes |
|---|---|---|
| `self.policy` | Logits defining action probabilities | `self.optimizer` |
| `self.critic` | Two Q vectors for each observation | `self.critic_optimizer` |
| `self.target_critic` | Slow Q estimates for next-state targets | Interpolation after each learner update |
| `self.log_alpha` | Positive entropy weight through `exp(log_alpha)` | `self.alpha_optimizer` |
| `self.vecenv` | Current observations and outcomes from asynchronous agent slots | `evaluate()` receives and sends every collection cycle |
| `self.replay` | Previously observed complete transitions | `evaluate()` appends or replaces; learner reads exact indexed batches |

Here **policy evaluation** means fitting critics to return targets. **Policy improvement** means changing action probabilities using those critics. Neither phrase refers to running a benchmark. Also, `PuffeSAC.evaluate()` has the same potentially confusing name as PPO's method: it **collects training experience**.

Modern SAC does not need a separate learned V network because a state value can be computed from Q values and the policy's probabilities. It also uses the current policy for next-state actions, so this version needs no target actor. The slow copies are the critics. The supplied [Spinning Up implementation](reference_implementations/open_ai_spinningup/sac.py) and its [SAC explanation](https://spinningup.openai.com/en/latest/algorithms/sac.html) provide the underlying continuous-SAC formulation; the finite action sum used here is developed below.

## 2. One transition through replay

### First complete a transition

Imagine agent slot 7 at simulator step 17. `PuffeSAC.evaluate()` has observation `o17`, runs the policy, samples category `a17`, maps that category to Drive controls, and sends those controls to the environment.

At that moment it does **not** yet know the reward or successor observation. It keeps the old action data in three slot-indexed arrays:

```text
self.observations[7] = o17
self.actions[7] = a17
self.pending[7] = True
```

Assume this is an ordinary, nonterminal step. When its outcome arrives, it provides reward `r18`, current observation `o18`, and boundary information. The complete transition is:

```text
(o17, a17, r18, o18, discount18)
```

The subscripts explain the timing: reward `r18` is produced by executing action `a17`. It is not the outcome of the next action that will be chosen from `o18`. PPO's rollout arrays place these quantities in shifted columns and compensate inside advantage computation. SAC's replay record stores the already matched pair together.

When `vecenv.recv()` later returns the outcome, `evaluate()` first joins and preflights this old action by slot ID. It then runs the policy on the current returned observations, inserts the previous completed transition, stores the new observation/action/pending state, and sends the controls. The actual order is `recv -> mask/boundary preflight -> policy sample -> replay insert -> save pending -> send`. The agent need not have completed its episode. One-step learning needs the next step's result; it does not need the complete future trajectory because the critic estimates the remaining return.

### Collecting, retaining and sampling are separate operations

These words are easy to mix up:

1. **Collecting** creates new experience by executing actions in the simulator.
2. **Retaining** keeps that experience in memory after the policy changes.
3. **Sampling** selects retained records for a gradient update.

Suppose yesterday's policy chose action 0 at observation s. That stored record continues to say “action 0 produced this reward and next observation.” Today's policy may prefer action 1. Sampling the old record does not rewrite its action or regenerate its outcome under today's policy. Instead, the learner uses the actual stored action for critic regression and today's policy when estimating what happens afterward.

This is the useful separation in off-policy learning: the policy that collected a transition need not be the policy currently being improved. Standard one-step SAC does not need PPO's old/current action-probability ratio for these updates.

### Indexed replacement controls replay retention

Replay has a fixed logical capacity. During the initial fill, complete copied records append into unoccupied indices. Once full, each collection refresh chooses the number of old records it plans to replace, then preselects that many distinct occupied indices uniformly without replacement. New eligible records overwrite those indices in arrival order. Replay does not evict the oldest records first.

For capacity four, imagine a refresh of two records whose preselected indices are `[3, 1]`:

| After insertion | Slot 0 | Slot 1 | Slot 2 | Slot 3 |
|---|---|---|---|---|
| Initial fill | A | B | C | D |
| Replace index 3 with E | A | B | C | E |
| Replace index 1 with F | A | F | C | E |

The next refresh draws a new set of distinct indices. This is uniform indexed replacement, not FIFO age replacement. The exact selected set for one refresh is fixed before collection starts, so each accepted new record has a unique target. `replace(indices, ...)` copies the incoming fields over those exact records; `get_batch(indices)` reads the exact requested records. CPU `int64` index vectors make that contract explicit.

Training uses a different permutation: each epoch shuffles all currently occupied replay indices, then visits them in sequential minibatches. A record appears exactly once within that epoch, although its order changes on the next epoch. Nearby replay indices are not reliable temporal neighbors: they may describe different agents, episodes or collection times. Each record carries its own next observation; the learner never treats an adjacent index as a successor state.

### Why asynchronous collection needs agent identity

`PuffeSAC.evaluate()` retains three arrays across collection/training cycles: `self.observations`, `self.actions` and `self.pending`. Each array is indexed by the actual agent-slot ID. Receive-row order alone cannot identify the pending action.

Workers can overwrite shared observations on the next `send()`. The trainer copies each new observation/action into its pending arrays, and `replay.add(obs, actions, rewards, next_obs, discounts)` must retain its own copies. A Python reference to worker memory would let future simulator steps change historical experience.

```text
recv outcome + saved observation/action for the same slot
    -> complete, eligible transition
    -> replay-owned copies retained across policy updates
    -> indexed into the full replay training dataset
```

### Walk one asynchronous receive with actual IDs

Suppose `self.pending` has actions for slots 2 and 7. Slot 2's action was sent at simulator time 40; slot 7's at time 43 because its worker returned later. A vector receive can return IDs in either order. The returned arrays below are ordered `[7, 2]`, so use those exact IDs to read pending data:

| Receive row | `env_id` | Saved action time | Returned reward | Current `obs` meaning | `pending` before join |
|---:|---:|---:|---:|---|---|
| 0 | 7 | 43 | `r44` | observation after action 7 | true |
| 1 | 2 | 40 | `r41` | observation after action 2 | true |

The first complete record joins row 0's reward to slot 7's saved observation and action. The second joins row 1 to slot 2. The batch is not a continuous slice from slot 2 through slot 7. It can be stored in receive-row order because each replay row is complete; temporal adjacency between those rows is irrelevant.

On the very first receive after `vecenv.async_reset()`, `self.pending` is false for every slot. The observations are valid inputs for the first policy call, but the returned reset rewards and flags have no preceding SAC action, so no replay record is created. `evaluate()` still samples actions, saves each observation/action under its actual ID, marks that slot pending, and sends controls.

### Keep action presence separate from the Drive mask

The boolean `self.pending[id]` answers “did this trainer send an action whose outcome has not been joined?” It is set true whenever that slot receives a sent action, regardless of `mask`. The mask returned alongside an outcome answers a different question: “was the action that just completed eligible to become replay?” Drive computes its mask before movement. That action can move the agent into a stopped/removed terminal condition after the mask was written, so a true mask may accompany a valid terminal outcome.

| State | Pending action? | Receive mask | Replay action outcome? | What `evaluate()` does |
|---|---:|---:|---|---|
| First reset receive | no | either | none exists | Skip insertion; choose and send the first action |
| Ordinary active step | yes | true | eligible | Store reward, returned next observation and discount gamma |
| Valid stop/removal terminal | yes | true | eligible, terminal | Store terminal reward with zero discount; do not pair it with reset observation |
| Masked/ineligible outcome | yes | false | not replay eligible | Do not insert; still process the returned observations for the next action |
| New reset observation after a prior terminal | prior outcome was matched | independent | new action may be sent | Set pending for that new action; do not reuse the previous receive mask |

This split prevents two opposite mistakes. Discarding every row with `terminated=True` would lose the reward from the last valid action. Carrying the old mask forward to filter the new observation could reject a valid first action after reset. The pending flag controls identity and timing; the current receive mask controls insertion eligibility. These semantics describe the supported Drive-style contract, not every vector environment that happens to implement the same four method names.

### See the policy action and the simulator control

The actor output for a batch has shape `[B,K]`, where K is the joint categorical action count. `sample_logits` samples a category ID for each row and maps those IDs to the matching continuous Drive controls because the configured environment uses a continuous action space. Replay stores the sampled category ID, since the discrete critic estimates `Q(s, a_id)`. `vecenv.send()` receives the mapped controls. For `B=1`, the returned control tensor still has its batch dimension, such as `[1, 2]`; do not squeeze it to `[2]`.

The collection path does not store PPO's old log probability, entropy estimate, scalar value prediction or recurrent episode-row indices. SAC's off-policy critic regression needs only the executed discrete action and its observed reward/successor. Current-policy probabilities are recomputed from replayed observations during each learner update.

## 3. One critic update

### Follow the tensors entering a replay update

`replay.get_batch(indices)` returns a plain tuple containing the exact requested records. `_train_sac_transition()` creates one shuffled CPU `int64` permutation per epoch, slices it into sequential minibatches, and transfers each exact batch to the learner device before calling the three math helpers:

```python
replay_indices = torch.randperm(len(self.replay), dtype=torch.int64)
for start in range(0, len(replay_indices), self.config["minibatch_size"]):
    batch_indices = replay_indices[start:start + self.config["minibatch_size"]]
    mb_obs, mb_actions, mb_rewards, mb_next_obs, mb_discounts = (
        tensor.to(self.device) for tensor in self.replay.get_batch(batch_indices)
    )
    # Apply one complete SAC update to this minibatch.
```

Repeat that traversal `update_epochs` times, creating a fresh permutation for each pass. A short final batch is still processed. `mb_` means minibatch, matching PPO's `mb_obs`, `mb_actions` and `mb_returns` naming. Let B be the actual number of records in this minibatch, K the number of discrete actions, and D the observation width in a simple flat example.

| Tensor | Shape | What one row means |
|---|---|---|
| `mb_obs`, `mb_next_obs` | `[B,D]` | The observation before and after one stored action |
| `mb_actions` | `[B]`, int64 | The category that actually executed |
| `mb_rewards`, `mb_discounts` | `[B]`, FP32 | Its reward and multiplier for remaining return |
| `logits`, `probs`, `logprobs`, `q1`, `q2` | `[B,K]` | One entry per action for each sampled observation |
| `mb_targets` | `[B]` | One regression target for each stored action |

Observations can have more dimensions than `[B,D]`; the leading dimension still identifies replay records in this minibatch. Rewards and discounts deliberately have shape `[B]`, not `[B,1]`. Mixing `[B,1]` and `[B]` can broadcast into `[B,B]`, comparing records against the wrong targets.

The replay implementation and `evaluate()` insertion boundary establish these contracts before storing a transition. The learner trusts them: it does not repeatedly scan each policy output for shape, dtype and finite values. Model outputs are internal tensors; repeated defensive checks there obscure the algorithm and can force device synchronization.

### What number should a critic predict?

For a stored `(s,a,r,s')`, the reward r has already happened. The remaining return is unknown. SAC estimates it by asking: at s', which actions would the **current** policy choose, and how good do the **target** critics think those actions are?

`_compute_targets()` first calls the policy on `mb_next_obs`. The policy returns logits, not normalized probabilities. `log_softmax` converts them to log probabilities; exponentiating yields probabilities:

```python
logits = self.policy(mb_next_obs)
logprobs = F.log_softmax(logits, dim=-1)
probs = logprobs.exp()
```

Each row of `probs` sums to one. The final dimension is the action dimension, so `dim=-1` applies the operation across actions, separately for each record.

Now take one example with two next-state actions:

```text
probs                         [ 0.25,      0.75     ]
logprobs                      [-1.386294, -0.287682 ]
target Q1                     [ 2.0,       5.0      ]
target Q2                     [ 3.0,       4.0      ]
minimum target Q              [ 2.0,       4.0      ]
```

The expected Q is `0.25×2 + 0.75×4 = 3.5`. “Expected” means probability-weighted average. There is no need to sample one next action because both possible actions fit in the vector.

SAC also includes an entropy contribution. Entropy here is `−sum(probs * logprobs)`, approximately `0.562335`. With alpha `0.2`, the additional contribution is approximately `0.112467`. The resulting soft next-state value is `3.612467`.

The actual code combines these operations:

```python
next_q1, next_q2 = self.target_critic(mb_next_obs)
next_value = (
    probs * (torch.minimum(next_q1, next_q2) - self.alpha * logprobs)
).sum(-1)
mb_targets = mb_rewards + mb_discounts * next_value
```

Notice the shapes changing: `[B,K]` becomes `[B]` at `.sum(-1)`. For each sampled transition, we sum the probability-weighted contributions of its actions. Averaging across actions would divide the expected value by K a second time.

In mathematical notation, exactly the same operations are:

\[
V_{soft}(s')=\sum_a \pi(a|s')\left[\min_i\bar Q_i(s',a)-\alpha\log\pi(a|s')\right],
\qquad y=r+dV_{soft}(s').
\]

The bar marks a target critic. The symbol d is the already prepared bootstrap discount, **not** a done flag. With reward 1 and discount 0.9, the target is `1 + 0.9×3.612467 ≈ 4.251220`. This number is what the online critics will be asked to predict for the stored `(s,a)`.

### Which critic output gets trained?

The target averages possible actions at the **next** observation. Regression selects the **executed** action at the original observation. These are different operations on different inputs.

For two stored records, let their executed actions be `[0,1]`:

```text
online Q1 at mb_obs        online Q2 at mb_obs
[[2, 5],                  [[3, 4],
 [2, 5]]                   [3, 4]]

mb_actions = [0, 1]
selected Q1 = [2, 5]
selected Q2 = [3, 4]
```

`gather(1, mb_actions.unsqueeze(-1))` selects column 0 for row 0 and column 1 for row 1. `unsqueeze` turns `[B]` indices into `[B,1]` for `gather`; `squeeze` removes that final singleton afterward.

Suppose the first target is the `4.251220` value above and the second is a terminal reward of 2. `_critic_loss()` computes:

```text
Q1 loss = mean([(2 − 4.251220)^2, (5 − 2)^2])
Q2 loss = mean([(3 − 4.251220)^2, (4 − 2)^2])
critic loss = Q1 loss + Q2 loss
```

This is the policy-evaluation step: adjust Q predictions toward observed immediate reward plus estimated future return. It does not require stored collection-time V estimates, old action log probabilities or PPO advantages.

### Why is the target detached?

`_compute_targets()` runs under `torch.no_grad()`. Target construction uses the policy, target critics and alpha to produce a fixed regression label for this update. Critic loss should move the online prediction toward that label; it should not change the label's generating networks to make the current error smaller.

During `critic_loss.backward()`, only the online critic graph exists. The policy and temperature get no gradient from this loss, and the target critic's parameters remain frozen. “Frozen” here means excluded from gradient learning; the later interpolation still changes target parameters deliberately.

### Termination and truncation change the multiplier

For a true terminal transition, no future return belongs to the episode. Discount is zero, so `target = reward` regardless of next Q estimates. For an ordinary continuing step, discount is gamma, currently configured with default `0.99`. The learner does not multiply by gamma again. A true terminal includes a valid action that stops or removes an agent; the reward from that action is retained exactly once.

An artificial time limit can stop data collection even though the underlying task could continue. Its discount stays gamma, and the backup needs the **final observation of the interrupted trajectory**. Drive resets before returning the observation, so an eligible pure truncation currently raises `NotImplementedError` before the batch is inserted into replay.

Consider these two observations:

```text
last observation before reset: car approaching an obstacle
first observation after reset: car at a new scenario's starting position
```

Using the reset observation would teach the critic that the old action leads to the new scenario. Shape checks cannot detect this error: both observations have the expected shape. The missing observation must be captured before C reset or Python map replacement; Python cannot recover it from the reset result.

For an inserted terminal record, `next_obs` uses a finite placeholder copied from that transition's saved pending observation. The record still satisfies the replay tensor shape contract, while zero discount makes the placeholder irrelevant to the target. The just-returned reset observation is never mislabeled as the terminal transition's successor. Keep the placeholder finite because the implementation evaluates the target network over minibatch rows before applying the discount; zero multiplied by a nonfinite value is not guaranteed to produce a finite target.

If both `terminated` and `truncated` are true, termination takes precedence and the discount is zero. The current design assumes continuing driving with `terminate_on_goal=0`; [the plan](planning.md#4-replay-collection-is-the-first-correctness-gate) records the task-boundary decisions.

PPO needs the same distinction. Its current truncation workaround adds gamma times the preceding stored value, then cuts return recursion. That preceding-state estimate is a proxy, and the workaround is skipped at column zero. It does not supply the true final-state bootstrap required by either algorithm. [Trace §3.5](trace_ppo_and_plan_sac.md#35-handle-masks-and-reset-boundaries-explicitly) identifies the source paths and pending Drive work.

## 4. Actor, temperature and targets

### Policy improvement: change probabilities using the critics

After the critic step, `_policy_loss()` runs the policy on `mb_obs`, the original sampled observations. The policy is allowed to consider **every** action at those states. It is not restricted to the actions that happened to be stored in replay.

For one state, suppose the updated minimum Q values are `[2,4]` and the current probabilities are `[0.25,0.75]`. Without entropy, shifting probability from action 0 to action 1 increases expected Q. The actor gradient provides the direction for that shift through the logits.

SAC includes entropy so the policy trades off concentrating on high-Q actions against retaining uncertainty. In code:

```python
logits = self.policy(mb_obs)
logprobs = F.log_softmax(logits, dim=-1)
probs = logprobs.exp()
with torch.no_grad():
    q1, q2 = self.critic(mb_obs)
policy_loss = (
    probs * (self.alpha * logprobs - torch.minimum(q1, q2))
).sum(-1).mean()
```

The actor minimizes negative expected Q minus alpha times entropy. Equivalently, it maximizes expected Q plus alpha times entropy. With the example values and alpha 0.2, the loss is approximately `−3.612467`. A negative loss is normal here; its sign is not a measure of success or failure.

There are two reductions with different meanings. `.sum(-1)` computes the action expectation for each state. `.mean()` averages those per-state objectives over sampled records. One minibatch row does not need to be consecutive in time with another.

### How can the actor learn if the critic forward has no gradient?

In discrete SAC, the Q vector acts as a fixed score for each category during this actor step. The trainable quantities are the probabilities that weight those scores. Autograd follows:

```text
policy parameters -> logits -> probs and logprobs -> policy_loss
                                   ^
                    fixed Q scores are multiplied here
```

The policy therefore gets a gradient even though the Q values are detached. We want to change which actions it chooses, not alter the critic's assessment just to make the actor objective look better. `self.alpha` is also detached in this objective. Temperature gets its own loss.

Separate optimizers provide separate parameter ownership. That alone is not enough to prevent unwanted gradients: the `no_grad()` and detach boundaries establish which graphs are built. Existing critic gradients may remain in `.grad` after its optimizer step, but actor backward adds none to them. Each optimizer clears its own gradients before its next backward.

### Temperature adjustment: derive the direction

Alpha sets the relative weight of entropy. Instead of choosing it permanently, this implementation trains `log_alpha`. Exponentiating gives a positive alpha without clipping it after every update.

For K categories, maximum entropy is `log(K)` with natural logarithms. The default target is `0.98×log(K)`. For K=2 it is approximately `0.679284`; the example `[0.25,0.75]` distribution has entropy `0.562335`, which is below that target.

The implementation uses the detached entropy from the actor forward and computes:

```python
alpha_loss = self.log_alpha * (entropy - self.config["target_entropy"])
```

In the example, the derivative with respect to `log_alpha` is approximately `−0.116949`. Gradient descent subtracts that negative gradient, increasing log-alpha and hence alpha. Future actor steps then put more weight on entropy. If entropy exceeds the target, the derivative is positive and alpha decreases instead.

Entropy is detached so this loss adjusts the weight rather than updating the actor a second time. It is measured before the actor optimizer step in this minibatch. The surrogate follows the supplied [discrete SAC reference](reference_implementations/sac-discrete.pytorch/sacd/agent/sacd.py); using `exp(log_alpha)` directly in the temperature objective would have the same desired direction but a different gradient scale.

### Target update: slowly follow the online critics

After all three optimizer steps, `_train_sac_transition()` updates the complete target critic with:

```text
target = (1 − tau) * target + tau * online
```

For one parameter, if the old target is 2, the new online value is 6, and tau is 0.25, the new target is 3. The default tau is much smaller, 0.005. It moves the target 0.5% of the distance toward the online value per learner update; it does not copy the online model every 200 steps.

This operation includes encoders and floating buffers. Integer buffers are copied because a fractional counter is not meaningful. No optimizer or autograd graph is involved. There is one target update per critic/policy/temperature update; delayed schedules are deferred.

### What would change for continuous SAC?

This learner enumerates discrete categories even though Drive accepts continuous control vectors. A continuous SAC actor would instead sample differentiable actions, usually `a = tanh(mu + sigma * noise)`. Its actor gradient must pass through the critic's action input, and its log probability needs the tanh change-of-variables correction. Those changes are outside the current implementation; the [continuous reference](reference_implementations/sac_continuous_action_cleanrl.py) shows them.

## 5. One complete training cycle

### Read the actual lifecycle

The run manager follows PPO's collection/update separation:

```python
pufferl.evaluate()              # recv -> preflight -> policy -> replay -> pending save -> send
maybe_logs = pufferl.train()    # traverse replay for update_epochs; logs follow PPO cadence
```

`evaluate()` directly receives from `self.vecenv` under `no_grad()`. It uses actual returned `env_id` values to identify pending actions and preflights eligibility and boundaries. On the initial call it appends valid transitions until replay reaches `batch_size`; later calls collect `ceil(replay_update_ratio * batch_size)` accepted valid transitions and replace preselected distinct replay indices. A call can end earlier at the received-row run budget. It samples the next actions, inserts or replaces completed transitions using the saved observations/actions, saves the newly chosen observations/actions into pending storage, and sends controls. If a receive crosses the remaining accepted-record quota, a uniform random subset fills that quota and other completed records are discarded; all returned rows are still matched, acted on and counted in `global_step`. Raw environment statistics and valid-insert counts are accumulated for bookkeeping. `train()` follows PPO's wrapper structure: set up profiling and loss accumulation, delegate to `_train_sac_transition()`, then finish profiling, advance the cycle, and handle logging/bookkeeping. If replay meets `learning_starts`, the helper makes `update_epochs` fresh shuffled passes over every current replay record in sequential minibatches, including the tail, and applies the same three optimizer steps and full target interpolation to each. A budget-limited partial refresh can still train once replay has reached the warmup threshold; below it, the learner makes no update.

```text
train(): tracking setup
    -> _train_sac_transition(): update_epochs full-replay passes
    -> train(): cycle bookkeeping; maybe emit logs and reset profile accumulators
```

This separation is the same one PPO uses between `train()` and `_train_ppo_transition()`. What changes inside the training helper is how data is selected and which losses are optimized.

Supplied online actor and critic retain the `.training` mode they had when passed to the trainer. The frozen target critic is a separate deep copy placed in evaluation mode. In PyTorch, `.train()` and `.eval()` select dropout and BatchNorm behavior; they do **not** enable or disable gradients. `torch.no_grad()` controls whether an operation builds a gradient graph. The learner therefore preserves the caller's online-model behavior and uses explicit gradient contexts for collection and target computation.

For each minibatch in each full-replay pass, `_train_sac_transition()` performs:

```text
sample minibatch from replay
    -> _compute_targets(rewards, next_obs, discounts)
    -> _critic_loss(obs, actions, targets)
    -> critic optimizer step
    -> _policy_loss(obs), using the updated online critics
    -> policy optimizer step
    -> temperature loss from detached entropy
    -> alpha optimizer step
    -> full Polyak interpolation of target critic
```

The critic target uses alpha at the beginning of this update. The actor uses that same alpha; temperature changes afterward. The next learner update sees the new alpha. Each forward graph is used for its own backward, and the actor Q forward is recomputed after changing the critic parameters.

### Understand replay refresh, full-replay epochs and the scheduler horizon

`batch_size=10,000` is the default logical replay capacity and initial-fill target. It matches `learning_starts=10,000`, the minimum occupancy for training; require `learning_starts <= batch_size`. If the received-row run budget ends before replay fills, `train()` updates only if replay has already reached that threshold. Later, `replay_update_ratio=0.1` makes a full refresh collect `ceil(0.1 * 10,000) = 1,000` valid transitions. The refresh preselects 1,000 distinct old indices uniformly; each accepted transition replaces one. On a receive that contains more eligible transitions than remain in the quota, the learner chooses a uniform random subset of the needed size and discards the extra completed records. It still matches every received row, chooses and sends next actions for every row, and counts all received rows in `global_step`. Collection stops at the accepted-record quota or the received-row budget, whichever comes first; the last complete vector receive may cross the run budget.

Every current replay record remains part of the training dataset, whether collected this cycle or retained from earlier cycles. For each `train()` call, `update_epochs=1` means one fresh shuffled permutation of all occupied replay indices. With 10,000 records and `minibatch_size=256`, this makes 40 optimizer batches: 39 batches of 256 and a final batch of 16. The final tail is processed. `sampled_transitions` increases by the actual number processed, 10,000 for this pass. Increasing `update_epochs` repeats the full traversal with a new permutation each time.

For a four-record example, let capacity and `learning_starts` both be four, `replay_update_ratio=0.5`, `minibatch_size=3`, and `update_epochs=2`. Initial `evaluate()` fills `[A,B,C,D]`; `train()` makes two permutations and applies batches of sizes `[3,1]` in each, processing every record twice. The next refresh preselects two distinct indices, say `[3,1]`, then overwrites those positions as new records arrive, leaving the other two old records in replay. The following `train()` still traverses all four records (old and new together), twice. Replacement controls replay age; the epoch permutation controls update order.

When cosine annealing is enabled, its horizon counts bounded training cycles, not received rows:

```text
cosine_T_max = max(1, ceil(total_timesteps / ceil(batch_size * replay_update_ratio)))
```

For example, a configured budget of 1,000,000 received rows with `batch_size=10,000` and ratio `0.1` gives a nominal cosine `T_max` of 1,000 cycles. This is an approximation: the budget counts every received slot row, while each refresh quota counts only eligible valid transitions. The outer run still stops when `global_step` reaches the received-row threshold; the final complete vector receive can cross it. Actor and critic scheduler state preserves the original `T_max` on resume, even when a later run budget changes. Each scheduler advances once per cycle that performs learner updates and reaches `eta_min` when its saved schedule horizon is complete. Alpha's learning rate stays fixed. The default disables cosine annealing.

### Which counter answers which question?

| Counter | Interpretation |
|---|---|
| `global_step` | Number of agent-slot rows returned by `vecenv.recv()`, including initial reset and masked rows, matching PPO's collection counter |
| `epoch` | Number of completed `train()` calls in the collection/training lifecycle |
| `valid_inserts` | Total transitions admitted to replay, including ones later overwritten; this is distinct from `global_step` |
| `learner_updates` | Number of completed replay minibatch updates; each advances all three optimizers and the target critic |
| `sampled_transitions` | Actual replay records processed by minibatch updates, including short tails and repeated epochs |

The log's `replay/size` is current occupancy, not cumulative inserts. `replay/sampled_transitions` is the saved actual-record counter, not `learner_updates * minibatch_size`; a short final minibatch contributes only the records it contains, and another epoch counts the records again.

The outer run stops when received `global_step` reaches `total_timesteps`; the received-row budget can exceed it only by the last full vector receive. `PuffeSAC.train()` returns `None` on cycles with no log emission. The outer run manager invokes callbacks only when the trainer emits logs, and the loss fields summarize the most recent collection/training cycle at that emission. Raw environment statistics accumulate across the interval and are reset only after they have been logged. Profiling names match PPO and feed its utilization, dashboard, uptime and SPS reports. Logging follows PPO's 0.25-second cadence, with a final-budget flush. The run manager always attempts environment/monitor cleanup if collection, optimization, logging or callbacks raise an exception; it closes its logger separately.

### Supplying components and restoring a learner

The configuration is an ordinary dictionary, as in `pufferl.py`. It requires `data_dir` and defaults `env` to `puffer_drive`; deterministic Torch behavior defaults off. Once compatible modules, replay and vecenv exist, the public component-supplied API is:

```python
from pufferlib.pufferl_sac import train

logs = train(
    args=dict(data_dir="/explicit/run/directory", action_count=12, total_timesteps=100_000),
    policy=policy, critic=critic, replay=replay, vecenv=vecenv,
)
```

This example supplies existing objects; it does not create them. `vecenv` uses PufferLib's standard vector methods and `evaluate()` calls them itself. The public scaffold does not expose a full Drive factory or Hydra path. `env_name` is passed by the outer run manager. `torch_deterministic` defaults to `false`; periodic checkpoint cadence defaults to 50 cycles, the final actor filename defaults to `final_model.pt`, and resume is opt-in through `resume_state_path` (default `None`). The component `train()` has no `checkpoint_path` or `resume_state_path` keyword and never resumes implicitly.

`save_checkpoint()` needs no path argument and returns the periodic actor artifact path under `models/model_<env>_<epoch>.pt`. Periodic actors use `models/`, the final actor is saved as `data_dir/final_model.pt`, and best actors use `best_models/`. Best-score updates are evaluated only when new logs are emitted for the current epoch. `trainer_state.pt` uses format 5 and stores the actor, online and target critics, temperature, three optimizer states, both scheduler states even when cosine annealing is disabled, counters including `sampled_transitions`, best score, and global RNG state. It stores no update credit; older state formats are rejected. It does not store replay, pending actions, simulator state, or Drive action-layout metadata. Restore into a fresh vecenv and empty replay, then refill before learning resumes. This restarts collection rather than continuing the saved simulator trajectory; [trace §5.4](trace_ppo_and_plan_sac.md#54-understand-exactly-what-gets-saved) gives the full checkpoint contract. `close()` closes the environment and monitor, saves the final actor, and returns its path; the outer `train()` closes the logger. Error cleanup closes resources without writing a partial final state.
