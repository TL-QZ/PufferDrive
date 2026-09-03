# How PufferDrive trains over a 128-step horizon

This tutorial follows one complete PufferDrive PPO update: collect 128 simulator
steps, turn the resulting trajectory into advantages and value targets, and then
replay the stored actions through the policy to compute gradients.

The most important conclusion is:

> In the current PufferDrive configuration, `bptt_horizon: 128` does **not** mean
> that the policy network backpropagates through 128 recurrent states. The
> configured policy is feed-forward (`rnn_name: null`). Time is preserved while
> computing advantages and returns, and then the 128-step rows are flattened
> into individual transitions for PPO training.

There is a true recurrent BPTT path in the code, but it is used only when an RNN
wrapper is enabled. This distinction prevents a common misreading of the buffer
name and shape.

## 1. The complete outer loop

The training loop repeatedly performs two phases:

```text
current policy parameters theta_old
             |
             v
  collect a fresh rollout with no gradients
  evaluate(): simulator <-> policy for 128 steps
             |
             v
  stored observations, actions, old log-probabilities,
  old values, rewards, terminals, and valid masks
             |
             v
  compute advantages A and value targets R
             |
             v
  train(): replay stored samples through the policy
             |
             v
  PPO actor loss + critic loss - entropy bonus
             |
             v
  backward(), gradient clipping, optimizer.step()
             |
             v
  updated parameters theta_new; collect a new rollout
```

The outer loop calls `evaluate()` and then `train()` on every iteration
([`pufferl.py`, lines 1590-1604](../../pufferlib/pufferl.py#L1590-L1604)). Despite
its name, `evaluate()` is the **training rollout collector**, not a standalone
benchmark evaluation. One outer “epoch” is therefore one newly collected
rollout followed by PPO updates on that rollout. It is not one simulator step,
and it is not one pass through a permanent dataset.

## 2. What is stored for a 128-step row?

Let:

- `H = 128` be the configured horizon;
- `S` be the number of trajectory segments on one training rank;
- `D` be the flat observation dimension;
- `A_shape` be the action tensor shape.

The rollout buffers are allocated as follows
([`pufferl.py`, lines 153-202](../../pufferlib/pufferl.py#L153-L202)):

| Buffer | Shape | Meaning at stored index `i` |
|---|---:|---|
| `observations` | `[S, 128, D]` | simulator observation received at index `i` |
| `actions` | `[S, 128, *A_shape]` | action sampled from the behavior policy at that observation |
| `logprobs` | `[S, 128]` | `log pi_old(a_i | o_i)` at collection time |
| `values` | `[S, 128]` | old critic estimate `V_old(o_i)` |
| `rewards` | `[S, 128]` | reward received with that observation, caused by the preceding action |
| `terminals` | `[S, 128]` | whether the preceding transition ended or truncated |
| `masks` | `[S, 128]` | whether this agent/sample is valid |

`batch_size = S * H`. When `batch_size: auto`, the code sets it to
`total_agents * bptt_horizon`, so normally `S = total_agents`
([`pufferl.py`, lines 153-164](../../pufferlib/pufferl.py#L153-L164)). Each row is
one agent's ordered temporal segment.

The buffer contains observations rather than a separate serialized “state”
object. The C simulator retains its full internal state; the learner stores the
policy-visible observation needed to recompute the actor and critic outputs.

## 3. What happens at one simulator time step?

For one agent, suppose action `a_(t-1)` has already been sent to the simulator.
The next collector iteration does this:

1. `vecenv.recv()` returns `o_t`, `r_t`, terminal/truncation flags, and a valid
   mask. Here `r_t` is the outcome of the **previous** action `a_(t-1)`
   ([`vector.py`, lines 513-531](../../pufferlib/vector.py#L513-L531)).
2. The observation is copied to the training device.
3. Under `torch.no_grad()`, the current policy performs one forward pass:

   ```text
   policy(o_t) -> action distribution pi_old(. | o_t), value V_old(o_t)
   ```

4. An action `a_t` is sampled from the distribution, and its log-probability
   `log pi_old(a_t | o_t)` is calculated
   ([`pytorch.py`, lines 193-246](../../pufferlib/pytorch.py#L193-L246)).
5. The collector stores `o_t`, `a_t`, its old log-probability, `V_old(o_t)`,
   `r_t`, the done flags, and the valid mask at the same buffer index
   ([`pufferl.py`, lines 404-453](../../pufferlib/pufferl.py#L404-L453)).
6. `vecenv.send(a_t)` gives the action to the simulator, which advances and
   eventually produces `(o_(t+1), r_(t+1), done_(t+1))`
   ([`vector.py`, lines 533-539](../../pufferlib/vector.py#L533-L539)).

In short:

```text
receive o_t and reward from a_(t-1)
       -> policy forward on o_t
       -> sample and store a_t
       -> send a_t to simulator
       -> receive its outcome on the next collector iteration
```

This rollout forward pass creates **no autograd graph**. It only generates the
on-policy data and records the actor and critic's old outputs.

### The feed-forward policy at time `t`

The active `Drive` policy sends `o_t` through actor and critic backbones. The
actor head produces an action distribution and the critic head produces one
scalar value estimate
([`ocean/torch.py`, lines 437-462](../../pufferlib/ocean/torch.py#L437-L462)).
With `rnn_name: null`, the output at time `t` has no neural hidden-state edge to
the output at `t-1`:

```text
o_t -> actor backbone  -> actor head  -> pi(. | o_t)
    -> critic backbone -> critic head -> V(o_t)
```

The observation can describe current velocity, nearby agents, lanes, traffic
controls, goals, and other current features, but the default network is not
given the preceding 127 observations as a sequence.

## 4. How the code aligns actions with next-step rewards

Because the vector environment returns a reward together with the next
observation, the stored arrays align like this:

| Stored index | Observation/action evaluated now | Reward stored now | Transition trained from this index |
|---:|---|---|---|
| `0` | `o_0`, `a_0`, `V(o_0)` | reward from before this segment | `a_0 -> r_1, o_1` |
| `1` | `o_1`, `a_1`, `V(o_1)` | `r_1`, caused by `a_0` | `a_1 -> r_2, o_2` |
| ... | ... | ... | ... |
| `126` | `o_126`, `a_126`, `V(o_126)` | `r_126` | `a_126 -> r_127, o_127` |
| `127` | `o_127`, `a_127`, `V(o_127)` | `r_127` | no next entry in this row |

The advantage kernel therefore iterates backward from stored index `126` to
`0`, explicitly using `reward[t + 1]`, `value[t + 1]`, and `done[t + 1]`
([`pufferlib.cpp`, lines 28-40](../../pufferlib/extensions/pufferlib.cpp#L28-L40)).
The current implementation leaves `advantage[127]` at zero. Thus, a buffer row
has 128 policy evaluations but 127 next-step temporal-difference calculations.
With the configured positive advantage-filter threshold, that zero-advantage
last entry will ordinarily not be selected for optimization.

## 5. Advantage computation: learning signal, not another network

There is no “advantage network” trained alongside the policy. Advantage is a
number computed from stored rewards and critic values. It answers:

> Was the chosen action followed by an outcome better or worse than the critic
> expected from this observation?

In the active non-recurrent path, the code first calls `_compute_advantages()`
with importance ratios and V-trace clips fixed to `1`
([`pufferl.py`, lines 711-719](../../pufferlib/pufferl.py#L711-L719)). For stored
index `t`, define:

```text
not_done_(t+1) = 1 - terminal_(t+1)

delta_t = reward_(t+1)
          + gamma * V_old(o_(t+1)) * not_done_(t+1)
          - V_old(o_t)

A_t = delta_t
      + gamma * lambda * not_done_(t+1) * A_(t+1)
```

The kernel starts with the future accumulator equal to zero and evaluates this
recursion from `t = 126` down to `t = 0`. This is GAE-style reverse-time credit
assignment. `gamma` controls reward discounting; `lambda` controls how much of
the later temporal-difference residuals flow backward within the 128-entry row.
A terminal cuts the recursion because `not_done_(t+1) = 0`.

The value-regression target is then:

```text
return_target_t = A_t + V_old(o_t)
```

This is created directly in `_compute_advantages()`
([`pufferl.py`, lines 625-643](../../pufferlib/pufferl.py#L625-L643)). The
reverse-time operation is implemented by a C/CUDA kernel and is not part of an
autograd graph. It computes targets; it does not itself update network weights.

For invalid samples, the mask is treated as a terminal boundary and the
advantage is set to zero. For truncations, the collector has a separate
value-bootstrap heuristic because Drive auto-resets before the learner can see
the true pre-reset next observation
([`pufferl.py`, lines 442-451](../../pufferlib/pufferl.py#L442-L451)).

## 6. From temporal rows to PPO samples

After advantages and returns have been computed on `[S, 128]`, the active
non-recurrent training path flattens all temporal dimensions:

```text
observations: [S, 128, D] -> [S * 128, D]
actions:      [S, 128, ...] -> [S * 128, ...]
logprobs:     [S, 128] -> [S * 128]
values:       [S, 128] -> [S * 128]
returns:      [S, 128] -> [S * 128]
advantages:   [S, 128] -> [S * 128]
```

This happens at
[`pufferl.py`, lines 721-730](../../pufferlib/pufferl.py#L721-L730). From this
point onward, a sample from time `t` and a sample from time `t+1` can be placed
in unrelated shuffled minibatches. Their temporal relationship has already
done its job by determining `A_t` and `return_target_t`.

The code then optionally removes low-signal transitions. With the current
default settings, it maintains an exponential moving average of the largest
absolute valid advantage and keeps transitions satisfying:

```text
abs(A_t) >= adv_filter_threshold_scale * EMA(max(abs(A)))
```

The default threshold scale is `0.01`
([`puffer_drive.yaml`, lines 263-268](../../pufferlib/config/puffer_drive.yaml#L263-L268)),
and the selection logic is at
[`pufferl.py`, lines 731-764](../../pufferlib/pufferl.py#L731-L764).

## 7. How the policy is “rolled out” again during gradient training

The simulator is **not** rerun inside the PPO update. Instead, the stored
observations and stored actions are replayed through the current network.

For each shuffled minibatch, `_ppo_loss()` performs a new forward pass with
gradients enabled:

```text
stored o_t -> current actor  -> new pi_theta(. | o_t)
           -> current critic -> new V_theta(o_t)

stored a_t + new distribution -> new log pi_theta(a_t | o_t)
```

It does not sample a replacement action for the loss. Passing the stored action
to `sample_logits()` asks what probability the **current** policy assigns to the
action that the behavior policy actually used
([`pytorch.py`, lines 221-234](../../pufferlib/pytorch.py#L221-L234)).

This second forward pass is at
[`pufferl.py`, lines 556-568](../../pufferlib/pufferl.py#L556-L568). It is the
forward pass whose autograd graph is retained and later backpropagated.

### Actor update

The stored old log-probability lets PPO compare the new policy with the policy
that collected the transition without keeping a second frozen model in memory:

```text
ratio_t = exp(
    new_logprob_theta(a_t | o_t)
    - old_logprob(a_t | o_t)
)
```

After normalizing advantages over the minibatch (and across DDP ranks), the
actor minimizes the worse of the unclipped and clipped surrogate losses:

```text
policy_loss_t = max(
    -A_normalized_t * ratio_t,
    -A_normalized_t * clip(ratio_t, 1 - epsilon, 1 + epsilon)
)
```

The implementation is at
[`pufferl.py`, lines 567-598](../../pufferlib/pufferl.py#L567-L598). A positive
advantage raises the probability of the stored action; a negative advantage
lowers it. PPO clipping limits how far the probability ratio can profitably move
in one update.

### Critic update

The critic learns to predict the return target computed from the rollout:

```text
value_loss_t = 0.5 * (V_theta(o_t) - return_target_t)^2
```

Value clipping is used only if `vf_clip_coef` is non-null; it is null in the
base PufferDrive config. The critic implementation is at
[`pufferl.py`, lines 600-607](../../pufferlib/pufferl.py#L600-L607).

### One combined backward pass

Actor and critic are not updated in two isolated training phases. Their losses
are combined with the entropy bonus:

```text
total_loss = policy_loss
             + vf_coef * value_loss
             - ent_coef * entropy
```

The entropy term rewards a less collapsed action distribution. One
`backward()` call differentiates this combined loss through the relevant actor
and critic parameters
([`pufferl.py`, lines 608-623](../../pufferlib/pufferl.py#L608-L623)). Because the
current Drive configuration has `shared_network: false`, actor and critic have
separate backbones, although they still participate in the same optimizer and
loss call
([`ocean/torch.py`, lines 395-435](../../pufferlib/ocean/torch.py#L395-L435)).

## 8. What happens across the three PPO passes?

The default `update_epochs: 3` means that the retained rollout samples are
reshuffled and replayed through the policy three times
([`pufferl.py`, lines 783-806](../../pufferlib/pufferl.py#L783-L806)). It does not
mean that the simulator collects three 128-step trajectories before updating.

For every pass:

1. Randomly permute the retained transition indices.
2. Slice a physical minibatch.
3. Run the current policy and critic on its stored observations.
4. Recompute new log-probabilities for its stored actions.
5. Compute actor, critic, entropy, and total losses.
6. Backpropagate.
7. Once the configured number of physical minibatches has accumulated, clip
   the gradient norm and call `optimizer.step()`.

After an optimizer step, later minibatches and later PPO passes use the newly
updated parameters. The rollout's stored old log-probabilities, old values,
advantages, and return targets remain fixed in the active feed-forward path.
The rollout is discarded/overwritten when the next outer epoch collects fresh
experience.

`minibatch_size` is the desired logical optimizer batch, while
`max_minibatch_size` caps the physical forward/backward microbatch. For example,
if these are `128,000` and `32,000`, four 32,000-transition backward passes are
divided by four and accumulated before one optimizer step. Gradient accumulation
is configured at
[`pufferl.py`, lines 219-237](../../pufferlib/pufferl.py#L219-L237) and executed at
[`pufferl.py`, lines 812-826](../../pufferlib/pufferl.py#L812-L826).

## 9. Concrete scale for the third-run two-GPU baseline

The third-run launcher/config gives a useful concrete example:

- 20 simulator environments per rank;
- 3,200 agents per environment;
- 128 stored time indices per agent;
- 2 DDP ranks/GPUs;
- logical minibatch 128,000 transitions per rank;
- physical microbatch 32,000 transitions per rank;
- 3 PPO passes.

These values come from the
[`nightly_best.yaml`](../third_run_nightly_best_config/override_config/nightly_best.yaml#L9-L24)
and
[`launch_2gpu.sh`](../third_run_nightly_best_config/train/launch_2gpu.sh#L57-L70).
They imply:

```text
agents per rank        = 20 * 3,200 = 64,000
buffer rows per rank   = 64,000
stored entries/rank    = 64,000 * 128 = 8,192,000
stored entries/global  = 2 * 8,192,000 = 16,384,000 per outer epoch
```

`vec.batch_size: 10` means each simulator/inference exchange covers ten of the
twenty environments, or `10 * 3,200 = 32,000` agents per rank. It controls how
environment work is scheduled; it is not the PPO optimizer minibatch. The
vector backend computes `agents_per_batch` separately from total agents
([`vector.py`, lines 325-347](../../pufferlib/vector.py#L325-L347)).

Advantage filtering means the exact retained transition count and optimizer-step
count can be smaller than the nominal count inferred from all 8,192,000 stored
entries. On DDP, ranks trim to the same retained count so that their update loops
remain synchronized
([`pufferl.py`, lines 745-756](../../pufferlib/pufferl.py#L745-L756)).

## 10. When would this be true BPTT?

The code sets `use_rnn` solely from whether `rnn_name` is non-null
([`pufferl.py`, lines 2438-2445](../../pufferlib/pufferl.py#L2438-L2445)). If
`rnn_name: Recurrent` is selected, the base Drive model is wrapped by
`LSTMWrapper` during policy construction
([`pufferl.py`, lines 2367-2381](../../pufferlib/pufferl.py#L2367-L2381)). Then:

- rollout inference uses an `LSTMCell` one simulator step at a time and carries
  hidden/cell state between steps;
- training selects whole `[B, 128, D]` trajectory rows rather than flattening
  them;
- the wrapper runs an `nn.LSTM` over the time dimension;
- autograd backpropagates through the 128 recurrent computations in that
  selected segment;
- final hidden/cell states are detached at the segment boundary, which truncates
  the gradient there.

The recurrent wrapper implements the step-wise and sequence-wise paths at
[`models.py`, lines 128-195](../../pufferlib/models.py#L128-L195), and
`train()` selects the trajectory PPO implementation at
[`pufferl.py`, lines 493-503](../../pufferlib/pufferl.py#L493-L503).

That is genuine **truncated backpropagation through time**. It is not what the
current PufferDrive baseline runs: the base config explicitly has
`rnn_name: null`
([`puffer_drive.yaml`, lines 25-30](../../pufferlib/config/puffer_drive.yaml#L25-L30)).
Also, the Drive model requires `shared_network: true` when used through this
LSTM wrapper
([`ocean/torch.py`, lines 473-476](../../pufferlib/ocean/torch.py#L473-L476)); the
current base configuration instead uses `shared_network: false`.

## 11. One transition, end to end

Putting everything together for stored transition `t`:

```text
ROLLOUT (no gradients)
  o_t
    -> old actor -> sample a_t; store old_logprob_t
    -> old critic -> store old_value_t
    -> simulator(a_t)
    -> receive reward_(t+1), o_(t+1), done_(t+1)

TARGET CONSTRUCTION (reverse over the 128-step row, no autograd)
  delta_t = reward_(t+1)
            + gamma * old_value_(t+1) * (1 - done_(t+1))
            - old_value_t
  advantage_t = delta_t
                + gamma * lambda * (1 - done_(t+1)) * advantage_(t+1)
  return_target_t = advantage_t + old_value_t

PPO REPLAY (gradients enabled)
  stored o_t -> current actor and critic
  stored a_t -> new_logprob_t
  ratio_t = exp(new_logprob_t - old_logprob_t)

  actor:  use advantage_t and clipped ratio_t
  critic: regress new_value_t toward return_target_t
  both:   combine losses, backward, clip gradients, optimizer.step()
```

So the 128-step horizon matters in the current baseline because it bounds how
far reward/TD information can propagate backward while constructing advantages.
It does **not** make the feed-forward policy consume 128 observations, and it
does **not** create an actor autograd graph across simulator time. The policy is
rolled out once against the simulator to create data, then evaluated again on
stored observations/actions to perform PPO learning.
