# PPO: critic estimates, advantages, and losses

**`mb_adv` combines observed rewards with critic estimates to train the policy.**
The critic learns to predict `mb_returns`; `mb_values` is a saved prediction,
not the critic's target.

| Variable | Meaning | Role |
| --- | --- | --- |
| `mb_values` | Saved critic estimates for sampled states | Baseline for advantages and value clipping |
| `mb_returns` | Raw advantages plus saved values | Critic training targets |
| `mb_adv` | Sampled advantage estimates, normalized inside the loss | Policy learning signal |
| `newvalue` | Current critic predictions during training | Predictions compared with `mb_returns` |
| `ratio` | Current action probability divided by rollout action probability | Differentiable policy term |

## 1. Collect rewards and critic estimates

In [`evaluate()`](../../pufferlib/pufferl.py#L405), rollout inference runs under
`torch.no_grad()`. The network predicts a value for the current observation,
and the collector stores rewards, values, actions, log-probabilities, and
episode boundary flags:

```python
self.rewards[batch_rows, l] = r
self.values[batch_rows, l] = value.flatten().float()
```

`V(s)` estimates discounted future reward from state `s`. It does not directly
estimate the advantage of the sampled action.

The stored reward may be transformed by `normalize_rewards`. If
`use_value_bootstrapping` is enabled, truncation rewards can also receive a
bootstrap adjustment. Because Drive auto-resets, the collector uses the previous
stored value as a heuristic for the unavailable terminal-state value
([collector code](../../pufferlib/pufferl.py#L430)).

## 2. Compute advantages backward through the rollout

[`_compute_advantages()`](../../pufferlib/pufferl.py#L625) passes stored values,
rewards, boundary flags, and importance ratios to `compute_puff_advantage()`.
That wrapper calls the [CUDA kernel](../../pufferlib/extensions/cuda/pufferlib.cu#L7)
or the equivalent [CPU implementation](../../pufferlib/extensions/pufferlib.cpp#L28).

For the non-recurrent path, importance ratios and both clipping limits are `1`.
The kernel therefore reduces to generalized advantage estimation (GAE):

```python
# Conceptual equivalent of the kernel, for one rollout row.
advantage = zeros_like(values)
next_advantage = 0
for t in reversed(range(horizon - 1)):
    nonterminal = 1 - dones[t + 1]
    delta = rewards[t + 1] + gamma * values[t + 1] * nonterminal - values[t]
    advantage[t] = delta + gamma * gae_lambda * nonterminal * next_advantage
    next_advantage = advantage[t]
```

`delta` is a one-step prediction error:

```text
delta = observed reward + discounted next-state estimate - current-state estimate
```

For example, with reward `2`, current value `5`, next value `4`, and
`gamma = 0.99`, a nonterminal transition gives `delta = 2 + 0.99 * 4 - 5 = 0.96`.
The observed outcome was better than expected.

GAE combines this error with later errors. Without an intervening boundary:

```text
A[t] = delta[t] + (gamma * lambda) * delta[t+1]
                  + (gamma * lambda)^2 * delta[t+2] + ...
```

`gamma` discounts future rewards; `gae_lambda` controls how much later prediction
errors contribute. Lower lambda relies more on the one-step estimate; higher
lambda includes more of the observed future trajectory.

Implementation boundaries:

- The reward arriving with observation `t + 1` belongs to the transition from
  state `t`, so the kernel reads `rewards[t + 1]`.
- `dones[t + 1]` stops both next-value bootstrapping and advantage recursion at
  an episode boundary. The collector combines termination and truncation flags.
- Invalid masks are treated as boundaries; their output advantages are zeroed.
- The loop starts at `horizon - 2`. The final slot's advantage remains zero;
  its stored value supplies the preceding transition's bootstrap estimate.

## 3. Build targets and select a minibatch

The advantage helper returns:

```python
returns = advantages + self.values
```

In [`_train_ppo_transition()`](../../pufferlib/pufferl.py#L711), these arrays are
flattened and sampled into minibatches:

```python
mb_values = flat_values[mb_idx]
mb_returns = flat_returns[mb_idx]
mb_adv = flat_advantages[mb_idx]
```

Advantages and return targets are computed once before this path's PPO update
epochs. Advantage filtering, when enabled, affects which transitions are kept.

The [recurrent trajectory path](../../pufferlib/pufferl.py#L645) differs: it
recomputes advantages between minibatches and updates stored values and ratios
using detached training outputs. Its kernel uses V-trace correction factors:

```python
rho = min(importance_ratio[t], rho_clip)
c = min(importance_ratio[t], c_clip)
delta = rho * (reward_next + gamma * value_next * nonterminal - value)
advantage = delta + gamma * gae_lambda * c * next_advantage * nonterminal
```

Thus, saved values remain rollout predictions throughout the non-recurrent
update, but can be refreshed during the recurrent update.

## 4. Use advantages to train the policy

[`_ppo_loss()`](../../pufferlib/pufferl.py#L556) evaluates stored observations and
actions with the current network. It computes the probability ratio and
normalizes advantages under `torch.no_grad()`:

```python
ratio = (newlogprob - mb_logprobs).exp()
mb_adv = (mb_adv - advantage_mean) / (advantage_std + 1e-8)
pg_loss1 = -mb_adv * ratio
pg_loss2 = -mb_adv * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
pg_loss = torch.max(pg_loss1, pg_loss2).mean()
```

`pg_loss` means policy gradient loss. Positive normalized advantages encourage
higher action probabilities; negative ones encourage lower probabilities,
subject to PPO clipping. Centering can change an individual advantage's sign.
Distributed training computes normalization statistics across ranks; the
recurrent path also applies sampling weights after normalization.

**Gradients flow through `ratio`, not through the advantage calculation.** The
critic helps construct the policy's learning signal, but `pg_loss` does not
backpropagate through those saved critic estimates. Shared network layers can
still receive gradients from both policy and value losses.

## 5. Train the critic against return targets

Without value clipping, the [value loss](../../pufferlib/pufferl.py#L600) is:

```python
v_loss = 0.5 * ((newvalue - mb_returns) ** 2).mean()
```

With value clipping:

```python
v_clipped = mb_values + torch.clamp(
    newvalue - mb_values, -vf_clip_coef, vf_clip_coef
)
v_loss_unclipped = (newvalue - mb_returns) ** 2
v_loss_clipped = (v_clipped - mb_returns) ** 2
v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
```

`mb_values` anchors the clipped prediction; `mb_returns` remains the target.
Taking the larger error limits the benefit of some large prediction changes.
It does not directly clamp the network's `newvalue` output.

The final objective combines policy learning, critic learning, and entropy:

```python
loss = pg_loss + vf_coef * v_loss - ent_coef * entropy_loss
```

Next reading: open the [two kernel equations](../../pufferlib/extensions/cuda/pufferlib.cu#L16)
and identify the prediction error followed by the backward advantage recursion.
For the broader rollout/update flow, see [the training tutorial](bptt_training_step.md).
