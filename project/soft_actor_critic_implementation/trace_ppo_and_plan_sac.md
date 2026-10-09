# Trace PPO training from start to finish, then map each step to SAC

**Start with the two `train` functions:** module-level `train(...)` owns the run; `PuffeRL.train()` updates the model. `PuffeRL.evaluate()` collects training experience despite its name.

This guide follows execution order for feed-forward Drive (`rnn_name: null`). The five phases and numbered subsection identifiers remain stable for [your notes](trace_ppo_and_plan_sac_note.md). The [implementation plan](planning.md) records decisions and remaining work; the [five-lesson tutorial](SAC_implementation_tutorial.md) explains the SAC mechanisms without repeating them here. PPO source was rechecked on 2026-10-07; original trace snapshot: `b1a12f64`.

SAC implements a feed-forward discrete learner with a PPO-aligned run lifecycle. `PuffeSAC` is independent of `PuffeRL` and reuses common PPO utilities and conventions; the public trainer still receives its vecenv, actor, critic, and replay as supplied components. Drive construction and final-observation transport remain outside this integration. Collection assumes Drive-style masks and same-receive auto-reset observations.

```text
PPO: main -> train -> load env/policy -> PuffeRL
       repeat: evaluate -> train -> logs/checkpoints/hooks
SAC: main -> NotImplementedError until Drive construction integration exists
     supplied train(vecenv, policy, critic, replay) -> PuffeSAC
       repeat: evaluate [recv -> mask/boundary preflight -> policy -> replay insert
                         -> pending save -> send]
              -> train (full-replay minibatch epochs) -> periodic logs/hooks
       actor and format-5 learner checkpoints -> close env/monitor, outer closes logger
```

## 1. Start the run

### 1.1 `main()` chooses training

PPO [`main`](../../pufferlib/pufferl.py#L2449) removes the command/environment arguments and dispatches its own run manager. `puffer train puffer_drive` reaches it via the console entry.

SAC has a separate `main()` and run manager; it never dispatches PPO training. `python -m pufferlib.pufferl_sac train puffer_drive` still raises an actionable `NotImplementedError` because there is no Drive construction factory. The SAC run manager propagates `env_name`, but the public scaffold has no Hydra loader or Drive model/replay factory. Supply a compatible vecenv, policy, critic and replay to `train(...)` to exercise the learner pipeline. Required `data_dir` and default `env: puffer_drive` are part of the flat SAC configuration.

### 1.2 `train()` resolves configuration and the evaluation schedule

PPO [`load_config`](../../pufferlib/pufferl.py#L2409) composes Hydra settings and validates the environment schema; outer [`train`](../../pufferlib/pufferl.py#L1432) owns the run and evaluation schedule. SAC accepts a flat config with required `data_dir`, `action_count`, and `total_timesteps`; `DEFAULT_CONFIG` supplies learner and PPO-alignment defaults, including `torch_deterministic: false`, `checkpoint_interval: 50`, `final_model_name: final_model.pt`, and `resume_state_path: null`. SAC has no Hydra loader. The outer loop stops when `global_step >= total_timesteps`. Separately, if cosine annealing is enabled, nominal `T_max = max(1, ceil(total_timesteps / ceil(batch_size * replay_update_ratio)))` estimates its cycle duration from the valid-record refresh quota; scheduler state retains that original horizon on resume even if the new budget differs. `train()` accepts neither `checkpoint_path` nor `resume_state_path` keywords. Replay-refresh and scheduler units are explained in [tutorial lesson 5](SAC_implementation_tutorial.md#5-one-complete-training-cycle).

### 1.3 If loading a model, restore its architecture before constructing it

PPO can read neighboring `config.yaml` to reconstruct policy settings before loading weights. SAC receives constructed modules; `load_training_state()` strictly loads their state dictionaries and checks learner configuration. It does not construct architectures or convert a PPO scalar value head to action-value heads.

### 1.4 Set rank, seeds, environment and model

PPO [`derive_rank_seeds`](../../pufferlib/pufferl.py#L1422), [`load_env`](../../pufferlib/pufferl.py#L2356) and [`load_policy`](../../pufferlib/pufferl.py#L2367) construct the seeded run. SAC's constructor follows PPO's concept blocks and common attributes for config, seeds, device/determinism, modules, optimizers, counters and monitoring. It reuses `torch_device` and `derive_rank_seeds` for one rank, then seeds global Python, NumPy and Torch streams; callers seed model construction and replay-private RNGs before supplying them. `policy(obs)` returns logits `[B,K]`; `critic(obs)` returns two Q tensors `[B,K]`; the target critic is a frozen deep copy in evaluation mode.

### 1.5 Wrap distributed training and choose the logger

SAC runs single-process. `NoLogger` is the default; injected loggers implement the PPO logger contract. The trainer closes its environment and monitor and saves the final actor; the outer run manager owns logger shutdown.

## 2. Construct and restore the trainer

### 2.1 Configure numerical execution and start the initial reset

PPO [`PuffeRL.__init__`](../../pufferlib/pufferl.py#L124) sets numerical options, allocates rollout storage, starts asynchronous reset, and distinguishes policy action IDs from environment controls. SAC also uses FP32: it allocates per-slot observation/action/pending arrays before calling `async_reset()`. Supplied online actor and critic preserve their incoming `.training` mode; the target critic is frozen and put in evaluation mode. The caller supplies feed-forward modules with disjoint actor and critic parameters.

### 2.2 Allocate PPO rollout storage

PPO allocates `[segments, horizon, ...]` observations/actions alongside rewards, values, old log probabilities, boundary flags and masks. `ep_indices`/`ep_lengths` connect slots to storage rows. A horizon is not evidence of BPTT with `rnn_name: null`.

SAC keeps `self.observations`, `self.actions` and `self.pending` indexed by actual agent-slot ID, retaining them across collection and learning calls. Replay uses `add(obs, actions, rewards, next_obs, discounts)` to append copied initial batches, `replace(indices, obs, actions, rewards, next_obs, discounts)` to copy over exact selected records, and `get_batch(indices)` to retrieve exact records. Index vectors are CPU `int64`; the replay factory and physical storage are still TODO. No old values, old log probabilities, advantages or PPO episode-row metadata are needed. Indexed replacement and full-replay training are explained in [tutorial lesson 2](SAC_implementation_tutorial.md#2-one-transition-through-replay).

### 2.3 Configure minibatches, compilation, optimizer and schedule

PPO configures rollout minibatches, AdamW, optional compilation, gradient accumulation and scheduling around its collection batch.

SAC has separate Adam optimizers for actor, joint critics and log-alpha. The actor optimizer is `self.optimizer`, matching PPO's common attribute. `batch_size=10,000` is the default logical replay capacity and initial-fill target; `learning_starts=10,000` requires replay occupancy at least that large. Later collection accepts `ceil(replay_update_ratio * batch_size)` valid records per refresh (`replay_update_ratio=0.1`), replacing distinct uniformly selected old indices. `minibatch_size=256` is the maximum records consumed by one learner update; the tail minibatch may be smaller. Defaults include `update_epochs=1` and `gamma=0.99`. Each train call shuffles all current replay indices once per epoch and processes the entire old/new mixture without replacement within that pass. Optional cosine annealing covers actor and critic learning rates, defaults off, and advances once per cycle that performs learner updates. The alpha learning rate remains fixed.

### 2.4 Initialize monitoring and optional recurrence

PPO initializes `epoch`, record counters, profiling and optional recurrent state. SAC reuses PPO's `Profile`, `Utilization`, dashboard, uptime and SPS reporting. `global_step` counts received rows, including initial-reset and masked rows, while `valid_inserts` counts records admitted to replay. `epoch`, `learner_updates` and `sampled_transitions` have separate units; sampled transitions counts actual records processed, including minibatch tails and repeated epochs.

### 2.5 Restore training state, then save the run configuration

PPO [`load_training_state`](../../pufferlib/pufferl.py#L329) can restore optimizer state and counters. SAC resumes only when `resume_state_path` is explicitly set in config; there is no automatic resume and no resume-path or checkpoint-path keyword on `train()`. Restore learner state into supplied compatible modules, a freshly reset vecenv, cleared pending flags, and empty replay. Replay must refill before learning resumes. See §5.4 for checkpoint contents and limits.

## 3. Collect experience

### 3.1 The outer loop calls `evaluate()` before `train()`

PPO runs a full rollout collection before its optimizer phase. Its `evaluate` name does not mean benchmark evaluation.

SAC follows the same split. The initial `evaluate()` appends valid transitions until replay reaches its `batch_size` capacity or the received-row budget ends. Later calls collect `ceil(replay_update_ratio * batch_size)` accepted valid transitions and overwrite distinct preselected replay indices, unless the run budget ends first. If the last receive contains more eligible records than remain in the quota, a uniform random subset fills it; all rows still have their preceding outcomes matched and receive next actions. `global_step` counts all received rows and collection checks the remaining run budget after every receive, so only the final complete receive can cross `total_timesteps`. Then `train()` makes `update_epochs` shuffled passes over the entire replay dataset, with sequential minibatches and an included tail. Episodes need not finish. When enabled, actor/critic cosine schedulers use the nominal horizon `T_max = max(1, ceil(total_timesteps / ceil(batch_size * replay_update_ratio)))`; it approximates cycles because the run budget counts received rows while refresh quotas count valid transitions. Their saved scheduler state preserves that original `T_max` on resume, independently of a changed run budget.

### 3.2 Receive observations and the preceding action's outcome

PPO [`evaluate`](../../pufferlib/pufferl.py#L378) receives:

```text
observation, reward, terminated, truncated, info, env_id, mask = vecenv.recv()
```

The reward and flags describe the preceding action; the observation may already be the reset observation. The first reset receive has no executed action. PPO's fast path treats received IDs as a contiguous slice and counts every received slot.

SAC counts `len(env_id)` on every receive, including reset and masked rows, but indexes pending state by the **actual IDs**, which may be reordered or noncontiguous. `self.pending[id]` distinguishes a preceding send from the initial reset and is set on every send. The receive mask decides whether that preceding action enters replay; it does not determine whether the current observation can be acted on. Collection assumes the receive supplies the new episode observation under Drive's mask convention.

### 3.3 Run the policy without gradients and choose an action

PPO [`Drive.forward`](../../pufferlib/ocean/torch.py#L437) produces actor logits plus scalar value estimates. [`sample_logits`](../../pufferlib/pytorch.py#L193) samples saved category IDs; Drive maps these to simulator controls. The default jerk actor has 12 classes, despite the environment's continuous control interface.

SAC samples from `policy(o_device)` logits under `no_grad()`, retains discrete joint-action IDs in `self.actions[env_id]`, maps them to environment controls and sends them. Replay learning recomputes probabilities, so it stores no PPO old log probability, entropy or scalar value. Controls keep their batch axis, including shape `[1, control_dim]` for one row. Warmup is a replay-fill threshold, not a random-action phase or reward transformation.

### 3.4 Write the rollout row: understand the shifted layout

PPO [`rollout writes`](../../pufferlib/pufferl.py#L425) save `observation[j]` and newly chosen `action[j]` together with the outcome of `action[j-1]`. Therefore `reward[j+1]` belongs to `action[j]`. The advantage kernel compensates for this shift. Storage rows may cross resets.

SAC holds the pending observation/action by agent ID. If slot 7 saved `(o17, a17)` and slot 2 saved `(o40, a40)`, a receive for IDs `[7, 2]` with rewards `[r18, r41]` forms `(o17, a17, r18, next_obs_7, d18)` and `(o40, a40, r41, next_obs_2, d41)`. It inserts nothing on the initial reset (`pending=False`) and never shifts a reward onto the new action. Adjacent replay indices need not be successors.

Collection order is `recv -> mask/boundary preflight -> policy sample -> replay.add(previous records) or replay.replace(selected indices, previous records) -> save new pending arrays -> send`. Replay receives complete copied batches `(obs, actions, rewards, next_obs, discounts)`, so later sends cannot overwrite stored experience. PPO's temporal layout could also store SAC experience; complete records make indexed replay access straightforward. SAC's losses do not need the stored values, log probabilities, advantages or episode-row metadata.

### 3.5 Handle masks and reset boundaries explicitly

PPO [`truncation handling`](../../pufferlib/pufferl.py#L442), Drive [`c_step`](../../pufferlib/ocean/drive/drive.h#L4497), and [`Drive.step`](../../pufferlib/ocean/drive/drive.py#L655) expose the boundary issue. Drive masks are computed before movement and describe the just-completed action's eligibility; a valid action can terminate while its receive mask remains true. Keep this mask separate from `self.pending` and from whether the returned observation is suitable for a new policy action.

| Received outcome | Replay discount | `next_obs` for that record |
|---|---:|---|
| Ordinary eligible action | `gamma` (`0.99` default) | Returned observation |
| Eligible true terminal, including stopped/removed | `0` | Finite placeholder copied from saved pending observation; never the reset observation |
| Eligible pure truncation that interrupts an active agent | `gamma` for continuing task | Requires final pre-reset observation, which current Drive transport does not supply |
| Both terminated and truncated | `0` | Terminal rule wins; reset observation remains for the next action |

The replay record carries the computed discount, not separate raw flags. Current SAC preflights the full eligible batch and raises `NotImplementedError` before `replay.add()` if it contains a pure eligible truncation, since there is no final-state value to store. It does not substitute the reset observation or silently assign zero discount. `terminate_on_goal=0` is the initial continuing-driving configuration. If a future run sets `terminate_on_goal=1`, a future Drive bridge must capture enough reset-reason information to identify final-goal success as a true terminal. This bridge is TODO; no metadata transport contract is implemented.

PPO bootstraps some truncated rewards from the preceding stored value and then stops recursion; SAC requires the true final observation for a continuing truncation. The boundary bridge in [planning section 4](planning.md#4-replay-collection-is-the-first-correctness-gate) is still Drive integration work.

### 3.6 Aggregate metrics, send actions and finish collection

PPO sends controls after each receive, including its final stored column; outcomes arrive later. SAC also sends after every receive. The final send remains pending through `train()` and is matched at the next `evaluate()`. Info becomes raw scalar `environment/` statistics, which accumulate until a log emission; `global_step` counts received slots and `valid_inserts` counts eligible replay rows.

## 4. Update the model

### 4.1 `PuffeRL.train()` selects the update path

PPO [`train`](../../pufferlib/pufferl.py#L493) is an orchestration wrapper. SAC now uses the same division of responsibility: `train()` sets up and delegates work, while `_train_sac_transition()` owns replay learning.

| Phase | `PuffeRL.train()` | `PuffeSAC.train()` |
|---|---|---|
| Before learning | Start PPO-named profiling; create `losses` | Same profiling names and setup |
| Delegate learning | `_train_ppo_transition(losses, profile, epoch)` for feed-forward PPO | `_train_sac_transition()` makes shuffled full-replay passes in sequential minibatches |
| After learning | Finish profiling, advance epoch, log and run configured operational hooks | Finish profiling, advance epoch, keep latest-cycle losses, emit on PPO cadence, reset profile data after logging |

`_train_sac_transition()` checks replay warmup, creates one fresh shuffled permutation of all `len(replay)` indices per `update_epochs` pass, and processes sequential minibatches including the tail. For each minibatch it fetches the exact indexed records, recomputes targets, performs critic/actor/temperature optimization, applies full Polyak interpolation, advances counters and averages update metrics into `losses`. `sampled_transitions` increases by the actual minibatch length. The only math helpers are `_compute_targets()`, `_critic_loss()`, and `_policy_loss()`. There is no recurrent branch.

The SAC run manager reuses PPO-style actor/critic cosine scheduling, timed logging/dashboard, checkpoint artifacts and hooks. Cosine annealing is optional and disabled by default; when enabled it advances once per cycle that performs learner updates. Alpha's learning rate stays fixed. Each scheduler's nominal `T_max` is `max(1, ceil(total_timesteps / ceil(batch_size * replay_update_ratio)))` cycles, estimated from the accepted-record refresh quota; the estimate differs from the received-row stop budget when masks make some rows ineligible. Scheduler state preserves that horizon across resume; after the schedule reaches `eta_min`, it remains there. The outer run's independent stop condition remains `global_step >= total_timesteps`.

### 4.2 Compute advantages and value targets once

PPO's feed-forward path computes advantages before optimizer passes. Stored ratios begin at one, reducing its advantage recurrence to GAE under the corresponding clipping settings. Collection-time values provide the bootstrap/return baseline; this computation understands shifted rewards.

SAC replaces this with detached `_compute_targets()`. Its current actor supplies next-state probabilities; frozen target critics supply Q estimates. [Tutorial lesson 3](SAC_implementation_tutorial.md#3-one-critic-update) gives the formula, shapes and numerical example. The PPO advantage kernel is not used.

### 4.3 Flatten, mask and optionally filter the training data

PPO flattens rollout rows and can filter by advantage. SAC decides eligibility on insertion or indexed replacement; once `len(replay) >= learning_starts`, every train call processes the complete current replay dataset, including both retained old and newly inserted records. Each epoch traverses one fresh shuffled permutation without replacement, split into sequential minibatches.

### 4.4 Build each minibatch and define every tensor

| PPO tensor | Meaning |
|---|---|
| `mb_values` | Collection-time scalar predictions on the feed-forward path |
| `mb_returns` | Detached regression targets formed from advantages plus stored values |
| `mb_adv` | Pre-update advantage estimates, optionally normalized for policy loss |
| `old_logprob` | Log probability of the stored action under its collection policy |
| `newvalue` | Current scalar prediction reevaluated during optimization |

`_train_sac_transition()` passes a CPU `int64` index vector to `replay.get_batch(indices)`, unpacks exact records into `mb_obs`, `mb_actions`, `mb_rewards`, `mb_next_obs` and `mb_discounts`, then transfers them to the device. Initial `add(...)` and later `replace(indices, ...)` copy all five fields; indexed replacement overwrites only the requested replay slots. The replay insertion boundary establishes the tensor contract: FP32 observations/rewards/discounts, int64 action IDs, scalar fields exactly `[B]`, valid action IDs and discounts in `[0,1]`. `discounts` already includes gamma. The learner trusts this internal contract and does not repeatedly validate model outputs. A final minibatch may have fewer than `minibatch_size` rows, but its leading tensor dimension remains consistent across all fields.

### 4.5 `_ppo_loss`: reevaluate the saved action

PPO forms the ratio `exp(new_logprob - old_logprob)` for its clipped policy objective. SAC `_critic_loss()` gathers online Qs at the stored action; `_policy_loss()` evaluates the current full categorical distribution. There is no behavior-policy ratio.

### 4.6 `_ppo_loss`: normalize advantages and combine three objectives

PPO combines clipped policy loss, value regression and entropy regularization. SAC uses separately owned objectives. Exact action sums, gradient detachment and alpha's adjustment direction are in [tutorial lesson 4](SAC_implementation_tutorial.md#4-actor-temperature-and-targets). This implementation uses action sum then batch mean, avoiding the extra `1/K` scaling in the supplied Atari example's actor loss.

### 4.7 Backward, gradient accumulation, clipping and optimizer step

PPO performs repeated minibatch backward passes and configured accumulation/clipping. SAC steps each optimizer once per replay batch, recomputes the actor objective after critic Adam steps, and uses that actor forward's entropy for alpha. `_train_sac_transition()` interpolates the full target critic after all three optimizer steps, including floating buffers and copying integer buffers.

### 4.8 The optional recurrent PPO branch is different

PPO [`_train_ppo_trajectory`](../../pufferlib/pufferl.py#L645) preserves sequences and refreshes advantages and values. SAC's replay learner is feed-forward; recurrent policies require sequence replay and are outside this trace.

## 5. Finish the cycle and the run

### 5.1 Return from the update helper: schedule, log and checkpoint

PPO advances its schedule and epoch. An epoch counts one collection/update cycle, not one episode or optimizer step.

SAC logs with PPO's `losses/`, `environment/` and `performance/` prefixes. `PuffeSAC.train()` returns `None` between log emissions and includes the latest cycle's losses at an emission. Logging follows the 0.25-second PPO cadence, with a final-budget flush. The outer run manager invokes callbacks only for emitted logs. Best-score checks use only a fresh emission for the current epoch. Warmup creates no learner updates until `len(replay) >= learning_starts`; once that threshold is met, even a budget-limited partial refresh can train over the full current replay. `save_checkpoint()` takes no path and returns the actor path under `models/`; `resume_state_path` is an explicit config field and resume is never automatic.

### 5.2 What `mean_and_log()` actually reports

SAC reports cumulative receive-slot `global_step`, admitted inserts, current replay occupancy, cumulative `sampled_transitions`, and cycle/update counts. `sampled_transitions` counts the actual records in every minibatch, including the final short batch and records revisited in later epochs; it is not calculated as `learner_updates * minibatch_size`. Loss/Q/entropy/alpha metrics are means over updates in the latest cycle. Raw environment statistics accumulate between emissions. Profiling names match PPO and reuse `Profile`/`Utilization`, dashboard, uptime and SPS reporting.

### 5.3 Return to the outer `train()`: optional benchmarks and early stopping

The outer `train()` calls its configured stop hook only when `PuffeSAC.train()` emits logs, including warmup emissions; `PuffeSAC.train()` itself does not invoke that callback. It returns `None` between emissions. The run stops at the configured received-row budget, and the final full collection batch can pass `total_timesteps`.

### 5.4 Understand exactly what gets saved

PPO separates inference weights, trainer state and configuration. SAC writes a format-5 **learner-only** `trainer_state.pt` containing actor, online/target critics, log-alpha, three optimizer states, both scheduler states even when annealing is disabled, configuration, scheduler horizon, counters including `sampled_transitions`, best score and global RNG state. It stores no update credit; older state formats are rejected. `load_training_state()` restores into caller-supplied compatible architectures; module state dictionaries must match.

Periodic actor artifacts use PPO's `models/` layout; the final actor is `data_dir/final_model.pt`, and best actors are under `best_models/`. `save_checkpoint()` requires no path and returns `models/model_<env>_<epoch>.pt`. Replay records, component-private RNGs, pending actions, environment state and Drive action-layout metadata are not saved. Restore with an explicitly configured `resume_state_path` into a fresh trainer, freshly reset vecenv, cleared pending arrays, and empty replay; replay must refill before learning resumes. This does not continue the same simulator trajectory.

### 5.5 Budget reached: final evaluation, final logs and shutdown

On normal completion or early stop, `close()` closes vecenv and monitor, saves the final actor, and returns its path. The outer run manager owns logger shutdown. Its final log flush occurs when the configured training budget is reached, not as an arbitrary extra shutdown emission. On errors, cleanup closes resources without publishing a partial final checkpoint. The outer run stops when restored `global_step` reaches the current configured budget; the saved scheduler state still retains its original cosine horizon. When collection continues, the last sent action remains pending until the next `evaluate()`.

For the supplied-component call contract, see [“Supplying components and restoring a learner”](SAC_implementation_tutorial.md#supplying-components-and-restoring-a-learner).

### Verification for this document

Historical validation on 2026-10-09, before the replay redesign: five focused training-wrapper checks and nine existing CPU checks passed for the update-credit implementation. Those checks covered delegation/bookkeeping, warmup and fractional credits, averaged losses, nested profile closure, and deterministic updates on fixed replay batches; they do not validate the current schedule. Current replay-design validation on 2026-10-09: syntax compilation/imports and one temporary CPU smoke passed. The smoke filled replay to seven records, performed three distinct replacements while retaining four old records, then ran two training calls with two epochs each. Every pass used minibatches of 3/3/1, for 12 learner updates and 28 actual sampled records total; it also checked CPU `int64` indices, finite losses, format-5 actor/counter restore, and environment/monitor cleanup. It did not test Drive integration, Hydra construction, GPU behavior, or the outer run wrapper. The smoke harness was temporary and is not a new test suite.
