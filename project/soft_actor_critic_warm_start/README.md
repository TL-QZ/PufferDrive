# PPO to SAC warm start: design

Status: implemented with native PPO/SAC learner reuse. See [validation evidence](VALIDATION.md) for the checks performed and remaining GPU/performance limits.

## Run inputs and commands

The entry point is `pufferlib.pufferl_sac_ppo_warm_start`. Its coordinator uses native `PuffeRL` and `PuffeSAC` operations. Edit **[`config/warm_start.yaml`](config/warm_start.yaml)** to configure this experiment. It contains a dedicated snapshot of the baseline PPO and selected round-four trial 97 settings, including the hybrid overrides; preparing a run no longer reloads those other experiments' training configs.

The launcher uses this YAML by default. Pass `--config PATH` to select another dedicated YAML. Runtime `--run-dir` determines output paths, and an explicit `--seed` overrides the YAML seed. Each run saves the resolved configuration as `<run-dir>/warm_start_config.yaml`.

Run these steps in the same shell, on the machine allocated for training. Only step 5 starts full training.

1. Enter the repository and activate its environment.

   ```bash
   cd /common/users/ql236/auto_driving/PufferDrive
   source .venv/bin/activate
   ```

2. Optionally run the reduced CPU smoke (small model/buffers; no W&B, compilation, or benchmark evaluation).

   ```bash
   bash project/soft_actor_critic_warm_start/scripts/launch.sh \
     --run-dir "/tmp/ppo_sac_smoke_$(date +%Y%m%d_%H%M%S)" --smoke
   ```

3. Prepare the full configuration without starting training. This creates a fresh run directory and prints the replay-memory estimate. W&B remains enabled in the saved configuration; the dry run does not initialize it. Add `--no-wandb` here if the actual training run should not use W&B.

   ```bash
   WARM_START_RUN_DIR="experiments/ppo_sac_seed0_$(date +%Y%m%d_%H%M%S)"
   bash project/soft_actor_critic_warm_start/scripts/launch.sh \
     --run-dir "$WARM_START_RUN_DIR" --seed 0 --gpus 0,1 --dry-run
   ```

4. Review the saved configuration; press `q` to leave `less`. Check `warm_start.handoff_steps: 2000000000`, `warm_start.sac_steps: 2000000000`, PPO's `total_timesteps: 10000000000`, and SAC's `learn_alpha: false` / `initial_alpha: 0.1`.

   ```bash
   less "$WARM_START_RUN_DIR/warm_start_config.yaml"
   ```

5. Start the exact configuration reviewed above on GPUs 0 and 1. Change `CUDA_VISIBLE_DEVICES` if different GPUs were allocated. This runs in the foreground; use your persistent terminal or job allocation for the long run.

   ```bash
   CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=1 \
     torchrun --standalone --nnodes=1 --nproc-per-node=2 --max-restarts=0 \
     -m pufferlib.pufferl_sac_ppo_warm_start \
     --config "$WARM_START_RUN_DIR/warm_start_config.yaml"
   ```

Do not rerun `launch.sh` with the directory created in step 3: the launcher intentionally refuses an existing directory. Step 5 directly consumes its saved configuration. Full training uses a 2G handoff, the unchanged 10G PPO schedule horizon, and 2G additional SAC steps. `--smoke` validates the lifecycle only; GPU/compiled parity and full-size memory/throughput remain unverified. Replay alone requires approximately 112.92 GiB host RAM across two ranks, before PPO/model/simulator storage. Exact interrupted-run resume is unsupported because native checkpoints omit replay and live simulator state.

Train the baseline PPO actor for approximately 2 billion collected agent steps while training independent SAC twin critics alongside it. Then switch actor learning from PPO to SAC, retaining the trained critics and replay. The hypothesis is that PPO supplies useful driving behavior early, while concurrent critic training avoids a cold critic start at the switch.

The observed approximately 0.42 PPO score near 2G steps and weak early SAC scores are user-reported motivation, not results established by this document. Warm starting does not establish that SAC will improve beyond continued PPO.

## 1. Agreed experiment

| Decision | Design |
| --- | --- |
| PPO source | Baseline `nightly_best.yaml`, root policy defaults, and baseline two-GPU launcher overrides |
| SAC source | Round 4 trial 97 configuration |
| PPO collection | Normal complete PPO rollout; no separate SAC simulator collection during this phase |
| PPO updates | Three epochs using the existing PPO loss, value function, filtering, and schedule |
| Replay refresh | Initial fill, then replace 10% once per full PPO rollout |
| Critic updates | Three shuffled passes over full replay after PPO updates: Option A |
| Critic target policy | The just-updated PPO actor, detached and unchanged throughout critic training |
| Temperature | Fixed alpha = 0.1 during critic pretraining and SAC continuation for this first design |
| Handoff | After a complete PPO-and-critic cycle at or above 2G global collected agent steps |
| SAC continuation | Trial 97 collection/update cadence, with fixed temperature as the explicit override |

There is no automatic temperature-tuning stage in this design. Adding one later is a separate experiment. Trial 97's target entropy is retained as source provenance but is inactive under fixed alpha.

## 2. Configuration sources and resolved controls

The authoritative training configuration is [`config/warm_start.yaml`](config/warm_start.yaml). The following files document where its initial settings came from; the benchmark catalog remains the shared evaluation input:

| Source | Role |
| --- | --- |
| [`puffer_drive.yaml`](../../pufferlib/config/puffer_drive.yaml) | PPO policy, optimizer, rollout, and environment defaults |
| [`nightly_best.yaml`](../baseline_run_sync_2026-08-24/override_config/nightly_best.yaml) | Baseline environment, schedule budget, and evaluation overrides |
| [`launch_2gpu.sh`](../baseline_run_sync_2026-08-24/train/launch_2gpu.sh) | Two ranks, 20 environments per rank, receive batch 10, minibatch sizes, selected seed |
| [Round 4 configuration directory](../soft_actor_critic_tuning/configs/round4) | Tunable, specific, fixed configuration layers and trial 97 grid selection |
| [`evaluation_benchmarks.yaml`](../baseline_run_sync_2026-08-24/override_config/evaluation_benchmarks.yaml) | Common evaluation protocol |

The dedicated YAML records trial 97's resolved gamma 0.999, tau 0.005, three training epochs, and replay refresh fraction 0.1. Its provenance identifies the original trial; no grid selection happens at launch.

| Control | PPO phase | SAC continuation |
| --- | --- | --- |
| GPUs / ranks | 2 | 2 |
| Environments per rank / agent slots per environment | 20 / 3,200 | Same |
| Training dt | 0.3 seconds | Same |
| Policy | Feed-forward discrete Drive actor | Compatible discrete DriveSAC actor |
| PPO rollout capacity per rank | 8,192,000 rows, horizon 128 | Released after handoff |
| SAC replay capacity per rank | 8,192,000 transitions | Retained |
| Logical minibatch / maximum microbatch per rank | 128,000 / 32,000 | Same |
| PPO actor/value initial learning rate | 0.0005, baseline cosine schedule | PPO optimizer retired |
| SAC critic learning rate | 0.0005, no annealing | Retained, no annealing |
| SAC actor learning rate | No SAC actor updates | 0.0005, fresh optimizer, no annealing |
| Gamma / tau | 0.999 / 0.005 for SAC critics | Same |
| Temperature | Fixed 0.1 | Fixed 0.1 |
| Precision / compilation | Baseline BF16 autocast and compilation | Trial 97 BF16 autocast and compilation |
| Replay placement | CPU, pinned for CUDA transfers | Retained |

Use seed 0 for the initial paired experiment, matching trial 97; baseline launcher supports selecting it. Preserve vector seed 42 and rank derivation. Store the fully resolved source configs and explicit hybrid overrides with each run.

Keep PPO's original **10G cosine schedule horizon**, separately from the **2G handoff threshold**. The scheduler in `PuffeRL.__init__` derives its cycle horizon from `total_timesteps / batch_size`; replacing that budget with 2G changes PPO's learning trajectory. The earlier conversational linear-decay example was illustrative: the actual implementation uses `CosineAnnealingLR`.

Continuation budget: **2G additional global received agent steps**, overriding trial 97's original 500M budget. This yields approximately 4G cumulative steps (2G PPO + 2G SAC), including cycle overshoot. The budget is summed across ranks, not per GPU.

## 3. Execution and ownership

```text
One shared vector environment
        |
        v
Collect full PPO rollout with actor version k
        |                         |
        v                         v
PPO rollout storage       Completed transition capture
        |                         |
        |                  Initial replay fill / 10% refresh
        v
Three PPO epochs -> actor version k+1
                                  |
                                  v
                   Three SAC critic epochs over replay
                   Targets use detached actor version k+1
                                  |
                   Repeat, or hand off after >= 2G
```

“Alongside PPO” means sequential phases within each cycle, not concurrent CUDA jobs or separate asynchronous trainers. The actor does not change while the critics train. Critic gradients must never reach PPO actor/value parameters.

During pretraining, use the PPO actor directly as the target-probability provider. Do not keep a second trainable actor copy that could become stale. Adapt the discrete PPO output tuple into logits `[B, K]`; do not use its value output in SAC targets.

Q1 and Q2 own independent backbones and heads. Proposed initial implementation: randomly initialize both full Q networks independently, then copy online critics into their corresponding target critics. This is a concrete starting choice; copying PPO features into critic backbones remains a later ablation. No Q parameters alias the PPO network or each other.

The existing PPO value function remains necessary for PPO advantages and losses. It has no role in SAC critic targets and is retired after the switch.

### Option A cadence

With replay capacity N and refresh fraction 0.1, one full PPO cycle refreshes 0.1N slots and trains critics on 3N replay samples. At the selected capacity, each critic phase has 64 logical minibatches per pass, or **192 twin-critic optimizer steps** across three passes, per synchronized rank. Each logical minibatch fits four 32,000-row microbatches.

Three epochs does not mean identical effective sample use between PPO and critics: PPO applies its masks/advantage filtering to fresh rollout data; critics use persistent replay. We deliberately accept that distinction.

After handoff, ordinary trial 97 cadence resumes: collect 0.1N new replay transitions, then run three SAC passes. Its optimizer work per new transition is higher than during pretraining. This change is intentional and should be visible in timing logs.

## 4. Transition capture and replay refresh

Reuse the same `vecenv.recv()` stream used by PPO. Capture transitions before PPO-specific reward transformations or advantage filtering. Baseline reward normalization is disabled, but keep raw simulator reward semantics explicit.

| Replay field | Contract |
| --- | --- |
| Observation | FP32 observation on which the previous action was selected |
| Action | Int64 joint discrete action ID actually sampled, not the decoded continuous control |
| Reward | FP32 reward returned for that preceding action |
| Successor observation | FP32 next observation, using the pre-reset final observation for pure truncations |
| Discount | Zero for a true terminal; gamma otherwise |

Maintain pending observation/action state indexed by stable agent-slot IDs. A received row completes a transition only when that slot has a pending action and its returned validity mask qualifies the preceding action. Initial/reset-only rows cannot be fabricated into transitions. For a true terminal, use the existing SAC convention of a finite owned observation with discount zero. Missing required truncation final observations are an error.

Enable the existing final-observation transport during hybrid PPO collection. The PPO collector must consume/remove private final-observation metadata before normal metric aggregation, while preserving PPO's existing rollout and bootstrapping behavior. Test that enabling transport does not change simulator trajectories.

Replay owns copies of all inserted tensors. Never retain references to vector zero-copy buffers or PPO storage that will be overwritten. Pending transitions must survive PPO cycle boundaries; an action sent before a boundary can complete afterward.

### Initial fill

Fill unused replay slots from eligible transitions during normal PPO collection. If the first rollout does not supply enough valid transitions, continue filling over later PPO cycles. Do not extend PPO collection merely to fill SAC replay and do not pad records. Skip critic training until replay is full; synchronize readiness across ranks so no rank enters critic collectives alone.

If replay becomes full partway through a cycle, finish that PPO rollout without beginning an additional 10% refresh. Train critics at its end once all ranks are ready. Start the regular refresh rule on the next cycle.

### Once-full refresh

At each collection-cycle start, select 819,200 distinct destination slots per rank using a dedicated replay RNG. Select fresh eligible transitions uniformly across the complete collection interval, rather than taking only its earliest chunk.

Proposed bounded-memory implementation: reservoir sampling into those designated replay slots while critic training is idle. The final contents are a uniform sample of eligible transitions seen during that cycle; repeated candidate writes may touch a designated slot, but the final changed-slot set remains the selected 10%. This avoids a third full transition buffer. Implement selection in batches with explicitly bounded work, and validate the sampling independently.

If fewer eligible transitions arrive than the quota, insert only those available, leave unused destination slots unchanged, and log the actual refresh count. This is variable valid-data volume, not an excuse to manufacture transitions. Baseline-scale expected quotas must be checked in a smoke run.

At 10% random replacement, a slot survives k cycles with probability 0.9^k; its expected replacement waiting time is ten cycles. Data age is therefore meaningful when the PPO actor changes. Log insertion cycle/step metadata if inexpensive, and report replay age alongside critic diagnostics.

## 5. Critic learning

For the just-updated PPO actor pi, target critics Qbar1/Qbar2, and fixed alpha 0.1:

```text
next_value = sum_a pi(a|next_obs) * (
    min(Qbar1(next_obs,a), Qbar2(next_obs,a)) - 0.1 * log pi(a|next_obs)
)
target = reward + discount * next_value
critic_loss = MSE(Q1(obs, action), target) + MSE(Q2(obs, action), target)
```

Reuse current SAC FP32 probability/target/loss arithmetic, microbatch accumulation, gradient clipping, and distributed synchronization. After each logical critic optimizer step, Polyak-update target parameters with tau 0.005, including the existing buffer-copy rules. Do not postpone target updates until the end of three epochs.

There are no SAC actor or alpha optimizer steps during this phase. Fixed temperature must be implemented explicitly: `initial_alpha=0.1` alone currently enables learned temperature and does not meet this design.

Critics are off-policy with respect to behavior data, but evaluate the current target actor. Historical PPO transitions are legitimate replay input; critic lag and weak estimates for rarely sampled actions remain possible. Stable losses alone do not prove handoff readiness or driving improvement.

## 6. Handoff contract

Perform the switch only after a complete PPO update followed by its critic phase. Compare the global sum of received agent rows against 2G, and log the actual boundary. Never trigger independently from rank-local counters or truncate the final PPO rollout to hit an exact number.

| State | Handoff action |
| --- | --- |
| Actor backbone/head | Copy into compatible DriveSAC actor; verify action probabilities before any SAC update |
| PPO value network and optimizer/scheduler | Save for provenance/continued-PPO control, then retire |
| Online/target critics and critic optimizer | Retain without reinitialization |
| Replay and refresh RNG | Retain, including occupancy and sampling state |
| Temperature | Retain fixed 0.1; no alpha optimizer |
| SAC actor optimizer | Create fresh with trial 97 optimizer settings |
| Environment and pending transitions | Preserve live state and outstanding receive/send ownership |
| Counters | Keep cumulative received steps; start separate SAC-phase steps at zero |

The preferred handoff occurs in one process without an environment reset. The SAC collector must inherit pending observation/action state and any outstanding vector work; constructing a fresh collector that resets the environment or discards pending transitions is not equivalent. Count every received row once, including rows completing pre-switch actions.

Do not construct a second replay or duplicate critic optimizer state during handoff. Release PPO rollout storage once its final update and required checkpoint are complete, before allocating avoidable SAC-only working storage. Validate memory peaks, not just final steady state.

Save a boundary artifact before SAC actor learning: actor, PPO continuation state, critics/targets/optimizer, phase, counters, resolved configuration, RNG states, and provenance. Existing trainer checkpoints do **not** save replay or live simulator state. Therefore they cannot promise exact hybrid resume. Initial scope should declare interrupted-run recovery unsupported unless replay/environment continuation is explicitly designed; do not silently reload weights with empty replay and call it the same continuation.

## 7. Reproducibility and accounting

Preserve baseline PPO random streams. Critic initialization, replay selection, and replay permutations need deterministic isolated streams derived from the experiment seed and rank. Explicit generators are appropriate for replay; initialization may require saving/restoring Torch CPU/CUDA RNG state. Check evaluation also preserves training RNG state.

PPO's current `torch_deterministic=false` means exact equality is not established by seeding alone. Required validation is a paired short baseline-versus-hybrid run on the same hardware/software, comparing actor samples, rollout ordering, PPO minibatch choices, and actor weights before divergence from numerical nondeterminism. Isolation must not alter the baseline's own seed derivation or optimizer schedule.

Use separate names in the future hybrid configuration for PPO rollout capacity, SAC replay capacity, PPO schedule horizon, handoff step, and SAC continuation budget. Both existing learners use `train.batch_size` for different purposes; passing one mutable configuration dictionary between them is unsafe.

Count collected experience only once even though both buffers consume it. Keep global cumulative received steps as the common curve axis, with separate valid replay inserts, critic optimizer steps, replay samples consumed, PPO cycles, and SAC-phase steps. Do not increment collected steps during critic training. PPO's existing rank-local budget normalization and SAC's global budget semantics must not double-divide the hybrid thresholds.

## 8. Memory and compute budget

For N transitions and D FP32 observation features, current `TensorReplay` uses approximately:

```text
N * (8*D + 16) bytes
# obs + next_obs: 8*D; action int64: 8; reward + discount: 8
```

At D=923 and N=8,192,000 per rank, replay alone is approximately **56.46 GiB per rank**, or **112.92 GiB across two ranks**, in host RAM. Confirm D from the resolved live environment before allocation. The existing 10% replay headroom estimate is approximately 124.2 GiB total; it excludes simulator/process memory and PPO storage.

One FP32 PPO observation rollout at the same N and D would be another **28.17 GiB per rank**, before PPO scalar fields, indexing, activations, or working copies. This is a size illustration; actual allocation depends on the collector's storage dtype and device. The baseline has `cpu_offload=false`, whereas SAC replay is CPU-offloaded. Measure host RSS/pinned memory and per-rank allocated/reserved VRAM separately; do not describe the entire combined budget as host RAM.

Additional persistent device allocations include two online critics, two target critics, critic gradients and AdamW state, and pending transition observations. Sequential PPO/critic phases avoid simultaneously live backward graphs, but compilation caches and allocator reservations can still raise peak memory. Reservoir refresh avoids another quota-sized pair of observation buffers.

Measure collection seconds, PPO update seconds, critic update seconds, evaluation seconds, and total elapsed seconds. Option A adds critic work once per PPO cycle; it does not preserve baseline wall-clock throughput. No full-scale capacity or throughput claim is established yet.

## 9. Comparison and acceptance criteria

Native learner parity is a requirement. During the PPO phase, the hybrid must execute the existing PPO collection and learning path, with critic work isolated from PPO parameters, data selection, scheduling, and randomness. The objective is to retain baseline PPO driving performance at matched collected steps; added critic compute may reduce wall-clock throughput. Reuse alone does not prove this requirement: paired runs must verify it.

During SAC continuation, execute the existing SAC learning path with the transferred state and the explicitly agreed fixed-temperature override. Given identical actor/critic/target weights, optimizer state, replay minibatches, temperature, and RNG state, the hybrid SAC updates must match native SAC updates configured equivalently within numerical tolerances. Warm-started SAC is not expected to reproduce a from-scratch trial's curve, because its starting state differs. Driving performance must still be measured, and performance regressions caused by orchestration are unacceptable.

The primary control is continued baseline PPO at matched cumulative steps and matched evaluation settings. The hybrid uses its first approximately 2G steps for PPO plus critic pretraining, then SAC. Compare both sample efficiency and total wall-clock time, including critic pretraining. Keep standalone trial 97 as context; fixed alpha makes it an imperfect algorithmic control, which must be stated.

Preserve baseline PPO-phase evaluation cadence and trial 97 SAC-phase cadence initially, and force an evaluation at the boundary before the first SAC actor update. Keep benchmark definition and action-selection mode identical across runs. Evaluation-cycle numbers are not a shared x-axis when cadence changes.

| Gate | Required evidence |
| --- | --- |
| Actor compatibility | Matching state shapes, action order, observations, logits/probabilities, and evaluation behavior before SAC updates |
| Transition correctness | Slot reorder, first receive, masks, terminal, pure truncation, cycle boundary, and handoff pending-action cases |
| Learning isolation | Critic updates change Q networks/targets but not PPO actor/value parameters or PPO RNG streams; alpha stays exactly fixed |
| Cycle/handoff correctness | Initial-fill readiness, refresh quotas, three full critic passes, 192 logical updates at full settings, synchronized switch, continuous counts |
| Runtime/experiment | Reduced two-rank smoke with finite losses and memory measurements; then driving score, progress/goals, speed, infractions, and stability after handoff |

Actor parity checks should use FP32 eager execution first and then BF16/compiled tolerances. Training parity and model loading are separate checks. Low critic loss is not a substitute for evaluating retained driving behavior.

## 10. Implementation boundaries and remaining decisions

Keep future experiment configs, launch wrapper, reports, and design in this directory. Preserve source baseline and round-four files unchanged. Introduce only the reusable core interfaces needed for the hybrid; avoid copying whole PPO/SAC trainers into an experiment script.

Reuse is the default architecture, not just a preference for similar formulas. Keep native PPO advantage calculation, filtering, loss, optimizer, scheduler, distributed synchronization, and collection ordering. Keep native SAC Bellman targets, critic and actor losses, optimizer mechanics, target updates, replay storage, and evaluation behavior. Do not create alternative learner implementations to make orchestration convenient.

The orchestration module is **[`pufferlib/pufferl_sac_ppo_warm_start.py`](../../pufferlib/pufferl_sac_ppo_warm_start.py)**, invoked as `pufferlib.pufferl_sac_ppo_warm_start`. Its responsibility is phase coordination and state transfer, not duplicated training algorithms.

For each proposed core change, first identify the native operation being reused and the exact missing interface. Prefer direct calls or a small optional interface with unchanged native defaults. If a shared critic-update operation must be extracted from SAC, both native SAC and the hybrid must call that same operation, with update-equivalence checks. Apply the same rule to transition assembly. Avoid a broad trainer-framework rewrite.

The reservoir sampler described above is a proposal, not a requirement to build a new sampling subsystem. Before implementing it, assess existing selection and replay APIs against the agreed once-per-rollout refresh contract. Retain new selection logic only where existing mechanisms cannot meet that contract without bias or excessive extra storage.

| Existing source | Intended integration responsibility |
| --- | --- |
| [`pufferl.py`](../../pufferlib/pufferl.py), `evaluate`, `train`, scheduler setup | Observe completed transitions without changing PPO rollout behavior; expose cycle boundary and separate schedule/stop semantics |
| [`pufferl_sac.py`](../../pufferlib/pufferl_sac.py), `_compute_targets`, `_critic_loss`, `_train_sac_transition` | Reuse critic-only updates with an external actor; fixed-alpha mode; retain complete SAC continuation behavior |
| [`sac_replay.py`](../../pufferlib/sac_replay.py), `TensorReplay` | Reuse owned transition storage and bounded refresh selection |
| [`ocean/torch.py`](../../pufferlib/ocean/torch.py) and [`ocean/torch_sac.py`](../../pufferlib/ocean/torch_sac.py) | Actor compatibility and independent critic ownership |
| `pufferlib/pufferl_sac_ppo_warm_start.py` | Coordinator: one vecenv owner, phase state, isolated RNG, budget/counter semantics, handoff, and artifact management |

Implementation order:

1. Resolve isolated configs and actor compatibility; add fixed-temperature and critic-only update contracts.
2. Integrate completed-transition capture, deterministic replay refresh, and paired PPO-isolation checks.
3. Add Option A orchestration and synchronized in-process handoff with continuous accounting.
4. Add boundary artifacts and evaluation reporting with explicit resume limitations.
5. Run reduced two-rank validation and memory preflight before considering full-scale training.

The prepared run uses a 2G SAC continuation budget. Before a full-scale launch, establish resource fit and decide whether crash recovery is required. Exact resume remains unsupported; completed learner artifacts can still be inspected and evaluated. No full-scale training was launched during implementation.

Validation status: native interface tests, paired CPU PPO parity, actor-transfer checks, and reduced real-simulator CPU runs have been performed. Full-scale GPU allocation, compiled/BF16 parity, throughput, and driving-score improvement remain unverified. See [VALIDATION.md](VALIDATION.md).
