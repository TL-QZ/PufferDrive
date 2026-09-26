# Observation decoder probe — draft for discussion

**Goal:** test whether the JEPA predicted endpoint latent contains useful information about the future observation. This decoder is separate from the existing action chunk decoder. **Status: implemented; two-GPU toy and regression tests passed.**

**Confirmed design:** frozen JEPA, full-observation reconstruction, permutation-aware matching within each object group, and fresh experience collected one round at a time. Delete completed training rounds after checkpointing. Production uses the toy-verified Condition B checkpoint, five epochs per collection, and constant learning rate `0.001`. Batch/GPU starting values come from profiling below.

## 1. Proposed experiment

| Decision | Proposed starting point | Reason |
|---|---|---|
| Training mode | Load one Condition B checkpoint; freeze student encoder, target encoder, predictor and action decoder | Measure the representation already learned |
| Probe training pairs | Normalized target-encoder latent of a real observation → that observation | Predictor learns to match this target latent space |
| Probe architecture | Separate MLP, latent width → 1024 → 1024 → observation heads; ReLU | Configurable nonlinear readout |
| Future test | Decode predicted endpoint latent using the same trained probe | Directly compare predicted and observed futures |
| Initial horizon | Existing `K=4`; 1.2 seconds at configured `dt=0.3` | Current predictor produces one endpoint, not four intermediate states |

- Freeze the checkpoint throughout probe training; gradients update only the observation decoder. Joint reconstruction training would be a separate experiment that changes JEPA representations.
- Train on real target latents only. Training the probe directly on predicted latents could teach it to compensate for predictor errors and obscure the diagnostic.
- Normalize both real and predicted latents before decoding. Current JEPA loss constrains latent direction, not matching vector magnitude.
- The output is the agent's local observation, not the complete simulator state. No claims about hidden objects or full-world reconstruction.

## 2. Inputs, outputs and loss

Let `normalize` mean unit L2 normalization, `g` the frozen target encoder, `f` the frozen student encoder, `h` the frozen JEPA predictor, and `R` the new trainable observation decoder.

| Path | Computation | Compare against |
|---|---|---|
| Probe training / real-latent reconstruction | `R(normalize(g(o[t+K])))` | Actual `o[t+K]` |
| Predicted-future reconstruction | `R(normalize(h(f(o[t]), a[t:t+K])))` | Same actual `o[t+K]` |
| Current-observation baseline | `o[t]` treated as a prediction of `o[t+K]` | Same actual `o[t+K]` |

Actions are the **executed logged controls**, as in current JEPA training. This tests prediction under observed interactions, not alternative-action rollouts or closed-loop driving.

| Observation group | Candidate output/loss | Handling |
|---|---|---|
| Continuous ego, goal and object attributes | Linear outputs; Smooth L1 initially; report MAE too | Existing observation units/scales; report physical units where inversion is defined |
| Categorical attributes | Per-field logits with cross-entropy | Audit field definitions and legal category values before implementation |
| Object presence/counts | Presence logits or count head; classification loss | Use actual valid counts for attribute-loss masks; score predicted presence separately |
| Reward-conditioning/context fields | Reconstruct if full observation is selected | Report separately; constant/easy fields must not dominate results |

- Derive dimensions and masks from checkpoint observation metadata, never a hard-coded observation width. Audit exact feature semantics in simulator observation construction before finalizing heads.
- Average valid attribute losses within each group, then combine group losses with explicit weights. Padded slots must not dominate accuracy; never infer validity merely from zero-valued features.
- **Object ordering:** use permutation-aware matching within each group. DriveBackbone pools objects using max operations, so a flat slotwise loss could punish harmless permutations. Matching uses squared continuous distance (weight 1), traffic category cross-entropy (1), and positive-presence cost (0.2). SciPy solves each rectangular assignment exactly.

## 3. Evaluation and interpretation

| Check | What it tells us |
|---|---|
| Decode actual future target latent | How much the frozen representation and probe can reconstruct together |
| Decode predicted future latent | End-to-end quality of the JEPA future representation |
| Difference between those errors | Additional error associated with prediction; not an exact decomposition |
| Current-observation and training-mean baselines | Whether decoded predictions improve on simple forecasts |
| Optional shuffled-action control | Sensitivity to action conditioning; cannot prove counterfactual accuracy |

- Use separate training, validation and test scenario identities. Audit existing manifests for scenario overlap: distinct collection seeds alone do not prove a held-out scenario split. Fit any extra scaling on training data only.
- Select probe checkpoints on real-latent validation reconstruction. Report both paths on the same held-out windows, with per-group errors and sample counts; keep test data out of tuning.
- Visuals: actual future, decoded real future, decoded predicted future in the **future agent-local frame**. Show object geometry where supported. Current plots use fixed first held-out windows; error-ranked examples can be added later.
- Good real-latent reconstruction but poor predicted reconstruction points toward a predictor problem. Poor results on both may reflect lost information, probe capacity or representation mismatch; they do not alone prove JEPA is useless for driving.
- For pretrained-versus-random encoder runs, train a separate probe per checkpoint with the same capacity, data and budget. Do not reuse probe weights across unaligned latent spaces.

### Observation rendering

Reuse `pufferlib.viz.plot_observation()` for a three-panel, 2D ego-centered view:

| Panel | Source |
|---|---|
| Actual future | Saved `o[t+K]` |
| Reconstructed real future | `R(normalize(g(o[t+K])))` |
| Predicted future | `R(normalize(h(f(o[t]), a[t:t+K])))` |

- Direct observation rendering is verified: one saved 1,292-value observation produced a 2,000 × 2,000 RGB image without a simulator or GPU. This is a geometry visualization, not a camera image.
- Pass the selected checkpoint's layout, slot counts and normalization scales; use identical axes and drawing settings for all panels.
- Convert categorical heads to category IDs and use **predicted presence** to zero absent decoded slots. The current plotter skips exactly-zero slots; raw floating-point decoder outputs would otherwise create spurious objects. Ground-truth masks must not hide prediction errors.
- Display ego quantities and per-group reconstruction errors alongside geometry. Use fixed held-out examples across evaluations; optionally save frame sequences using the same observation-local convention.

## 4. Fresh experience and training schedule

The default probe collects fresh teacher-driven experience from the source JEPA checkpoint's simulator recipe. It keeps only the current training collection per rank, then deletes it after training and checkpointing. Historical JEPA training caches are no longer a prerequisite.

| Stored item | Use for decoder probe |
|---|---|
| `observations.npy` | Actual current/future observations; reconstruction labels |
| `controls.npy` | Executed controls for predicted-future evaluation |
| `window_indices.npy` and validity metadata | Existing eligible windows; preserve reset/agent-boundary filtering |
| Manifest layout and effective config | Feature layout, normalization and compatibility checks |
| Teacher logits | Written by the shared collector, but not required for observation reconstruction |

1. Load **one fixed JEPA checkpoint** and initialize the observation decoder once. Recompute latents from stored observations using that checkpoint; do not change encoders between collections.
2. Collect one fresh round per training rank and pool those manifests as one logical collection. Two GPUs produce two shards of the same training stage.
3. Shuffle valid windows and train for `epochs_per_collection` passes over this collection. An epoch means one pass over its selected valid windows, not one optimizer step.
4. Evaluate on fixed held-out data and save the decoder, optimizer and progress. Close all readers and delete the completed training round before collecting the next one, retaining decoder/optimizer state.
5. Stop after the configured collections or an explicit optimizer-step cap. A mid-round stop retains its one active collection for sampler resume; completed rounds are removed. Resume from the latest checkpoint. Subsequent fresh collections restart the simulator because its live state is not checkpointed.

| Proposed setting | Decision |
|---|---|
| Training storage | Dedicated probe run data, separate from historical JEPA caches |
| Collection schedule | Fresh experience, one collection at a time |
| `training.epochs_per_collection` | **5 full passes per collection**, then evaluate and plot five examples |
| Batch / microbatch / GPU count | Global 16,384 / per-GPU 1,024 / two GPUs; profiling starting point below |
| Learning rate / step cap | Constant `0.001`; no optimizer-step cap |
| Validation / plots | After each collection's training initially; configurable cadence |

- Reuse the streaming format through `ProbeCollectionDataset` and pooled rank-manifest loading, with bounded memory-mapped batches. Both the decoder probe and Condition B training collect fresh experience and remove completed training rounds.
- For full epochs without a step cap, updates are `sum over rounds [epochs_per_collection × ceil(selected windows in round / global effective batch)]`, subject to documented distributed padding/tail handling. Compute counts from actual valid windows, not the nominal transition budget.
- Collect a fixed validation set once with a separate seed, or explicitly provide a compatible external validation manifest. Retain this small dataset for evaluation and saved-decoder inspection.
- Distinct seeds do not establish scenario-disjoint evaluation; manifests do not expose enough scenario identity to make that claim.
- `data.mode=online` is the default. `data.collection_root` is a base directory;
  runtime data lives under `<collection_root>/<training.run_id>/rank_NNN/`.
  Null collection budgets inherit the source checkpoint's recipe. Explicit
  `collection.num_collections`, `transitions_per_round`, `max_transitions`,
  `max_disk_bytes`, and `validation_transitions` override those budgets.
  Environment/vector overrides under `collection` provide bounded toy settings.

## 5. Implementation sequence after decisions

1. **Audit:** checkpoint/layout compatibility, feature scales/categories, padding/order and scenario split isolation. Fix the output contract and masks.
2. **Build:** isolated `observation_probe/` model, saved-collection training loop and train/evaluate CLI; reuse checkpoint loading, streaming windows, pooled manifests and W&B infrastructure where suitable.
3. **Verify:** shape/mask/loss tests, frozen JEPA parameters, small-subset overfit check, saved-probe reload and bounded GPU smoke test. GPU selection must be explicit and checked for availability.
4. **Evaluate:** held-out reconstruction/prediction tables and three-panel observation plots; W&B namespaces `probe/reconstruction/*` and `probe/prediction/*`, plus collection/epoch/optimizer-step progress, separate run and output directory.
5. **Package:** full probe config and launch script for the user. No Condition B retraining or official probe launch during implementation verification.

## 6. Questions — edit answers here

Q1, Q2 and Q4 have confirmed baselines below; answer spaces remain for refinements. Production settings are resolved in `config/observation_probe.yaml`. The bounded implementation check uses `condition_b_jepa_distill/checkpoint.pt` (step 40,200).

### Q1. Diagnostic only, or also train JEPA with reconstruction?

**Recommendation:** diagnostic only, using a frozen saved checkpoint. Joint training can be a later ablation.

**Confirmed baseline:** diagnostic only; no reconstruction gradients into JEPA.

**Your answer / thoughts:**

> 

### Q2. What should we reconstruct first?

**Options:** full observation; ego state plus nearby agents; ego state only. Full observation best matches the stated goal but needs object matching and categorical/padding handling.

**Discussion so far:** leaning full observation. The full vector enters the encoder, but object groups are pooled and the four valid-count fields do not affect the latent with the current disabled padding mask. Report counts separately.

**Confirmed baseline:** full observation, with per-component metrics. Predict object presence and derive counts; do not treat counts as another continuous attribute head.

**Your answer / thoughts:**

> 

### Q3. Which checkpoint(s), and which dataset?

**Data direction confirmed:** reuse the 15 saved collections sequentially as described above. **Toy verified:** `experiments/jepa_distill/runs/condition_b_jepa_distill/checkpoint.pt`, step 40,200. **Production choice:** use that checkpoint and `datasets/condition_b_jepa_distill/rank_000/validation/round_0000/manifest.json` for fixed validation. Other checkpoints remain optional later comparisons. The schedule is five epochs per collection, with evaluation and five example plots afterward.

**Your answer / thoughts:**

> 

### Q4. Exact observation slots, or meaningful scene geometry?

**Recommendation:** permutation-aware object matching if geometry is the goal; retain slotwise metrics only as a secondary check. Ego/goal fields retain fixed positions. Pooling means original object order may not be recoverable.

**Confirmed baseline:** permutation-aware matching within object groups; fixed ego/goal/context fields keep their meanings.

**Your answer / thoughts:**

> 

## 7. Implementation and entrypoints

| Module | Responsibility |
|---|---|
| `config.py`, `data.py` | Validate config/manifests; read current/future observations and logged controls with bounded memory |
| `online.py` | Inspect source recipe, collect one round per rank, retain held-out data, and manage collection ownership |
| `schema.py`, `model.py` | Full observation field contract, frozen JEPA loading, trainable decoder heads |
| `losses.py` | Exact object matching; per-window/group Smooth L1, categorical CE and presence BCE |
| `train.py` | Sequential collections, microbatch accumulation, DDP, resume, checkpoints and W&B |
| `evaluate.py`, `render.py` | Reconstruction/prediction/baselines, per-group metrics, three-panel PNGs |
| `evaluate_checkpoint.py`, `profile.py` | Standalone saved-decoder evaluation; bounded real-data update profiling |

**Order:** preflight → freeze JEPA → initialize decoder/optimizer → collect experience → train epochs → validate/render/checkpoint → close and delete training collection → next collection. Only the decoder changes.

```bash
# Read-only inspection; no GPU, simulator or W&B initialization.
project/jepa_distill/scripts/train_observation_probe.sh --dry-run

# Bounded two-GPU toy: 2 collections × 1 epoch × 1,024 windows.
CUDA_VISIBLE_DEVICES=2,3 project/jepa_distill/scripts/train_observation_probe_2gpu.sh \
  --config project/jepa_distill/config/observation_probe_toy.yaml \
  --set training.run_id=YOUR_UNIQUE_TOY_RUN
```

- Install the additional dependency into `.venv`: `pip install -r project/jepa_distill/observation_probe/requirements.txt`.
- `batch_size` is **global windows/update**; `microbatch_size` is **windows/GPU/pass**. With two GPUs, local effective batch = global batch / 2. Accumulation preserves a per-window mean objective.
- A collection pools freshly collected rank manifests. Deterministic block shuffling limits index memory; the final partial block stays last. Full epochs visit every valid window, with only the final batch padded by duplicates to the configured global batch size. Padding counts are reported.
- Normal training requires a resolved source checkpoint, run ID, collection budgets, epochs, batch sizes, learning rate, device and world size. Production training uses five epochs per collection; the bounded toy keeps its own smaller settings.
- Output: `experiments/jepa_distill/observation_probe/runs/<run_id>/`. Existing nonempty directories require explicit `training.resume_checkpoint`. Resume binds the frozen checkpoint and scientific configuration; it does not require completed training collections that have been deleted. Existing historical caches are never purged automatically.
- Online collection data: `experiments/jepa_distill/observation_probe/collections/<run_id>/`.
  Checkpoints save the first-round mean baseline, so standalone decoder evaluation
  needs the frozen JEPA checkpoint and held-out data, not removed training rounds.

### Evaluate a saved decoder without training

```bash
source .venv/bin/activate
python -m project.jepa_distill.observation_probe.evaluate_checkpoint \
  --probe-checkpoint /path/to/probe/best.pt \
  --split validation --device cpu --max-windows 64 --render-samples 3 \
  --output-dir /path/to/NEW_EVALUATION_DIRECTORY
```

- Rechecks frozen-checkpoint/manifest hashes, restores decoder weights, and writes `evaluation.json` plus `renders/*.png`. It does not train or initialize W&B. `--split test` requires a test manifest in the saved run config.
- Existing nonempty output directories are rejected. The real toy `best.pt` passed this CLI on CPU with eight windows and one rendered comparison.

### Loss and observation contract

| Group | Active layout | Prediction / target handling |
|---|---|---|
| Ego + context | 10 + 26 continuous fields | Fixed field order; saved zero goal rows remain reconstruction targets |
| Partners | 16 × 9 | Continuous attributes + presence |
| Lanes / boundaries | 70 × 9 / 50 × 9 | Continuous attributes + presence; match within each group |
| Traffic controls | 4 × 7 | Five geometry values; type (4 classes), state (5 classes); presence |
| Counts | Four values | Derived from predicted presence; true counts define attribute masks |

- Actual sizes come from checkpoint metadata. The schema rejects unsupported feature widths.
- Each window averages valid objects/attributes within a group, then group weights combine losses. Empty groups still train absence, and all decoder heads stay connected for DDP.
- Predictions are linear geometry values, not physically constrained scenes. Rendering compacts presence-selected rows and zeroes padding. It does not use true object masks.
- The renderer passes checkpoint observation settings and corrects the plotter's road half-length interpretation locally. PNGs depict agent-local geometry, not simulator rollouts.

### Validation and limitations

- Train five epochs on each collection, then validate and save **five three-panel example plots** before moving to the next collection. Also validate at a capped run's final update. Score real-latent reconstruction, predicted-latent reconstruction, raw current-observation persistence and a slotwise training-mean baseline.
- The mean baseline is fitted only on a bounded subset of the first training collection and saved before that collection is deleted; category IDs/counts are rounded means. Persistence copies the current local vector without ego-motion compensation.
- Report per-group MAE, position MAE in meters, presence precision/recall, count MAE and traffic category accuracy, with denominators. Three-panel PNGs use fixed first held-out windows.
- W&B uploads [12 selected task metrics](observation_probe_metrics.md) plus progress; local `metrics.jsonl` retains all diagnostics. This probe performs **no closed-loop driving evaluation**; it diagnoses the frozen representation.
- Held-out layout, horizon, actions and observation normalization must match training/checkpoint metadata. Scenario disjointness remains **unverified**, because the available manifests do not identify scenarios.

### Previous cached-data verification and profiling

The results below describe the original cached-data implementation. They do not establish runtime throughput or simulator collection performance for the new online lifecycle.

- **55 tests passed**, including two-process CPU DDP, matching/masks, microbatch equivalence, data validity, resume and evaluation/rendering.
- **Two-GPU end-to-end toy passed:** 2 collections, 4 optimizer updates, validation after steps 2 and 4, PNGs, latest/best checkpoints and W&B sync. Real decoder/AdamW reload and frozen JEPA flags were checked.
- Toy validation reconstruction loss decreased from 4.5701 to 3.7808. Four updates verify plumbing, not a trained representation or useful scene forecasts.
- Successful W&B run: [observation_probe_toy_20260925_02](https://wandb.ai/tobieliu825/pufferdrive-jepa-distill-toy/runs/z329evr9).
- Workspace quota blocked the first toy's plot directory. Successful artifacts are at `/tmp/observation_probe_runs/observation_probe_toy_20260925_02`; raw profiling results are at `/tmp/observation_probe_profiling_20260925`. Choose storage with quota headroom before official training.

### Two-A6000 profile — 2026-09-25

| Global batch | Microbatch / GPU | Seconds / update | Global windows / second | Peak allocated / GPU |
|---:|---:|---:|---:|---:|
| **16,384** | **1,024** | **6.88** | **2,382** | **0.29 GiB** |
| 65,536 | 4,096 | 28.48 | 2,301 | 0.77 GiB |
| 65,536 | 32,768 | 28.12 | 2,330 | 4.68 GiB |

- **Starting recommendation:** global `batch_size=16384`, per-GPU `microbatch_size=1024`, two GPUs. These are now the base config values. Differences are small; this is the best measured bounded candidate, not a proven universal optimum. The smaller batch also shortens update latency; learning-rate/convergence effects were not tuned.
- Each candidate used a fresh two-GPU process, one warmup and two timed **complete optimizer updates** on new indexed real-data windows. The toy subset cap was explicitly removed, and ranks consume disjoint windows. An earlier repeated-subset screen is excluded from this table. Timing includes memmap reads, transfer, target encoder, matching, decoder backward and AdamW. It excludes startup, evaluation and checkpoint writes; OS file caching can affect measurements.
- Larger passes increased memory and brief utilization peaks, without increasing throughput. Per-rank CPU use was about one full core and sampled GPU utilization was low through most of each run. This points toward CPU matching/data handling and synchronization as the next performance targets; filling 48 GB alone will not fix it.
- At the measured throughput, one pass over all 206,968,824 valid windows extrapolates to about **24 hours**, before evaluation/checkpoint overhead. This is not a full-epoch benchmark.
- The bounded toy exited normally; no official training was launched. Production configuration is resolved: learning rate `0.001`, the toy-verified checkpoint/validation paths, and run ID `condition_b_jepa_distill_probe`. W&B is enabled at `tobieliu825/pufferdrive`, group `observation_probe`. The production schedule is five epochs per collection.

Reproduce one candidate (activate `.venv`; use available GPUs):

```bash
CUDA_VISIBLE_DEVICES=2,3 torchrun --standalone --nproc_per_node=2 \
  -m project.jepa_distill.observation_probe.profile \
  --config project/jepa_distill/config/observation_probe_toy.yaml \
  --set data.max_windows_per_collection=null \
  --batch-size 16384 --microbatch-size 1024 --warmup-steps 1 --steps 2 \
  --output /tmp/YOUR_UNIQUE_PROFILE.json
```

For the full launch, use `train_observation_probe_2gpu.sh` with the resolved `observation_probe.yaml` and ensure output storage has sufficient quota. Stored ranks remain pooled into 15 sequential collections.
