# Observation probe: metrics to watch

**Default W&B logging: 12 task metrics**, plus optimizer/collection/epoch progress and W&B's system telemetry. Complete diagnostics remain in each run's `metrics.jsonl` and standalone `evaluation.json`.

## What were the 201 probe scalars?

| Prefix | Question | Scalars |
|---|---|---:|
| `probe/reconstruction/` | Can the decoder recover the real future observation from its target-encoder latent? | 50 |
| `probe/prediction/` | Can it recover that future from the JEPA-predicted latent? | 50 |
| `probe/persistence/` | What happens if we simply reuse the current observation? | 50 |
| `probe/training_mean/` | What happens if we always predict the training mean? | 50 |
| `probe/windows` | How many held-out windows were evaluated? | 1 |

Each path had **25 measurements + 25 `/elements` denominators**:

- Total loss; ego and context MAE: 3 measurements.
- Partners, lanes, boundaries and traffic controls: 5 each (attribute MAE, position MAE, presence precision, presence recall, count MAE).
- Traffic-control type and state accuracy: 2 measurements.

Thus **100 of the 201 scalars were counts used to aggregate errors**, not performance scores. They are retained locally, without W&B charts.

## Selected 12 metrics

In the names below, braces list alternative path components, not literal key names.

| Category | W&B metric names | Count / interpretation |
|---|---|---|
| Training | `train/loss_total` | 1; lower means the decoder fits training targets better |
| Overall held-out quality | `probe/{reconstruction,prediction}/loss_total` | 2; lower is better; reconstruction selects `best.pt` |
| Scene geometry | `probe/{reconstruction,prediction,persistence,training_mean}/partners/position_mae_m`; `probe/prediction/{lanes,boundaries}/position_mae_m` | 6; lower is better; compare nearby-agent error against both baselines |
| Object presence | `probe/prediction/partners/{presence_precision,presence_recall}` | 2; higher is better; detects extra/missing nearby agents |
| Traffic state | `probe/prediction/traffic_controls.state/accuracy` | 1; higher is better; omitted when no valid traffic controls exist |

### Reading the curves

- Start with the two held-out total losses. Poor reconstruction limits what we can conclude about JEPA prediction; their difference is not an exact error decomposition.
- Compare predicted nearby-agent position error with reconstruction and both baselines. Baseline **total losses are excluded** because hard presence/category predictions have different confidence penalties from learned logits.
- Position MAE is the mean absolute error of matched x/y coordinates in meters, not Euclidean displacement. Matching uses all configured attributes; geometry scores alone do not measure detection quality.
- Read presence precision/recall alongside position error. Presence is measured against the slots assigned by the reconstruction matcher, not a distance-threshold object-detection benchmark. Road presence, ego/context errors and other detailed scores remain local.
- Optional final test evaluation logs the same 11 evaluation metrics under `test/`. Undefined metrics stay absent; progress and system telemetry are additional to the 12 task metrics.

## Scope of the change

- `observation_probe/monitoring.py` filters uploads after the full local event is saved. Training, loss computation, checkpoint selection and the shared Condition B logger keep their existing behavior.
- The selection takes effect in newly started processes, including resumed training. Existing W&B history/panels are not deleted or reorganized by this code change; use the names above when selecting panels for an existing workspace.
- Verification: 13 focused monitoring/trainer tests passed, including full local retention, selected remote payloads, test namespaces, resume cursors and the shared logger's existing behavior. No GPU training or live W&B run was needed.
