# Fresh-target PQL, one epoch and one-third refresh

This experiment copies `../one_epoch_1b/` and changes only the training variant
from `fixed_targets_actor_q` to `stock`. Source YAML is byte-identical. Retained:
GPU replay, no filtering, one epoch per cycle, one-third FIFO refresh, seed 0,
1B global received-step budget, two A6000 GPUs, 2.048M replay rows per rank,
128k logical / 32k microbatch, 20 environments x 3200 agent slots per rank,
PQL horizon 128/lambda .95, compilation, and SAC optimizer/entropy settings.

Stock PQL computes targets before each logical minibatch using current actor,
alpha and target critics. Its actor loss uses the current online critics after
the critic update. There are no cycle-wide target or actor-Q caches. This also
restores stock PQL's packed-sequence training batches, whereas the cached variant
shuffles individual rows after target construction. This is a stock-vs-cached
implementation comparison, not an isolated target-refresh-order ablation.

Default GPUs 0,1 avoid the concurrent cached unfiltered run on 4,5 and cached
filtered run on 6,7. All use A6000s; shared host CPU/storage contention can affect
timing. Compare the same first three non-warmup cycles, and later compare longer
received-step windows. Do not compare cycle time without received-row counts.

Outputs are local with W&B disabled; periodic/final learner snapshots follow the
other runs and do not support exact environment/replay resume. Warmup contributes
to the 1B budget but is excluded from steady-state timing summaries.

```bash
CUDA_VISIBLE_DEVICES=0,1 bash project/PPO_SAC_Profiling/one_epoch_fresh_1b/launch_2gpu.sh \
  project/PPO_SAC_Profiling/results/pql_one_epoch_fresh_1b_UNIQUE_NAME
```
