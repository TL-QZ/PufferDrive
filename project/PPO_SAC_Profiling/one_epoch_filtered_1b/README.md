# One epoch and one-third refresh with EWMA filtering

Controlled derivative of `../one_epoch_1b/`: identical source configuration,
1B global received-step budget, seed 0, one epoch, one-third FIFO replay refresh,
2.048M replay rows per rank, 128k logical / 32k microbatch, compiled models,
PQL horizon 128 / lambda .95, fixed critic-target and actor-Q caches, GPU replay,
and SAC environment/optimizer/entropy settings.

The only training change is enabling `target_advantage` scoring and
`ewma_threshold` selection with beta .25 and threshold scale .01. Scores are
absolute cached PQL target minus cycle-start expected online twin-min Q.
Keep scores >= .01 times the rank-local EWMA maximum, synchronize counts across
ranks with MIN, and train the surviving rows once. No minimum retention or padding.
This is sample filtering, not PPO policy-ratio clipping or loss-value clamping.

Default GPUs are 6,7 so the concurrent unfiltered comparison on GPUs 4,5 is
uninterrupted. Both pairs are A6000s, but concurrent jobs share CPU and storage;
absolute timing comparisons must account for possible host contention.
Snapshots, local training metrics and stopped-at-budget behavior match the
unfiltered run. W&B remains disabled. Generated outputs remain ignored.

```bash
CUDA_VISIBLE_DEVICES=6,7 bash project/PPO_SAC_Profiling/one_epoch_filtered_1b/launch_2gpu.sh \
  project/PPO_SAC_Profiling/results/pql_one_epoch_filtered_1b_UNIQUE_NAME
```
