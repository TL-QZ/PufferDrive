# Fresh PQL with 64K microbatches

Copies ../one_epoch_fresh_1b with only microbatch increased from 32000 to 64000. Logical batch remains 128000, GPU replay 2048000 rows per rank, one epoch, one-third FIFO replacement, fresh PQL targets, no filtering, and 1B global received steps. Source SAC configuration is unchanged. Two GPUs; W&B disabled; local metrics and learner snapshots.
