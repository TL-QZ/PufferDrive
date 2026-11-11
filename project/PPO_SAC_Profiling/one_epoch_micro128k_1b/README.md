# Fresh PQL with 128K microbatches

Copies ../one_epoch_fresh_1b with only microbatch increased from 32000 to 128000. Logical batch remains 128000, GPU replay 2048000 rows per rank, one epoch, one-third FIFO replacement, fresh PQL targets, no filtering, and 1B global received steps. Uses two GPUs. Source SAC configuration is unchanged. W&B is disabled; metrics and learner snapshots are local.

Validate GPU fit with a bounded probe before launching launch_2gpu.sh with a new output directory.
