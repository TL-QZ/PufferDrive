# Enhanced Round 2 PQL

Derived from `../round2_enhance` in the multi-step-return project. This set
preserves trial 01 (the launch script also accepts alias `1`), its 4B global
received-row budget, replay ratio 0.25, adaptive TES, and other recipe settings.
Its benchmark catalog is copied byte-for-byte into this directory.

The return controls are `n_steps: 1`, `return_estimator: pql`, horizon 128,
lambda 0.95, and `soft_each_return_endpoint` entropy. The learner's inherited
`critic_entropy_in_target: true` default is pinned explicitly. W&B, learner-data,
and evaluation-catalog paths are isolated for this PQL project.

See the [PQL project guide](../../README.md) for preview commands, source parity
validation, and the horizon/memory caveat.
