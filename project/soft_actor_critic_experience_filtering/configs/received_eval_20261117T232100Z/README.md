# Received-row evaluation round, 2026-11-17

Copies of the clipped TES and PQL trial-98 recipes. Training settings are unchanged; evaluation uses global received agent rows every 32,768,000 instead of every two collection cycles. Both runs retain 4B budgets, GPU replay, one epoch, one-third replacement, and shared actor/critic TD-error filtering.

TES uses GPUs 0,1 with logical minibatches 128,000 and microbatches 64,000. PQL uses GPUs 2,3 with logical minibatches and microbatches 64,000, preserving the respective source recipes.

Launch scripts are `../../launch_tes_received_eval_20261117T232100Z.sh` and `../../launch_pql_received_eval_20261117T232100Z.sh` relative to this directory. They enable online W&B. Existing output directories are intentionally not reusable.

Tmux session: `received_eval_20261117T232100Z`, windows `tes` and `pql`.

Local artifacts: `project/soft_actor_critic_experience_filtering/runs/received_eval_20261117T232100Z/{tes_live,pql_live}/trial_98/`.

W&B: https://wandb.ai/tobieliu825/pufferdrive_vanilla_sac_tuning/runs/tes_live_trial_98 and https://wandb.ai/tobieliu825/pufferdrive_vanilla_sac_tuning/runs/pql_live_trial_98 .
