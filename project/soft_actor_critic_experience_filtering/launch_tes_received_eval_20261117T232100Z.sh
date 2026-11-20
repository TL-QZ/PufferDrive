#!/usr/bin/env bash
set -euo pipefail
cd /common/users/ql236/auto_driving/PufferDrive
export WANDB_MODE=online
exec bash project/soft_actor_critic_tuning/scripts/launch_sweep.sh --config-dir project/soft_actor_critic_experience_filtering/configs/received_eval_20261117T232100Z/tes --trial-ids 98 --gpu-groups 0,1 --max-concurrent 1 --output-dir project/soft_actor_critic_experience_filtering/runs/received_eval_20261117T232100Z/tes_live
