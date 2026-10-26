#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# Defaults select six paired-rate trials on GPUs 0 through 5 with no deadline.
# Forward the native launcher's options, including --dry-run and --gpu-groups.
exec "$SCRIPT_DIR/launch_sweep.sh" \
    --config-dir project/soft_actor_critic_tuning/configs/round5 "$@"
