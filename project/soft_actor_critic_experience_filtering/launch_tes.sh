#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
source "$REPO_ROOT/.venv/bin/activate"
cd "$REPO_ROOT"

usage() {
    cat <<'HELP'
Usage: launch_tes.sh [--dry-run] [--gpu-groups LIST] [--output-dir PATH] [--no-wandb]

Launch trial 98 on GPUs 0,1 by default with one concurrent trial. W&B is online by default.
HELP
}

fail() {
    printf 'launch_tes.sh: %s\n' "$1" >&2
    exit 2
}

run_stamp="$(date -u +%Y%m%dT%H%M%SZ)_$$"
output_dir="$SCRIPT_DIR/runs/tes_actor_critic_filtering_${run_stamp}"
launcher_args=(
    --config-dir project/soft_actor_critic_experience_filtering/configs/tes
    --trial-ids 98
    --gpu-groups 0,1
    --max-concurrent 1
)

while (($# > 0)); do
    case "$1" in
        --dry-run)
            launcher_args+=(--dry-run)
            shift
            ;;
        --gpu-groups)
            (($# >= 2)) || fail "--gpu-groups requires a list"
            [[ "$2" != --* ]] || fail "--gpu-groups requires a list"
            launcher_args+=(--gpu-groups "$2")
            shift 2
            ;;
        --output-dir)
            (($# >= 2)) || fail "--output-dir requires a path"
            [[ "$2" != --* ]] || fail "--output-dir requires a path"
            output_dir="$2"
            shift 2
            ;;
        --no-wandb)
            launcher_args+=(--no-wandb)
            shift
            ;;
        --help)
            usage
            exit 0
            ;;
        *)
            fail "unknown argument: $1"
            ;;
    esac
done

export WANDB_MODE=online
exec "$REPO_ROOT/project/soft_actor_critic_tuning/scripts/launch_sweep.sh" \
    "${launcher_args[@]}" --output-dir "$output_dir"
