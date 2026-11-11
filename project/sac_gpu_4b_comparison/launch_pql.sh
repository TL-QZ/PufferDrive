#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
source "$REPO_ROOT/.venv/bin/activate"
cd "$REPO_ROOT"

usage() {
    cat <<'EOF'
Usage: launch_pql.sh [--dry-run] [--output-dir PATH] [--no-wandb]

Launch trial 98 on GPUs 2,3 with one concurrent trial. W&B is online by default.
EOF
}

fail() {
    printf 'launch_pql.sh: %s\n' "$1" >&2
    exit 2
}

run_stamp="$(date -u +%Y%m%dT%H%M%SZ)_$$"
output_dir="$SCRIPT_DIR/runs/pql_official_round2_gpu_1ep_highlr_${run_stamp}"
launcher_args=(
    --config-dir project/sac_gpu_4b_comparison/configs/pql
    --trial-ids 98
    --gpu-groups 2,3
    --max-concurrent 1
)

while (($# > 0)); do
    case "$1" in
        --dry-run)
            launcher_args+=(--dry-run)
            shift
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
            fail "unknown argument: $1" >&2
            ;;
    esac
done

export WANDB_MODE=online
exec "$REPO_ROOT/project/soft_actor_critic_tuning/scripts/launch_sweep.sh" \
    "${launcher_args[@]}" --output-dir "$output_dir"
