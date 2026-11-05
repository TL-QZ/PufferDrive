#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd -- "$PROJECT_DIR/../.." && pwd)"

usage() {
    cat <<'EOF'
Usage: launch_PQL_official_round2.sh --trial 98|01 [options]

Preview or launch one PQL official Round 2 run.

Options:
  --gpus ID,ID       Two distinct nonnegative GPU IDs (default: 0,1)
  --dry-run          Resolve configuration and print the command without training
  --output-dir PATH  Sweep output directory (default: timestamped under runs/PQL_official_round2)
  --no-wandb         Disable remote and local experiment loggers
  --help             Show this help
EOF
}

fail() {
    printf 'launch_PQL_official_round2.sh: %s\n' "$1" >&2
    printf 'Try --help for usage.\n' >&2
    exit 2
}

normalize_gpu_id() {
    local gpu_id="$1"
    while [[ ${#gpu_id} -gt 1 && "${gpu_id:0:1}" == "0" ]]; do
        gpu_id="${gpu_id:1}"
    done
    printf '%s' "$gpu_id"
}

trial_selector=""
trial_was_set=false
gpu_ids="0,1"
output_dir=""
output_dir_was_set=false
dry_run=false
no_wandb=false

while (($# > 0)); do
    case "$1" in
        --trial)
            (($# >= 2)) || fail "--trial requires 98 or 01"
            [[ "$2" != --* ]] || fail "--trial requires 98 or 01"
            [[ "$trial_was_set" == false ]] || fail "--trial may be specified only once"
            trial_selector="$2"
            trial_was_set=true
            shift 2
            ;;
        --gpus)
            (($# >= 2)) || fail "--gpus requires two comma-separated GPU IDs"
            [[ "$2" != --* ]] || fail "--gpus requires two comma-separated GPU IDs"
            gpu_ids="$2"
            shift 2
            ;;
        --dry-run)
            dry_run=true
            shift
            ;;
        --output-dir)
            (($# >= 2)) || fail "--output-dir requires a path"
            [[ "$2" != --* && -n "$2" ]] || fail "--output-dir requires a path"
            [[ "$output_dir_was_set" == false ]] || fail "--output-dir may be specified only once"
            output_dir="$2"
            output_dir_was_set=true
            shift 2
            ;;
        --no-wandb)
            no_wandb=true
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

case "$trial_selector" in
    98)
        trial_id=98
        config_name=trial_98
        code_name=98
        ;;
    01|1)
        trial_id=1
        config_name=trial_01
        code_name=01
        ;;
    *)
        fail "--trial must be 98 or 01"
        ;;
esac

if [[ ! "$gpu_ids" =~ ^[0-9]+,[0-9]+$ ]]; then
    fail "--gpus must contain exactly two nonnegative integer IDs, such as 0,1"
fi
IFS=, read -r first_gpu_id second_gpu_id <<< "$gpu_ids"
first_gpu_id="$(normalize_gpu_id "$first_gpu_id")"
second_gpu_id="$(normalize_gpu_id "$second_gpu_id")"
[[ "$first_gpu_id" != "$second_gpu_id" ]] || fail "--gpus IDs must be distinct"

if [[ "$output_dir_was_set" == false ]]; then
    timestamp="$(date -u +%Y%m%dT%H%M%SZ%N)"
    output_dir="$PROJECT_DIR/runs/PQL_official_round2/run_${code_name}_$timestamp"
fi

launcher_args=(
    --config-dir "project/soft_actor_critic_multi_step_return_PQL/configs/PQL_official_round2/$config_name"
    --trial-ids "$trial_id"
    --gpu-groups "$first_gpu_id,$second_gpu_id"
    --max-concurrent 1
    --output-dir "$output_dir"
)
[[ "$dry_run" == true ]] && launcher_args+=(--dry-run)
[[ "$no_wandb" == true ]] && launcher_args+=(--no-wandb)

exec "$REPO_ROOT/project/soft_actor_critic_tuning/scripts/launch_sweep.sh" "${launcher_args[@]}"
