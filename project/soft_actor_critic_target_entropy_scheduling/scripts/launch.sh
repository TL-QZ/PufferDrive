#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

usage() {
    cat <<'EOF'
Usage: launch.sh --trial 97|98 [options]

Launch one target-entropy-scheduling candidate on exactly two GPUs. Trial IDs
preserve the round-three grid mapping. Each invocation launches one process group.

Options:
  --gpus ID,ID       Two distinct nonnegative GPU IDs (default: 0,1)
  --dry-run          Resolve configuration and print the command without training
  --output-dir PATH  Create the sweep output in PATH (default: isolated runs/ path)
  --no-wandb         Disable remote and local experiment loggers
  --help             Show this help
EOF
}

fail() {
    printf 'launch.sh: %s\n' "$1" >&2
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

trial_id=""
trial_was_set=false
gpu_ids="0,1"
gpu_ids_were_set=false
dry_run=false
output_dir=""
output_dir_was_set=false
no_wandb=false

while (($# > 0)); do
    case "$1" in
        --trial)
            (($# >= 2)) || fail "--trial requires 97 or 98"
            [[ "$2" != --* ]] || fail "--trial requires 97 or 98"
            [[ "$trial_was_set" == false ]] || fail "--trial may be specified only once"
            [[ -n "$2" ]] || fail "--trial requires 97 or 98"
            trial_id="$2"
            trial_was_set=true
            shift 2
            ;;
        --gpus)
            (($# >= 2)) || fail "--gpus requires two comma-separated GPU IDs"
            [[ "$2" != --* ]] || fail "--gpus requires two comma-separated GPU IDs"
            [[ "$gpu_ids_were_set" == false ]] || fail "--gpus may be specified only once"
            gpu_ids="$2"
            gpu_ids_were_set=true
            shift 2
            ;;
        --dry-run)
            [[ "$dry_run" == false ]] || fail "--dry-run may be specified only once"
            dry_run=true
            shift
            ;;
        --output-dir)
            (($# >= 2)) || fail "--output-dir requires a path"
            [[ "$2" != --* ]] || fail "--output-dir requires a path"
            [[ "$output_dir_was_set" == false ]] || fail "--output-dir may be specified only once"
            [[ -n "$2" ]] || fail "--output-dir requires a path"
            output_dir="$2"
            output_dir_was_set=true
            shift 2
            ;;
        --no-wandb)
            [[ "$no_wandb" == false ]] || fail "--no-wandb may be specified only once"
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

[[ -n "$trial_id" ]] || fail "--trial is required (choose 97 or 98)"
[[ "$trial_id" == 97 || "$trial_id" == 98 ]] || fail "--trial must be 97 or 98"

if [[ ! "$gpu_ids" =~ ^[0-9]+,[0-9]+$ ]]; then
    fail "--gpus must contain exactly two nonnegative integer IDs, such as 0,1"
fi
IFS=, read -r first_gpu_id second_gpu_id <<< "$gpu_ids"
first_gpu_id="$(normalize_gpu_id "$first_gpu_id")"
second_gpu_id="$(normalize_gpu_id "$second_gpu_id")"
[[ "$first_gpu_id" != "$second_gpu_id" ]] || fail "--gpus IDs must be distinct"

launcher_args=(
    --config-dir project/soft_actor_critic_target_entropy_scheduling/configs
    --trial-ids "$trial_id"
    --gpu-groups "$first_gpu_id,$second_gpu_id"
    --max-concurrent 1
)
[[ "$dry_run" == true ]] && launcher_args+=(--dry-run)
if [[ -z "$output_dir" ]]; then
    run_stamp="$(date -u +%Y%m%dT%H%M%S.%N)_$$"
    output_dir="$SCRIPT_DIR/../runs/target_entropy_scheduling_${run_stamp}"
fi
launcher_args+=(--output-dir "$output_dir")
[[ "$no_wandb" == true ]] && launcher_args+=(--no-wandb)

exec "$SCRIPT_DIR/../../soft_actor_critic_tuning/scripts/launch_sweep.sh" "${launcher_args[@]}"
