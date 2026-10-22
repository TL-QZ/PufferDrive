#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd -- "$EXPERIMENT_DIR/../.." && pwd)"

usage() {
    cat <<'EOF'
Usage: launch.sh --trial 97|98 [options]

Launch one PPO-behavior SAC trial on exactly two GPUs.

Options:
  --gpus ID,ID       Two distinct nonnegative GPU IDs (default: 0,1)
  --dry-run          Prepare the run config and print the exact torchrun command
  --output-dir PATH  Use a new run directory (default: experiment-local timestamped path)
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
gpu_ids="$first_gpu_id,$second_gpu_id"

cd "$REPO_ROOT"
source "$REPO_ROOT/.venv/bin/activate"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

if [[ -z "$output_dir" ]]; then
    timestamp="$(date -u +%Y%m%dT%H%M%S%6NZ)"
    output_dir="$EXPERIMENT_DIR/runs/${timestamp}_trial_${trial_id}"
elif [[ "$output_dir" != /* ]]; then
    output_dir="$REPO_ROOT/$output_dir"
fi

python -m project.soft_actor_critic_PPO_init_behavior_policy.configuration validate --trial "$trial_id"
[[ ! -e "$output_dir" ]] || fail "refusing existing run directory: $output_dir"

prepare_command=(
    python -m project.soft_actor_critic_PPO_init_behavior_policy.configuration
    prepare --trial "$trial_id" --run-dir "$output_dir"
)
[[ "$no_wandb" == false ]] || prepare_command+=(--no-wandb)
"${prepare_command[@]}"

config_path="$output_dir/config.yaml"
command=(
    torchrun --standalone --nnodes=1 --nproc-per-node=2 --max-restarts=0
    -m pufferlib.pufferl_sac_ppo_init_behavior_policy --config "$config_path"
)
printf 'Command: CUDA_VISIBLE_DEVICES=%s OMP_NUM_THREADS=%s ' "$gpu_ids" "$OMP_NUM_THREADS"
printf '%q ' "${command[@]}"
printf '\n'

if [[ "$dry_run" == true ]]; then
    exit 0
fi

export CUDA_VISIBLE_DEVICES="$gpu_ids"
if [[ "$no_wandb" == true ]]; then
    export WANDB_MODE=disabled
fi
exec "${command[@]}"
