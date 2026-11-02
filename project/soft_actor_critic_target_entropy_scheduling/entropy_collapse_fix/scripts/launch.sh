#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../../../.." && pwd)"

usage() {
    cat <<'EOF'
Usage: launch.sh --round 2|3 [--variant constant_control|stabilized] [--trial 97|98] [native options]

Defaults: --variant stabilized, --trial 98.
Native options: --gpu-groups GROUPS, --dry-run, --no-wandb, --output-dir PATH
EOF
}

fail() {
    printf 'launch.sh: %s\n' "$1" >&2
    printf 'Try --help for usage.\n' >&2
    exit 2
}

round_id=""
variant="stabilized"
trial_id="98"
round_was_set=false
variant_was_set=false
trial_was_set=false
gpu_groups=""
gpu_groups_was_set=false
dry_run=false
no_wandb=false
output_dir=""
output_dir_was_set=false

while (($# > 0)); do
    case "$1" in
        --round)
            (($# >= 2)) || fail "--round requires 2 or 3"
            [[ "$2" != --* ]] || fail "--round requires 2 or 3"
            [[ "$round_was_set" == false ]] || fail "--round may be specified only once"
            round_id="$2"
            round_was_set=true
            shift 2
            ;;
        --variant)
            (($# >= 2)) || fail "--variant requires constant_control or stabilized"
            [[ "$2" != --* ]] || fail "--variant requires constant_control or stabilized"
            [[ "$variant_was_set" == false ]] || fail "--variant may be specified only once"
            variant="$2"
            variant_was_set=true
            shift 2
            ;;
        --trial)
            (($# >= 2)) || fail "--trial requires 97 or 98"
            [[ "$2" != --* ]] || fail "--trial requires 97 or 98"
            [[ "$trial_was_set" == false ]] || fail "--trial may be specified only once"
            trial_id="$2"
            trial_was_set=true
            shift 2
            ;;
        --gpu-groups)
            (($# >= 2)) || fail "--gpu-groups requires a group specification"
            [[ "$2" != --* ]] || fail "--gpu-groups requires a group specification"
            [[ "$gpu_groups_was_set" == false ]] || fail "--gpu-groups may be specified only once"
            gpu_groups="$2"
            gpu_groups_was_set=true
            shift 2
            ;;
        --dry-run)
            [[ "$dry_run" == false ]] || fail "--dry-run may be specified only once"
            dry_run=true
            shift
            ;;
        --no-wandb)
            [[ "$no_wandb" == false ]] || fail "--no-wandb may be specified only once"
            no_wandb=true
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
        --help)
            usage
            exit 0
            ;;
        *)
            fail "unknown argument: $1"
            ;;
    esac
done

[[ "$round_id" == 2 || "$round_id" == 3 ]] || fail "--round must be 2 or 3"
[[ "$variant" == constant_control || "$variant" == stabilized ]] || fail "--variant must be constant_control or stabilized"
[[ "$trial_id" == 97 || "$trial_id" == 98 ]] || fail "--trial must be 97 or 98"

config_dir="$PROJECT_DIR/configs/round${round_id}_${variant}"
[[ -d "$config_dir" ]] || fail "config directory does not exist: $config_dir"

launcher_args=(
    --config-dir "$config_dir"
    --trial-ids "$trial_id"
    --max-concurrent 1
)
[[ "$gpu_groups_was_set" == false ]] || launcher_args+=(--gpu-groups "$gpu_groups")
[[ "$dry_run" == false ]] || launcher_args+=(--dry-run)
[[ "$no_wandb" == false ]] || launcher_args+=(--no-wandb)

if [[ "$output_dir_was_set" == false ]]; then
    run_stamp="$(date -u +%Y%m%dT%H%M%S.%N)_$$"
    output_dir="$PROJECT_DIR/runs/round${round_id}_${variant}_trial${trial_id}_${run_stamp}"
fi
launcher_args+=(--output-dir "$output_dir")

source "$REPO_ROOT/.venv/bin/activate"
cd "$REPO_ROOT"
exec "$REPO_ROOT/project/soft_actor_critic_tuning/scripts/launch_sweep.sh" "${launcher_args[@]}"
