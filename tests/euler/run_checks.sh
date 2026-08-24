#!/usr/bin/env bash
# Run the Euler checks for SLURM array job dispatch.
#
# Installs nothing. The checkout goes on PYTHONPATH so it shadows any installed
# copy of the plugin, and the run aborts if that does not take effect. All
# output lands under the work directory you pass in.
#
# Usage:
#   ./run_checks.sh <workdir> [scenario ...]
#
# With no scenario names, every scenario its prerequisites allow is run.

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
euler_dir="$repo_root/tests/euler"

if [ $# -lt 1 ]; then
    echo "usage: $0 <workdir> [scenario ...]" >&2
    exit 2
fi

workdir="$1"
shift

export PYTHONPATH="$repo_root${PYTHONPATH:+:$PYTHONPATH}"
export EULER_PROBE="$euler_dir/probe.sh"
export EULER_SLEEP="${EULER_SLEEP:-5}"

resolved="$(python -c \
    'import snakemake_executor_plugin_slurm as m; print(m.__file__)')"
case "$resolved" in
    "$repo_root"/*) ;;
    *)
        echo "refusing to run: snakemake would load $resolved," >&2
        echo "not this checkout. Run preflight.sh and fix the environment." >&2
        exit 1
        ;;
esac
echo "plugin under test: $resolved"

mkdir -p "$workdir"
workdir="$(cd "$workdir" && pwd)"
echo "work directory: $workdir"
echo

# Scenario name -> extra snakemake arguments. In partial_batch --jobs sits
# below the number of ready jobs, which is the throttling under test.
declare -A scenario_args=(
    [partial_batch]="--jobs 4 --slurm-array-jobs=all"
    [single_job]="--jobs 4 --slurm-array-jobs=all"
    [partition_split]="--jobs 8 --slurm-array-jobs=all"
    [account_split]="--jobs 8 --slurm-array-jobs=all"
    [retry_memory]="--jobs 8 --slurm-array-jobs=all --retries 1"
    [array_chunking]="--jobs 8 --slurm-array-jobs=all --slurm-array-limit 2"
    [two_rules]="--jobs 8 --slurm-array-jobs=all"
)

all_scenarios=(
    partial_batch
    single_job
    partition_split
    account_split
    retry_memory
    array_chunking
    two_rules
)

if [ $# -gt 0 ]; then
    scenarios=("$@")
else
    scenarios=("${all_scenarios[@]}")
fi

skip_reason() {
    case "$1" in
        partition_split)
            if [ -z "${EULER_PARTITION_A:-}" ] || [ -z "${EULER_PARTITION_B:-}" ]; then
                echo "EULER_PARTITION_A and EULER_PARTITION_B are not both set"
            fi
            ;;
        account_split)
            if [ -z "${EULER_ACCOUNT_A:-}" ] || [ -z "${EULER_ACCOUNT_B:-}" ]; then
                echo "EULER_ACCOUNT_A and EULER_ACCOUNT_B are not both set"
            fi
            ;;
    esac
}

failed=()
skipped=()

for scenario in "${scenarios[@]}"; do
    if [ -z "${scenario_args[$scenario]+set}" ]; then
        echo "unknown scenario: $scenario" >&2
        exit 2
    fi

    reason="$(skip_reason "$scenario")"
    if [ -n "$reason" ]; then
        echo "== $scenario: skipped ($reason)"
        skipped+=("$scenario")
        echo
        continue
    fi

    run_dir="$workdir/$scenario"
    probe_dir="$run_dir/probes"
    log_dir="$run_dir/slurm_logs"
    rm -rf "$run_dir"
    mkdir -p "$probe_dir" "$log_dir"
    export EULER_PROBE_DIR="$probe_dir"

    default_resources=()
    if [ -n "${EULER_ACCOUNT:-}" ]; then
        default_resources=(--default-resources "slurm_account=$EULER_ACCOUNT")
    fi

    echo "== $scenario"
    # shellcheck disable=SC2086
    if snakemake \
        --snakefile "$euler_dir/workflows/$scenario.smk" \
        --directory "$run_dir" \
        --executor slurm \
        --slurm-logdir "$log_dir" \
        --latency-wait 60 \
        "${default_resources[@]+"${default_resources[@]}"}" \
        ${scenario_args[$scenario]} \
        > "$run_dir/snakemake.log" 2>&1
    then
        echo "  snakemake finished"
    else
        echo "  snakemake exited non-zero, see $run_dir/snakemake.log"
        failed+=("$scenario")
    fi

    if ! python "$euler_dir/check_results.py" "$scenario" "$probe_dir" \
        --logdir "$log_dir" \
        --partition-a "${EULER_PARTITION_A:-}" \
        --partition-b "${EULER_PARTITION_B:-}" \
        --account-a "${EULER_ACCOUNT_A:-}" \
        --account-b "${EULER_ACCOUNT_B:-}"
    then
        failed+=("$scenario")
    fi
    echo
done

echo "================================"
if [ ${#skipped[@]} -gt 0 ]; then
    echo "skipped: ${skipped[*]}"
fi
if [ ${#failed[@]} -gt 0 ]; then
    # A scenario can be listed twice, once for snakemake and once for the checks.
    echo "failed: $(printf '%s\n' "${failed[@]}" | sort -u | tr '\n' ' ')"
    exit 1
fi
echo "all checks passed"
