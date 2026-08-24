#!/usr/bin/env bash
# Record the SLURM identity of the running task as one JSON file.
#
# The name carries the job and array task id, so concurrent tasks on a shared
# filesystem never write to the same path.

set -euo pipefail

label="$1"
probe_dir="${EULER_PROBE_DIR:?EULER_PROBE_DIR must be set}"
mkdir -p "$probe_dir"

array_job_id="${SLURM_ARRAY_JOB_ID:-}"
array_task_id="${SLURM_ARRAY_TASK_ID:-}"
job_id="${SLURM_JOB_ID:-nojob}"

if [ -n "$array_job_id" ]; then
    stamp="${array_job_id}_${array_task_id}"
else
    stamp="$job_id"
fi

cat > "${probe_dir}/${label}.${stamp}.json" <<JSON
{
  "label": "${label}",
  "job_id": "${job_id}",
  "array_job_id": "${array_job_id}",
  "array_task_id": "${array_task_id}",
  "partition": "${SLURM_JOB_PARTITION:-}",
  "account": "${SLURM_JOB_ACCOUNT:-}",
  "mem_per_node": "${SLURM_MEM_PER_NODE:-}",
  "mem_per_cpu": "${SLURM_MEM_PER_CPU:-}",
  "cpus_per_task": "${SLURM_CPUS_PER_TASK:-}",
  "host": "$(hostname)"
}
JSON
