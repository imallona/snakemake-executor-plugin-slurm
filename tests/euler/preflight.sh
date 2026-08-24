#!/usr/bin/env bash
# Check that an Euler run would exercise this checkout and nothing else.
#
# Submits nothing and writes nothing. Run it before run_checks.sh: the plugin
# path it prints is the one the checks would exercise.

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

echo "repo checkout: $repo_root"
echo

echo "--- plugin resolution ---"
echo "without PYTHONPATH (what your environment normally uses):"
python -c 'import snakemake_executor_plugin_slurm as m; print("  ", m.__file__)' \
    2>/dev/null || echo "   not importable"

echo "with PYTHONPATH=$repo_root (what run_checks.sh uses):"
PYTHONPATH="$repo_root${PYTHONPATH:+:$PYTHONPATH}" python -c \
    'import snakemake_executor_plugin_slurm as m; print("  ", m.__file__)'

resolved="$(PYTHONPATH="$repo_root${PYTHONPATH:+:$PYTHONPATH}" python -c \
    'import snakemake_executor_plugin_slurm as m; print(m.__file__)')"
case "$resolved" in
    "$repo_root"/*)
        echo "  -> the checkout shadows the installed plugin, as intended"
        ;;
    *)
        echo "  -> WARNING: the installed plugin still wins. The checks would"
        echo "     test the installed copy, not this branch."
        exit 1
        ;;
esac
echo

echo "--- snakemake ---"
python -c 'import snakemake; print("  version", snakemake.__version__)'
echo

echo "--- slurm ---"
for tool in sbatch sacct squeue sinfo scontrol; do
    printf "  %-9s %s\n" "$tool" "$(command -v "$tool" || echo "MISSING")"
done
echo
echo "  MaxArraySize: $(scontrol show config 2>/dev/null \
    | awk '/MaxArraySize/ {print $3}' || echo unknown)"
echo

echo "--- partitions you may submit to ---"
sinfo -h -o "  %P  state=%a  nodes=%D  maxtime=%l" 2>/dev/null \
    || echo "  sinfo failed"
echo

echo "--- accounts you may charge ---"
sacctmgr -n -P show assoc user="$USER" format=Account,Partition 2>/dev/null \
    | sort -u | sed 's/^/  /' || echo "  sacctmgr unavailable"
echo

cat <<'HINT'
Pick two partitions from the list above and export them, then run the checks:

  export EULER_PARTITION_A=<one>
  export EULER_PARTITION_B=<another>
  export EULER_ACCOUNT_A=<one>     # optional, for the account check
  export EULER_ACCOUNT_B=<another> # optional
  ./tests/euler/run_checks.sh /cluster/scratch/$USER/slurm_plugin_checks
HINT
