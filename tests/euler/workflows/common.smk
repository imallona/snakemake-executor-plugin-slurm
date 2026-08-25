# Shared bits for the Euler check workflows.
#
# Each cluster job records what SLURM gave it by calling probe.sh.
# check_results.py asserts on those probe files, not on the snakemake log.

import os

PROBE = os.environ["EULER_PROBE"]
SLEEP = os.environ.get("EULER_SLEEP", "5")


def probe_call(label):
    return f"bash {PROBE} {label}; sleep {SLEEP}"
