# Euler checks for SLURM array job dispatch

Real cluster runs that check what the plugin does with array jobs. The unit tests can only assert what the plugin tries to submit; these assert what SLURM ran. The whole battery submits about 30 short tasks.

## Isolation

Nothing here installs, upgrades, or removes anything. `run_checks.sh` puts this checkout at the front of `PYTHONPATH` so it shadows any installed copy of the plugin, and aborts if that does not take effect. All output lands under the work directory you pass in.

Run `preflight.sh` first and read the plugin path it prints.

## Running

```
./tests/euler/preflight.sh

export EULER_PARTITION_A=<a partition you may use>
export EULER_PARTITION_B=<a different one>
export EULER_ACCOUNT=<the account to charge>        # optional
export EULER_ACCOUNT_A=<one account>                # optional
export EULER_ACCOUNT_B=<a different one>            # optional

./tests/euler/run_checks.sh /cluster/scratch/$USER/slurm_plugin_checks
```

Pass scenario names to run a subset:

```
./tests/euler/run_checks.sh /cluster/scratch/$USER/checks partition_split
```

`partition_split` is skipped unless both partition variables are set, and `account_split` unless both account variables are set. Everything else runs with no configuration.

## Scenarios

| Scenario | Submits | Must hold |
| --- | --- | --- |
| `partial_batch` | 10 jobs of one rule, snakemake throttled to 4 at a time | all 10 run and the workflow does not hang. The scheduler counts a job as running once it hands it over, so a job held back to fill a larger array is never offered again |
| `single_job` | one job of a rule with array submission requested | it goes out as a plain job, not a one task array |
| `partition_split` | 4 jobs of one rule, alternating between two partitions | each task runs on the partition it asked for, and no submission spans both |
| `account_split` | 4 jobs of one rule, alternating between two accounts | the same, on the account |
| `retry_memory` | 3 jobs, one failing once, memory scaled by attempt | the retry runs with more memory and in its own submission |
| `array_chunking` | 6 jobs with `--slurm-array-limit 2` | all 6 run, over submissions of at most 2 tasks |
| `two_rules` | two array rules ready together | each rule gets its own submission |

One invariant is checked in every scenario: no single sbatch call may carry tasks with differing account or partition.

## Assertions

Each task calls `probe.sh`, which writes one JSON file recording the SLURM job id, array job and task id, partition, account and memory it got. The filename carries the job and task id, so concurrent tasks on a shared filesystem never collide.

`check_results.py` reads that directory back. Tasks sharing a `SLURM_ARRAY_JOB_ID` came from one sbatch call, which is how the checks tell grouping right from grouping wrong.

`tests/test_euler_checks.py` tests the checker itself on a laptop, with no cluster: every scenario is fed both the probe records a correct plugin leaves and records it must reject. It also dry runs each workflow, so a broken Snakefile shows up before anyone reaches for a cluster.

## Failures

`run_checks.sh` keeps `<workdir>/<scenario>/snakemake.log`, the probe files under `probes/`, and the SLURM logs under `slurm_logs/`. A `partial_batch` run that hangs rather than failing is the deadlock: interrupt it and count the probe files.
