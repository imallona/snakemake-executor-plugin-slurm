# One rule, two partitions chosen per wildcard. Each task runs on the
# partition it asked for.

include: "common.smk"

N = 4
PARTITION_A = os.environ["EULER_PARTITION_A"]
PARTITION_B = os.environ["EULER_PARTITION_B"]


def partition_for(wildcards):
    return PARTITION_A if int(wildcards.i) % 2 == 0 else PARTITION_B


rule all:
    input:
        expand("done/partition_split/{i}.txt", i=range(N)),


rule work:
    output:
        "done/partition_split/{i}.txt",
    resources:
        runtime=5,
        mem_mb_per_cpu=500,
        slurm_partition=partition_for,
    shell:
        probe_call("partition_split_{wildcards.i}") + "; touch {output}"
