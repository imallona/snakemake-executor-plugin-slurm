# More ready jobs than the array limit allows in one sbatch call. The tasks
# spread over several array submissions and all of them run.

include: "common.smk"

N = 6


rule all:
    input:
        expand("done/array_chunking/{i}.txt", i=range(N)),


rule work:
    output:
        "done/array_chunking/{i}.txt",
    resources:
        runtime=5,
        mem_mb_per_cpu=500,
    shell:
        probe_call("array_chunking_{wildcards.i}") + "; touch {output}"
