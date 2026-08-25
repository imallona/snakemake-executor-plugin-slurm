# The scheduler hands over fewer jobs than the array chunk size. Run with
# fewer --jobs than N; all N must still run.

include: "common.smk"

N = 10


rule all:
    input:
        expand("done/partial_batch/{i}.txt", i=range(N)),


rule work:
    output:
        "done/partial_batch/{i}.txt",
    resources:
        runtime=5,
        mem_mb_per_cpu=500,
    shell:
        probe_call("partial_batch_{wildcards.i}") + "; touch {output}"
