# Task 0 fails once and is retried with more memory. The retry must not join
# an array carrying the memory it just died on.

include: "common.smk"

N = 3
FLAG_DIR = "flags/retry_memory"


rule all:
    input:
        expand("done/retry_memory/{i}.txt", i=range(N)),


rule work:
    output:
        "done/retry_memory/{i}.txt",
    resources:
        runtime=5,
        mem_mb_per_cpu=lambda wildcards, attempt: 500 * attempt,
    params:
        flag=lambda wildcards: f"{FLAG_DIR}/{wildcards.i}",
        fails=lambda wildcards: int(wildcards.i) == 0,
    shell:
        probe_call("retry_memory_{wildcards.i}") + "; "
        "if [ {params.fails} = True ] && [ ! -f {params.flag} ]; then "
        "  mkdir -p " + FLAG_DIR + "; touch {params.flag}; exit 1; "
        "fi; "
        "touch {output}"
