# Two array rules ready in the same dispatch. Each gets its own array
# submission.

include: "common.smk"

N = 3


rule all:
    input:
        expand("done/two_rules/alpha_{i}.txt", i=range(N)),
        expand("done/two_rules/beta_{i}.txt", i=range(N)),


rule alpha:
    output:
        "done/two_rules/alpha_{i}.txt",
    resources:
        runtime=5,
        mem_mb=500,
    shell:
        probe_call("two_rules_alpha_{wildcards.i}") + "; touch {output}"


rule beta:
    output:
        "done/two_rules/beta_{i}.txt",
    resources:
        runtime=5,
        mem_mb=500,
    shell:
        probe_call("two_rules_beta_{wildcards.i}") + "; touch {output}"
