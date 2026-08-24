# A rule with one job while array submission is requested. It goes out as a
# plain job.

include: "common.smk"


rule all:
    input:
        "done/single_job/only.txt",


rule work:
    output:
        "done/single_job/only.txt",
    resources:
        runtime=5,
        mem_mb=500,
    shell:
        probe_call("single_job_only") + "; touch {output}"
