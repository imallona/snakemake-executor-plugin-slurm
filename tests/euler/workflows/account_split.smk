# The same on the account. Needs two accounts you may charge; skipped when
# only one is configured.

include: "common.smk"

N = 4
ACCOUNT_A = os.environ["EULER_ACCOUNT_A"]
ACCOUNT_B = os.environ["EULER_ACCOUNT_B"]


def account_for(wildcards):
    return ACCOUNT_A if int(wildcards.i) % 2 == 0 else ACCOUNT_B


rule all:
    input:
        expand("done/account_split/{i}.txt", i=range(N)),


rule work:
    output:
        "done/account_split/{i}.txt",
    resources:
        runtime=5,
        mem_mb_per_cpu=500,
        slurm_account=account_for,
    shell:
        probe_call("account_split_{wildcards.i}") + "; touch {output}"
