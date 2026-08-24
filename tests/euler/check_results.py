#!/usr/bin/env python3
"""Assert what SLURM did with the tasks of one Euler check.

Reads the probe files written by probe.sh, not the snakemake log, so the
assertions cover the jobs that ran rather than the ones the plugin meant to
submit.
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


def load_probes(probe_dir):
    probes = []
    for path in sorted(Path(probe_dir).glob("*.json")):
        with open(path) as handle:
            probes.append(json.load(handle))
    return probes


def submission_of(probe):
    """The sbatch call a task came from.

    Array tasks share SLURM_ARRAY_JOB_ID; a plain job has only its job id.
    """
    return probe["array_job_id"] or probe["job_id"]


def group_by_submission(probes):
    groups = defaultdict(list)
    for probe in probes:
        groups[submission_of(probe)].append(probe)
    return groups


class Report:
    def __init__(self, scenario):
        self.scenario = scenario
        self.failures = []
        self.notes = []

    def check(self, condition, requirement, observed=None):
        """requirement is what must hold; observed is what was seen."""
        line = requirement if observed is None else f"{requirement}: {observed}"
        if condition:
            self.notes.append(f"ok: {line}")
        else:
            self.failures.append(line)

    def note(self, message):
        self.notes.append(f"info: {message}")

    def summary(self):
        for line in self.notes:
            print(f"  {line}")
        for line in self.failures:
            print(f"  FAIL: {line}")
        verdict = "PASS" if not self.failures else "FAIL"
        print(f"{verdict}: {self.scenario}")
        return not self.failures


def check_no_mixed_routing(probes, report):
    """No single sbatch call may hold tasks of differing account or partition.

    Checked in every scenario, not only the ones that set routing.
    """
    for submission, members in group_by_submission(probes).items():
        partitions = {p["partition"] for p in members}
        accounts = {p["account"] for p in members}
        report.check(
            len(partitions) == 1,
            f"submission {submission} uses one partition",
            sorted(partitions),
        )
        report.check(
            len(accounts) == 1,
            f"submission {submission} uses one account",
            sorted(accounts),
        )


def check_partial_batch(probes, report, expected=10):
    labels = {p["label"] for p in probes}
    report.check(
        len(labels) == expected,
        f"all {expected} tasks ran",
        len(labels),
    )
    report.note(f"tasks spread over {len(group_by_submission(probes))} submissions")


def check_single_job(probes, report):
    report.check(len(probes) == 1, "one task ran", len(probes))
    if probes:
        report.check(
            probes[0]["array_task_id"] == "",
            "the lone job went out as a plain job, not an array task",
            f"array_task_id={probes[0]['array_task_id']!r}",
        )


def check_routing_split(probes, report, field, expected_by_label):
    """Tasks asking for different accounts or partitions go out separately.

    Only the split is asserted. Where a task then runs is the cluster's call:
    Euler derives the partition from the resources and ignores the request.
    """
    by_label = {p["label"]: p for p in probes}
    report.check(
        len(by_label) == len(expected_by_label),
        f"all {len(expected_by_label)} tasks ran",
        len(by_label),
    )
    for label, wanted in expected_by_label.items():
        probe = by_label.get(label)
        if probe is None:
            report.failures.append(f"task {label} did not run")
            continue
        report.note(f"{label} asked for {wanted!r}, ran on {probe[field]!r}")
    submissions = {submission_of(p) for p in probes}
    report.check(
        len(submissions) >= 2,
        f"tasks needing different {field}s went out separately",
        f"{len(submissions)} submission(s)",
    )


def check_retry_memory(probes, report, retried_label="retry_memory_0"):
    attempts = [p for p in probes if p["label"] == retried_label]
    report.check(
        len(attempts) == 2,
        f"{retried_label} ran twice",
        len(attempts),
    )
    if len(attempts) != 2:
        return
    memories = {p["mem_per_node"] or p["mem_per_cpu"] for p in attempts}
    report.check(
        len(memories) == 2,
        "the retry ran with different memory than the failed attempt",
        sorted(memories),
    )
    submissions = {submission_of(p) for p in attempts}
    report.check(
        len(submissions) == 2,
        "the retry had its own submission",
        f"{len(submissions)} submission(s)",
    )
    report.note(f"attempt memories: {sorted(memories)}")

    # A task that was not meant to fail must not be reported failed either.
    # Misreported array status shows up here as a spurious second attempt.
    runs = Counter(p["label"] for p in probes)
    for label, count in sorted(runs.items()):
        if label == retried_label:
            continue
        report.check(count == 1, f"{label} ran once", count)


def check_array_chunking(probes, report, expected=6, limit=2):
    labels = {p["label"] for p in probes}
    report.check(
        len(labels) == expected,
        f"all {expected} tasks ran",
        len(labels),
    )
    groups = group_by_submission(probes)
    for submission, members in groups.items():
        report.check(
            len(members) <= limit,
            f"submission {submission} stayed within the limit of {limit} tasks",
            len(members),
        )
    report.note(f"{len(labels)} tasks over {len(groups)} submissions")


def check_two_rules(probes, report):
    for submission, members in group_by_submission(probes).items():
        rules = {p["label"].rsplit("_", 1)[0] for p in members}
        report.check(
            len(rules) == 1,
            f"submission {submission} holds one rule",
            sorted(rules),
        )


def check_logdir(logdir, probes, report):
    """Every array task leaves its own SLURM log under rule_<name>/."""
    logs = list(Path(logdir).rglob("*.log"))
    report.note(f"{len(logs)} slurm log files under {logdir}")
    for probe in [p for p in probes if p["array_task_id"]]:
        wanted = f"{probe['array_job_id']}_{probe['array_task_id']}.log"
        report.check(
            any(log.name == wanted for log in logs),
            f"slurm log {wanted} exists",
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("scenario")
    parser.add_argument("probe_dir")
    parser.add_argument("--logdir")
    parser.add_argument("--partition-a")
    parser.add_argument("--partition-b")
    parser.add_argument("--account-a")
    parser.add_argument("--account-b")
    args = parser.parse_args()

    probes = load_probes(args.probe_dir)
    report = Report(args.scenario)
    report.note(f"{len(probes)} probe records")
    if not probes:
        report.failures.append("no task recorded anything; nothing ran")
        return 0 if report.summary() else 1

    check_no_mixed_routing(probes, report)

    if args.scenario == "partial_batch":
        check_partial_batch(probes, report)
    elif args.scenario == "single_job":
        check_single_job(probes, report)
    elif args.scenario == "partition_split":
        wanted = {
            f"partition_split_{i}": (
                args.partition_a if i % 2 == 0 else args.partition_b
            )
            for i in range(4)
        }
        check_routing_split(probes, report, "partition", wanted)
    elif args.scenario == "account_split":
        wanted = {
            f"account_split_{i}": (args.account_a if i % 2 == 0 else args.account_b)
            for i in range(4)
        }
        check_routing_split(probes, report, "account", wanted)
    elif args.scenario == "retry_memory":
        check_retry_memory(probes, report)
    elif args.scenario == "array_chunking":
        check_array_chunking(probes, report)
    elif args.scenario == "two_rules":
        check_two_rules(probes, report)
    else:
        report.failures.append(f"unknown scenario {args.scenario}")

    if args.logdir:
        check_logdir(args.logdir, probes, report)

    return 0 if report.summary() else 1


if __name__ == "__main__":
    sys.exit(main())
