"""Tests for the Euler check harness in tests/euler.

A harness that never fails is worth nothing, so each case feeds it two sets of
probe records: one a correct plugin leaves, and one it must reject.
"""

import importlib.util
import json
import os
from pathlib import Path

import pytest

CHECKER_PATH = Path(__file__).parent / "euler" / "check_results.py"
EULER_DIR = Path(__file__).parent / "euler"

SCENARIOS = [
    "partial_batch",
    "single_job",
    "partition_split",
    "account_split",
    "retry_memory",
    "array_chunking",
    "two_rules",
]


def _load_checker():
    spec = importlib.util.spec_from_file_location("euler_check", CHECKER_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


checker = _load_checker()


def _probe(label, array_job_id="", array_task_id="", **overrides):
    record = {
        "label": label,
        "job_id": overrides.pop("job_id", "9000"),
        "array_job_id": array_job_id,
        "array_task_id": array_task_id,
        "partition": "cpu",
        "account": "acct_a",
        "mem_per_node": "500",
        "mem_per_cpu": "",
        "cpus_per_task": "1",
        "host": "node01",
    }
    record.update(overrides)
    return record


def _run_routing_invariant(scenario, probes):
    report = checker.Report(scenario)
    checker.check_no_mixed_routing(probes, report)
    return report


class TestProbeShapes:
    """The checker reads exactly what probe.sh writes."""

    def test_probe_script_fields_match_checker_expectations(self, tmp_path):
        import subprocess

        env = {
            "EULER_PROBE_DIR": str(tmp_path),
            "SLURM_ARRAY_JOB_ID": "4242",
            "SLURM_ARRAY_TASK_ID": "3",
            "SLURM_JOB_PARTITION": "cpu",
            "SLURM_JOB_ACCOUNT": "acct_a",
            "SLURM_MEM_PER_NODE": "500",
            "PATH": "/usr/bin:/bin",
        }
        subprocess.run(
            ["bash", str(EULER_DIR / "probe.sh"), "some_label"],
            env=env,
            check=True,
        )

        written = list(tmp_path.glob("*.json"))
        assert len(written) == 1
        assert written[0].name == "some_label.4242_3.json"
        record = json.loads(written[0].read_text())
        assert checker.submission_of(record) == "4242"
        for key in (
            "label",
            "job_id",
            "array_job_id",
            "array_task_id",
            "partition",
            "account",
            "mem_per_node",
            "mem_per_cpu",
        ):
            assert key in record

    def test_a_plain_job_is_its_own_submission(self):
        record = _probe("solo", job_id="777")
        assert checker.submission_of(record) == "777"


class TestMixedRoutingDetection:
    """The invariant the array signature keeps."""

    def test_one_array_spanning_two_partitions_fails(self):
        probes = [
            _probe("t0", array_job_id="100", array_task_id="0", partition="cpu"),
            _probe("t1", array_job_id="100", array_task_id="1", partition="gpu"),
        ]
        report = _run_routing_invariant("partition_split", probes)
        assert report.failures

    def test_one_array_spanning_two_accounts_fails(self):
        probes = [
            _probe("t0", array_job_id="100", array_task_id="0", account="acct_a"),
            _probe("t1", array_job_id="100", array_task_id="1", account="acct_b"),
        ]
        report = _run_routing_invariant("account_split", probes)
        assert report.failures

    def test_one_array_per_partition_passes(self):
        probes = [
            _probe("t0", array_job_id="100", array_task_id="0", partition="cpu"),
            _probe("t1", array_job_id="101", array_task_id="0", partition="gpu"),
        ]
        report = _run_routing_invariant("partition_split", probes)
        assert not report.failures


class TestRoutingSplitScenario:
    def test_one_submission_for_two_partitions_fails(self):
        """One array for both partitions is the failure, however it ran."""
        probes = [
            _probe(
                f"partition_split_{i}",
                array_job_id="100",
                array_task_id=str(i),
                partition="cpu",
            )
            for i in range(4)
        ]
        wanted = {
            f"partition_split_{i}": ("cpu" if i % 2 == 0 else "gpu") for i in range(4)
        }
        report = checker.Report("partition_split")
        checker.check_routing_split(probes, report, "partition", wanted)
        assert any("went out separately: 1 submission(s)" in f for f in report.failures)

    def test_tasks_routed_as_asked_pass(self):
        probes = [
            _probe(
                f"partition_split_{i}",
                array_job_id="100" if i % 2 == 0 else "101",
                array_task_id=str(i // 2),
                partition="cpu" if i % 2 == 0 else "gpu",
            )
            for i in range(4)
        ]
        wanted = {
            f"partition_split_{i}": ("cpu" if i % 2 == 0 else "gpu") for i in range(4)
        }
        report = checker.Report("partition_split")
        checker.check_routing_split(probes, report, "partition", wanted)
        assert not report.failures

    def test_a_task_that_never_ran_fails(self):
        probes = [
            _probe("partition_split_0", array_job_id="100", array_task_id="0"),
        ]
        wanted = {"partition_split_0": "cpu", "partition_split_1": "gpu"}
        report = checker.Report("partition_split")
        checker.check_routing_split(probes, report, "partition", wanted)
        assert any("did not run" in f for f in report.failures)


class TestPartialBatchScenario:
    def test_a_deadlocked_run_fails(self):
        """Fewer tasks ran than the scenario submits."""
        probes = [
            _probe(f"partial_batch_{i}", array_job_id="100", array_task_id=str(i))
            for i in range(4)
        ]
        report = checker.Report("partial_batch")
        checker.check_partial_batch(probes, report, expected=10)
        assert report.failures

    def test_every_job_running_passes_however_it_was_batched(self):
        probes = [
            _probe(
                f"partial_batch_{i}",
                array_job_id=str(100 + i // 4),
                array_task_id=str(i % 4),
            )
            for i in range(10)
        ]
        report = checker.Report("partial_batch")
        checker.check_partial_batch(probes, report, expected=10)
        assert not report.failures


class TestSingleJobScenario:
    def test_a_lone_job_submitted_as_an_array_fails(self):
        probes = [_probe("single_job_only", array_job_id="100", array_task_id="0")]
        report = checker.Report("single_job")
        checker.check_single_job(probes, report)
        assert report.failures

    def test_a_lone_job_submitted_plainly_passes(self):
        probes = [_probe("single_job_only", job_id="100")]
        report = checker.Report("single_job")
        checker.check_single_job(probes, report)
        assert not report.failures


class TestRetryMemoryScenario:
    def test_a_retry_reusing_the_failed_memory_fails(self):
        probes = [
            _probe("retry_memory_0", array_job_id="100", array_task_id="0"),
            _probe("retry_memory_0", array_job_id="101", array_task_id="0"),
        ]
        report = checker.Report("retry_memory")
        checker.check_retry_memory(probes, report)
        assert any("different memory" in f for f in report.failures)

    def test_a_retry_sharing_its_submission_fails(self):
        probes = [
            _probe("retry_memory_0", array_job_id="100", array_task_id="0"),
            _probe(
                "retry_memory_0",
                array_job_id="100",
                array_task_id="1",
                mem_per_node="1000",
            ),
        ]
        report = checker.Report("retry_memory")
        checker.check_retry_memory(probes, report)
        assert any("its own submission" in f for f in report.failures)

    def test_a_retry_with_scaled_memory_passes(self):
        probes = [
            _probe("retry_memory_0", array_job_id="100", array_task_id="0"),
            _probe("retry_memory_0", job_id="102", mem_per_node="1000"),
            _probe("retry_memory_1", array_job_id="100", array_task_id="1"),
        ]
        report = checker.Report("retry_memory")
        checker.check_retry_memory(probes, report)
        assert not report.failures

    def test_a_sibling_retried_without_failing_fails(self):
        """Misreported array status shows up as a spurious second attempt."""
        probes = [
            _probe("retry_memory_0", array_job_id="100", array_task_id="0"),
            _probe("retry_memory_0", job_id="102", mem_per_node="1000"),
            _probe("retry_memory_1", array_job_id="100", array_task_id="1"),
            _probe("retry_memory_1", job_id="103", mem_per_node="1000"),
        ]
        report = checker.Report("retry_memory")
        checker.check_retry_memory(probes, report)
        assert any("retry_memory_1 ran once: 2" in f for f in report.failures)

    def test_a_job_that_never_retried_fails(self):
        probes = [_probe("retry_memory_0", array_job_id="100", array_task_id="0")]
        report = checker.Report("retry_memory")
        checker.check_retry_memory(probes, report)
        assert report.failures


class TestArrayChunkingScenario:
    def test_a_submission_over_the_limit_fails(self):
        probes = [
            _probe(f"array_chunking_{i}", array_job_id="100", array_task_id=str(i))
            for i in range(6)
        ]
        report = checker.Report("array_chunking")
        checker.check_array_chunking(probes, report, expected=6, limit=2)
        assert any("limit of 2 tasks" in f for f in report.failures)

    def test_chunks_at_the_limit_pass(self):
        probes = [
            _probe(
                f"array_chunking_{i}",
                array_job_id=str(100 + i // 2),
                array_task_id=str(i % 2),
            )
            for i in range(6)
        ]
        report = checker.Report("array_chunking")
        checker.check_array_chunking(probes, report, expected=6, limit=2)
        assert not report.failures


class TestTwoRulesScenario:
    def test_two_rules_in_one_array_fails(self):
        probes = [
            _probe("two_rules_alpha_0", array_job_id="100", array_task_id="0"),
            _probe("two_rules_beta_0", array_job_id="100", array_task_id="1"),
        ]
        report = checker.Report("two_rules")
        checker.check_two_rules(probes, report)
        assert report.failures

    def test_one_array_per_rule_passes(self):
        probes = [
            _probe("two_rules_alpha_0", array_job_id="100", array_task_id="0"),
            _probe("two_rules_beta_0", array_job_id="101", array_task_id="0"),
        ]
        report = checker.Report("two_rules")
        checker.check_two_rules(probes, report)
        assert not report.failures


class TestLogdirCheck:
    def test_a_missing_task_log_fails(self, tmp_path):
        probes = [_probe("t0", array_job_id="100", array_task_id="0")]
        report = checker.Report("partial_batch")
        checker.check_logdir(tmp_path, probes, report)
        assert any("slurm log 100_0.log exists" in f for f in report.failures)

    def test_a_log_per_task_passes(self, tmp_path):
        rule_dir = tmp_path / "rule_work"
        rule_dir.mkdir()
        (rule_dir / "100_0.log").write_text("")
        probes = [_probe("t0", array_job_id="100", array_task_id="0")]
        report = checker.Report("partial_batch")
        checker.check_logdir(tmp_path, probes, report)
        assert not report.failures


class TestWorkflowsParse:
    """Snakemake must be able to read every scenario workflow."""

    @pytest.mark.parametrize("scenario", SCENARIOS)
    def test_dry_run_succeeds(self, scenario, tmp_path):
        import shutil
        import subprocess

        if shutil.which("snakemake") is None:
            pytest.skip("snakemake CLI not on PATH")

        env = {
            "PATH": os.environ["PATH"],
            "HOME": str(tmp_path),
            "EULER_PROBE": str(EULER_DIR / "probe.sh"),
            "EULER_PROBE_DIR": str(tmp_path / "probes"),
            "EULER_PARTITION_A": "cpu",
            "EULER_PARTITION_B": "gpu",
            "EULER_ACCOUNT_A": "acct_a",
            "EULER_ACCOUNT_B": "acct_b",
        }
        result = subprocess.run(
            [
                "snakemake",
                "--snakefile",
                str(EULER_DIR / "workflows" / f"{scenario}.smk"),
                "--directory",
                str(tmp_path / scenario),
                "--dry-run",
                "--quiet",
            ],
            env=env,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr or result.stdout


class TestRunnerRefusesTheInstalledPlugin:
    """run_checks.sh must not test the copy already on the cluster."""

    def test_the_guard_is_present(self):
        script = (EULER_DIR / "run_checks.sh").read_text()
        assert "refusing to run" in script
        assert "PYTHONPATH" in script
