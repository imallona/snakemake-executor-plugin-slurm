"""Unit tests for SLURM array-job functionality.

These tests do NOT require a live SLURM cluster and do NOT subclass
TestWorkflows. They cover:

  - ExecutorSettings array-job field defaults and parsing  (TestArrayJobsSettings)
  - run_jobs() dispatch routing                            (TestRunJobsRouting)
  - run_array_jobs() sbatch construction and chunking      (TestRunArrayJobs)
  - _status_lookup_ids() helper edge cases                 (TestStatusLookupIds)
  - check_active_jobs() status resolution for array tasks  (TestCheckActiveArrayJobs)
"""

import asyncio
import base64
import json
import re
import zlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from snakemake_executor_plugin_slurm import (
    Executor,
    ExecutorSettings,
    _status_lookup_ids,
)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


class _Resources(dict):
    """Dict-like resources with attribute access for known keys only."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as e:
            raise AttributeError(name) from e


def _make_mock_job(
    rule_name="myrule",
    name=None,
    wildcards=None,
    jobid=1,
    is_group=False,
    **resources,
):
    """Return a minimal mock job compatible with run_jobs / run_array_jobs."""
    mock_resources = _Resources(resources)

    mock_rule = MagicMock()
    mock_rule.name = rule_name

    job = MagicMock()
    job.resources = mock_resources
    job.rule = mock_rule
    job.name = name if name is not None else rule_name
    job.wildcards = wildcards if wildcards is not None else {}
    job.is_group.return_value = is_group
    job.threads = resources.get("threads", 1)
    job.jobid = jobid
    return job


def _make_executor_stub(array_jobs=None, array_limit=100):
    """Return a minimal Executor stub (bypasses __post_init__ entirely)."""
    executor = Executor.__new__(Executor)
    executor.logger = MagicMock()
    executor.run_uuid = "test-run-uuid"
    executor._fallback_account_arg = None
    executor._fallback_partition = None
    executor._partitions = None
    executor._failed_nodes = set()
    executor._main_event_loop = None
    executor._status_query_calls = 0
    executor._status_query_failures = 0
    executor._status_query_total_seconds = 0.0
    executor._status_query_min_seconds = None
    executor._status_query_max_seconds = 0.0
    executor._status_query_cycle_rows = []
    executor._preemption_warning = False
    executor._submitted_job_clusters = set()

    # Replicate the array_jobs parsing from Executor.__post_init__
    if array_jobs:
        normalized = array_jobs.replace(";", ",")
        executor.array_jobs = {r.strip() for r in normalized.split(",") if r.strip()}
    else:
        executor.array_jobs = set()
    executor.max_array_size = int(array_limit)

    executor.slurm_logdir = Path("/tmp/test_slurm_logs")
    executor.workflow = SimpleNamespace(
        executor_settings=SimpleNamespace(
            array_limit=array_limit,
            status_attempts=1,
            init_seconds_before_status_checks=40,
            disable_memory_fudge=False,
            keep_successful_logs=False,
            requeue=False,
            no_requeue=False,
            qos=None,
            reservation=None,
            pass_command_as_script=False,
        ),
        workdir_init=Path("/tmp"),
        # get_python_executable reads this to decide between sys.executable and
        # a bare "python". run_array_jobs needs it now that the batch script
        # names the interpreter that decodes the per task payload.
        storage_settings=SimpleNamespace(shared_fs_usage=set()),
    )

    executor._job_submission_executor = MagicMock()
    executor.report_job_success = MagicMock()
    executor.report_job_error = MagicMock()
    executor._report_job_submission_threadsafe = MagicMock()
    executor._report_job_error_threadsafe = MagicMock()
    return executor


class TestArrayJobsSettings:
    """Tests for ExecutorSettings array-job fields and their defaults."""

    def test_array_jobs_default_is_none(self):
        """array_jobs field defaults to None."""
        settings = ExecutorSettings()
        assert settings.array_jobs is None

    def test_array_limit_default_is_1000(self):
        """array_limit field defaults to 1000."""
        settings = ExecutorSettings()
        assert settings.array_limit == 1000

    def test_disable_memory_fudge_defaults_to_false(self):
        """Existing array memory behavior remains enabled by default."""
        settings = ExecutorSettings()
        assert settings.disable_memory_fudge is False

    def test_array_jobs_none_yields_empty_set_on_executor(self):
        """Executor with array_jobs=None initialises self.array_jobs as empty set."""
        executor = _make_executor_stub(array_jobs=None)
        assert executor.array_jobs == set()

    def test_array_jobs_comma_separated_parsed(self):
        """Comma-separated rule names are split into a set."""
        executor = _make_executor_stub(array_jobs="rule1, rule2")
        assert executor.array_jobs == {"rule1", "rule2"}

    def test_array_jobs_semicolons_normalised(self):
        """Semicolons are normalised to commas before splitting."""
        executor = _make_executor_stub(array_jobs="rule1; rule2")
        assert executor.array_jobs == {"rule1", "rule2"}

    def test_array_jobs_all_keyword_preserved(self):
        """The magic keyword 'all' is preserved as a set member."""
        executor = _make_executor_stub(array_jobs="all")
        assert executor.array_jobs == {"all"}

    def test_array_jobs_extra_whitespace_stripped(self):
        """Leading/trailing whitespace is stripped from each rule name."""
        executor = _make_executor_stub(array_jobs="  rule1 ,  rule2  ")
        assert executor.array_jobs == {"rule1", "rule2"}


class TestRunJobsRouting:
    """Tests that run_jobs dispatches to run_job or run_array_jobs correctly."""

    def test_emits_job_info_once_for_each_submitted_job(self):
        """Every submitted job logs its standard JOB_INFO metadata first."""
        executor = _make_executor_stub(array_jobs="myrule")
        jobs = [_make_mock_job(rule_name="myrule", jobid=i) for i in range(3)]

        executor.run_jobs(jobs)

        for job in jobs:
            job.log_info.assert_called_once()

    def test_single_non_array_job_uses_run_job(self):
        """One job with no array setting → run_job is enqueued."""
        executor = _make_executor_stub()
        job = _make_mock_job(rule_name="myrule")
        executor.run_jobs([job])

        calls = executor._job_submission_executor.submit.call_args_list
        assert len(calls) == 1
        assert calls[0][0][0] == executor.run_job

    def test_multiple_non_array_jobs_each_get_run_job(self):
        """Three jobs, no array setting → three individual run_job submissions."""
        executor = _make_executor_stub()
        jobs = [_make_mock_job(rule_name="myrule", jobid=i) for i in range(3)]
        executor.run_jobs(jobs)

        calls = executor._job_submission_executor.submit.call_args_list
        assert len(calls) == 3
        for c in calls:
            assert c[0][0] == executor.run_job

    def test_array_rule_single_ready_job_falls_back_to_run_job(self):
        """Array selected for rule but only 1 ready job → run_job, debug log emitted."""
        executor = _make_executor_stub(array_jobs="myrule")
        job = _make_mock_job(rule_name="myrule")
        executor.run_jobs([job])

        calls = executor._job_submission_executor.submit.call_args_list
        assert len(calls) == 1
        assert calls[0][0][0] == executor.run_job
        # A debug-level message explains the single-job fallback
        executor.logger.debug.assert_called()

    def test_array_rule_multiple_jobs_use_run_array_jobs(self):
        """Array selected + 3 ready jobs for the same rule → one run_array_jobs call."""
        executor = _make_executor_stub(array_jobs="myrule")
        jobs = [_make_mock_job(rule_name="myrule", jobid=i) for i in range(3)]
        executor.run_jobs(jobs)

        calls = executor._job_submission_executor.submit.call_args_list
        assert len(calls) == 1
        assert calls[0][0][0] == executor.run_array_jobs
        # All 3 jobs are forwarded together
        assert calls[0][0][1] == jobs

    def test_group_job_for_array_rule_uses_run_job_with_warning(self):
        """Group job whose rule is in array_jobs → run_job; logger.warning called."""
        executor = _make_executor_stub(array_jobs="myrule")
        job = _make_mock_job(rule_name="myrule", is_group=True)
        executor.run_jobs([job])

        calls = executor._job_submission_executor.submit.call_args_list
        assert len(calls) == 1
        assert calls[0][0][0] == executor.run_job
        executor.logger.warning.assert_called()

    def test_all_keyword_routes_to_run_array_jobs(self):
        """array_jobs='all' + 2 regular jobs for any rule → run_array_jobs."""
        executor = _make_executor_stub(array_jobs="all")
        jobs = [_make_mock_job(rule_name="anyrule", jobid=i) for i in range(2)]
        executor.run_jobs(jobs)

        calls = executor._job_submission_executor.submit.call_args_list
        assert len(calls) == 1
        assert calls[0][0][0] == executor.run_array_jobs

    def test_mixed_group_and_regular_jobs_routed_independently(self):
        """Group job → run_job; regular job pair for array rule → run_array_jobs."""
        executor = _make_executor_stub(array_jobs="myrule")
        group_job = _make_mock_job(rule_name="myrule", jobid=0, is_group=True)
        regular_jobs = [
            _make_mock_job(rule_name="myrule", jobid=i) for i in range(1, 3)
        ]
        executor.run_jobs([group_job] + regular_jobs)

        calls = executor._job_submission_executor.submit.call_args_list
        assert len(calls) == 2
        methods = [c[0][0] for c in calls]
        assert executor.run_job in methods
        assert executor.run_array_jobs in methods

    def test_array_rule_submits_a_partial_batch(self):
        """
        Fewer ready jobs than the DAG has pending, and fewer than one chunk,
        still go out. Held back they would never be submitted at all: the
        scheduler counts every job it hands over as running and never offers
        it again.
        """
        executor = _make_executor_stub(array_jobs="myrule", array_limit=10)
        ready_jobs = [_make_mock_job(rule_name="myrule", jobid=i) for i in range(1, 6)]
        pending_jobs = [
            _make_mock_job(rule_name="myrule", jobid=i) for i in range(1, 101)
        ]
        executor.workflow.dag = SimpleNamespace(needrun_jobs=lambda: pending_jobs)

        executor.run_jobs(ready_jobs)

        calls = executor._job_submission_executor.submit.call_args_list
        assert len(calls) == 1
        assert calls[0][0][0] == executor.run_array_jobs
        assert calls[0][0][1] == ready_jobs

    def test_array_rule_single_ready_job_submits_while_others_pend(self):
        """One ready job of many pending is submitted, not held for an array."""
        executor = _make_executor_stub(array_jobs="myrule", array_limit=10)
        ready_job = _make_mock_job(rule_name="myrule", jobid=1)
        pending_jobs = [
            _make_mock_job(rule_name="myrule", jobid=i) for i in range(1, 101)
        ]
        executor.workflow.dag = SimpleNamespace(needrun_jobs=lambda: pending_jobs)

        executor.run_jobs([ready_job])

        calls = executor._job_submission_executor.submit.call_args_list
        assert len(calls) == 1
        assert calls[0][0][0] == executor.run_job
        ready_job.log_info.assert_called_once()

    def test_array_rule_splits_jobs_that_need_different_sbatch_options(self):
        """A retry with more memory is submitted apart from the first attempts.

        One array submission carries the options of its first task, so mixing
        them would run the retry with the memory it already died on.
        """
        executor = _make_executor_stub(array_jobs="myrule", array_limit=10)
        first_attempt = [
            _make_mock_job(rule_name="myrule", jobid=i, mem_mb=1000) for i in (1, 2)
        ]
        retry = _make_mock_job(rule_name="myrule", jobid=3, mem_mb=4000)

        executor.run_jobs(first_attempt + [retry])

        calls = executor._job_submission_executor.submit.call_args_list
        assert len(calls) == 2
        array_call = [c for c in calls if c[0][0] == executor.run_array_jobs]
        single_call = [c for c in calls if c[0][0] == executor.run_job]
        assert len(array_call) == 1
        assert array_call[0][0][1] == first_attempt
        assert len(single_call) == 1
        assert single_call[0][0][1] == retry

    def test_array_rule_splits_jobs_bound_for_different_partitions(self):
        """One array carries the partition of its first task, so tasks
        asking for different partitions go out separately.
        """
        executor = _make_executor_stub(array_jobs="myrule", array_limit=10)
        cpu_jobs = [
            _make_mock_job(rule_name="myrule", jobid=i, slurm_partition="cpu")
            for i in (1, 2)
        ]
        gpu_jobs = [
            _make_mock_job(rule_name="myrule", jobid=i, slurm_partition="gpu")
            for i in (3, 4)
        ]

        executor.run_jobs(cpu_jobs + gpu_jobs)

        calls = executor._job_submission_executor.submit.call_args_list
        assert len(calls) == 2
        assert all(c[0][0] == executor.run_array_jobs for c in calls)
        submitted = [c[0][1] for c in calls]
        assert cpu_jobs in submitted
        assert gpu_jobs in submitted

    def test_array_rule_splits_jobs_bound_for_different_accounts(self):
        """Tasks billed to different accounts do not share a submission."""
        executor = _make_executor_stub(array_jobs="myrule", array_limit=10)
        first_account = [
            _make_mock_job(rule_name="myrule", jobid=i, slurm_account="acct_a")
            for i in (1, 2)
        ]
        second_account = [
            _make_mock_job(rule_name="myrule", jobid=i, slurm_account="acct_b")
            for i in (3, 4)
        ]

        executor.run_jobs(first_account + second_account)

        calls = executor._job_submission_executor.submit.call_args_list
        assert len(calls) == 2
        submitted = [c[0][1] for c in calls]
        assert first_account in submitted
        assert second_account in submitted

    def test_array_rule_keeps_matching_routing_in_one_submission(self):
        """Equal account and partition requests share one array."""
        executor = _make_executor_stub(array_jobs="myrule", array_limit=10)
        jobs = [
            _make_mock_job(
                rule_name="myrule",
                jobid=i,
                slurm_account="acct_a",
                slurm_partition="cpu",
            )
            for i in (1, 2, 3)
        ]

        executor.run_jobs(jobs)

        calls = executor._job_submission_executor.submit.call_args_list
        assert len(calls) == 1
        assert calls[0][0][0] == executor.run_array_jobs
        assert calls[0][0][1] == jobs

    def test_a_numeric_account_groups_with_its_string_spelling(self):
        """YAML may hand over an account as int; both spell the same account."""
        executor = _make_executor_stub(array_jobs="myrule", array_limit=10)
        jobs = [
            _make_mock_job(rule_name="myrule", jobid=1, slurm_account=123456),
            _make_mock_job(rule_name="myrule", jobid=2, slurm_account="123456"),
        ]

        executor.run_jobs(jobs)

        calls = executor._job_submission_executor.submit.call_args_list
        assert len(calls) == 1
        assert calls[0][0][1] == jobs

    def test_grouping_does_not_resolve_accounts_or_partitions(self):
        """Grouping reads the requested resources, it does not resolve them.

        Resolving validates the account against the cluster and may run
        partition auto-selection, too slow to repeat for every job on every
        dispatch.
        """
        executor = _make_executor_stub(array_jobs="myrule", array_limit=10)
        executor.get_account_arg = MagicMock()
        executor.get_partition_arg = MagicMock()
        jobs = [
            _make_mock_job(rule_name="myrule", jobid=i, slurm_partition="cpu")
            for i in (1, 2)
        ]

        executor.run_jobs(jobs)

        executor.get_account_arg.assert_not_called()
        executor.get_partition_arg.assert_not_called()

    def test_array_rule_submits_a_full_chunk(self):
        """A batch at chunk size goes out as one array submission."""
        executor = _make_executor_stub(array_jobs="myrule", array_limit=10)
        ready_jobs = [_make_mock_job(rule_name="myrule", jobid=i) for i in range(1, 11)]
        pending_jobs = [
            _make_mock_job(rule_name="myrule", jobid=i) for i in range(1, 101)
        ]
        executor.workflow.dag = SimpleNamespace(needrun_jobs=lambda: pending_jobs)

        executor.run_jobs(ready_jobs)

        calls = executor._job_submission_executor.submit.call_args_list
        assert len(calls) == 1
        assert calls[0][0][0] == executor.run_array_jobs
        assert calls[0][0][1] == ready_jobs

    def test_a_throttled_rule_submits_every_job_over_several_rounds(self):
        """A rule the scheduler feeds in batches gets all its jobs submitted.

        Snakemake offers as many jobs as a resource allows, holds the rest,
        and offers them again only when the running ones finish. 34 jobs at 16
        per round, the case that stalled with the previous code.
        """
        executor = _make_executor_stub(array_jobs="myrule", array_limit=200)
        all_jobs = [_make_mock_job(rule_name="myrule", jobid=i) for i in range(1, 35)]
        executor.workflow.dag = SimpleNamespace(needrun_jobs=lambda: all_jobs)

        submitted = []
        for start in range(0, len(all_jobs), 16):
            round_jobs = all_jobs[start : start + 16]
            executor.run_jobs(round_jobs)
            for call in executor._job_submission_executor.submit.call_args_list:
                target, payload = call[0][0], call[0][1]
                submitted.extend(
                    payload if target == executor.run_array_jobs else [payload]
                )
            executor._job_submission_executor.submit.reset_mock()

        assert submitted == all_jobs


class TestRunJobErrorHandling:
    """Tests that single-job submission failures are surfaced to Snakemake."""

    def test_run_job_reports_unhandled_exception(self):
        """Exceptions in run_job are caught and reported via job_error callback."""
        executor = _make_executor_stub()
        job = _make_mock_job(rule_name="myrule")

        def _raise_account_error(_job):
            raise RuntimeError("account lookup failed")

        executor.get_account_arg = _raise_account_error

        # Should not raise, but report a job error to avoid stalling scheduler.
        executor.run_job(job)

        executor._report_job_error_threadsafe.assert_called_once()
        submitted_info, message = executor._report_job_error_threadsafe.call_args[0]
        assert submitted_info.job == job
        assert "account lookup failed" in message


class TestRunArrayJobs:
    """Tests for run_array_jobs: sbatch command structure, chunking, error handling."""

    # --- fixtures & helpers ------------------------------------------------

    @pytest.fixture
    def mock_popen_success(self):
        """Popen mock that returns a successful sbatch response with job ID 987654."""
        with patch("snakemake_executor_plugin_slurm.subprocess.Popen") as mock_popen:
            proc = MagicMock()
            proc.communicate.return_value = ("987654", "")
            proc.returncode = 0
            mock_popen.return_value = proc
            yield mock_popen

    def _build_executor(self, tmp_path, array_limit=1000):
        executor = _make_executor_stub(array_limit=array_limit)
        executor.slurm_logdir = tmp_path / "slurm_logs"
        executor.get_account_arg = MagicMock(
            side_effect=lambda job: iter(["-A testaccount"])
        )
        executor.get_partition_arg = MagicMock(return_value="-p main")
        executor.format_job_exec = MagicMock(
            side_effect=lambda job: f"snakemake_exec_{job.jobid}"
        )
        return executor

    def _make_jobs(self, n=3, rule_name="myrule"):
        return [_make_mock_job(rule_name=rule_name, jobid=i) for i in range(1, n + 1)]

    # --- tests -------------------------------------------------------------

    def test_logfile_per_task_uses_resolved_jobid_and_index(
        self, tmp_path, mock_popen_success
    ):
        """Reported logfile for each task is '<slurm_id>_<index>.log' (1-based)."""
        executor = self._build_executor(tmp_path)
        jobs = self._make_jobs(n=2)
        executor.run_array_jobs(jobs)

        calls = executor._report_job_submission_threadsafe.call_args_list
        assert len(calls) == 2
        for idx, c in enumerate(calls, start=1):
            job_info = c[0][0]
            assert job_info.aux["slurm_logfile"].name == f"987654_{idx}.log"

    def test_logfile_per_task_sits_where_sbatch_writes_it(
        self, tmp_path, mock_popen_success
    ):
        """The reported log path is the one sbatch --output names.

        Error reports quote this path and successful logs are deleted through
        it, so a path nobody writes to breaks both.
        """
        executor = self._build_executor(tmp_path)
        jobs = self._make_jobs(n=2, rule_name="myrule")
        jobs[0].wildcards = {"sample": "a"}
        jobs[1].wildcards = {"sample": "b"}
        executor.run_array_jobs(jobs)

        call = mock_popen_success.call_args[0][0]
        output_dir = Path(re.search(r"--output\s+'?([^'\s]+)", call).group(1)).parent
        for c in executor._report_job_submission_threadsafe.call_args_list:
            assert c[0][0].aux["slurm_logfile"].parent == output_dir

    def test_external_jobid_per_task_is_jobid_underscore_index(
        self, tmp_path, mock_popen_success
    ):
        """external_jobid for each task is '<slurm_id>_<index>' (1-based)."""
        executor = self._build_executor(tmp_path)
        jobs = self._make_jobs(n=3)
        executor.run_array_jobs(jobs)

        calls = executor._report_job_submission_threadsafe.call_args_list
        assert len(calls) == 3
        external_ids = [c[0][0].external_jobid for c in calls]
        assert external_ids == ["987654_1", "987654_2", "987654_3"]

    def test_array_execs_covers_every_task_of_the_chunk(
        self, tmp_path, mock_popen_success
    ):
        """The payload has a key per array task, the first one included.

        The batch script looks its own $SLURM_ARRAY_TASK_ID up in this map, so
        a task missing from it has no command to run.
        """
        executor = self._build_executor(tmp_path)
        jobs = self._make_jobs(n=3)
        executor.run_array_jobs(jobs)
        script = mock_popen_success.return_value.communicate.call_args.kwargs["input"]
        match = re.search(r"([A-Za-z0-9+/=]{16,})\)", script)
        assert match, f"no payload in the batch script:\n{script}"
        array_execs = json.loads(base64.b64decode(match.group(1)).decode("utf-8"))
        assert set(array_execs) == {"1", "2", "3"}

    def test_memory_fudge_can_be_disabled(self, tmp_path, mock_popen_success):
        executor = self._build_executor(tmp_path)
        executor.workflow.executor_settings.disable_memory_fudge = True
        jobs = self._make_jobs(n=2)

        executor.run_array_jobs(jobs)

        popen_call_str = mock_popen_success.call_args_list[0][0][0]
        assert "--mem " not in popen_call_str
        assert "--mem-per-cpu " not in popen_call_str

    def test_memory_fudge_remains_enabled_by_default(
        self, tmp_path, mock_popen_success
    ):
        executor = self._build_executor(tmp_path)
        jobs = self._make_jobs(n=2)

        executor.run_array_jobs(jobs)

        popen_call_str = mock_popen_success.call_args_list[0][0][0]
        assert "--mem 1" in popen_call_str

    def test_array_execs_covers_every_task_of_each_chunk(self, tmp_path):
        """Each chunk's payload covers exactly that chunk's tasks."""
        executor = self._build_executor(tmp_path, array_limit=3)
        jobs = self._make_jobs(n=5)

        with patch("snakemake_executor_plugin_slurm.subprocess.Popen") as mock_popen:
            proc = MagicMock()
            proc.communicate.return_value = ("333333", "")
            proc.returncode = 0
            mock_popen.return_value = proc
            executor.run_array_jobs(jobs)

        maps = []
        for call in proc.communicate.call_args_list:
            script = call.kwargs["input"]
            match = re.search(r"([A-Za-z0-9+/=]{16,})\)", script)
            assert match, f"no payload in the batch script:\n{script}"
            maps.append(json.loads(base64.b64decode(match.group(1)).decode("utf-8")))

        assert set(maps[0]) == {"1", "2", "3"}
        assert set(maps[1]) == {"4", "5"}

    def test_array_limit_produces_chunked_sbatch_calls(self, tmp_path):
        """5 jobs with array_limit=3 → 2 Popen calls: --array=1-3 and --array=4-5."""
        executor = self._build_executor(tmp_path, array_limit=3)
        jobs = self._make_jobs(n=5)

        with patch("snakemake_executor_plugin_slurm.subprocess.Popen") as mock_popen:
            proc = MagicMock()
            proc.communicate.return_value = ("111111", "")
            proc.returncode = 0
            mock_popen.return_value = proc
            executor.run_array_jobs(jobs)

        assert mock_popen.call_count == 2
        first_call_str = mock_popen.call_args_list[0][0][0]
        second_call_str = mock_popen.call_args_list[1][0][0]
        assert "--array=1-3" in first_call_str
        assert "--array=4-5" in second_call_str

    def test_each_task_gets_its_own_command(self, tmp_path):
        """Decoding the payload for task k yields job k's command.

        Before this, every task ran the chunk's first job's command as its
        wrapper and only the nested job step was swapped, so the wrapper
        postprocessed the first job in every task.
        """
        executor = self._build_executor(tmp_path)
        jobs = self._make_jobs(n=4)

        with patch("snakemake_executor_plugin_slurm.subprocess.Popen") as mock_popen:
            proc = MagicMock()
            proc.communicate.return_value = ("444444", "")
            proc.returncode = 0
            mock_popen.return_value = proc
            executor.run_array_jobs(jobs)

        script = proc.communicate.call_args.kwargs["input"]
        match = re.search(r"([A-Za-z0-9+/=]{16,})\)", script)
        assert match, f"no payload in the batch script:\n{script}"
        array_execs = json.loads(base64.b64decode(match.group(1)).decode("utf-8"))

        for index in range(1, 5):
            command = zlib.decompress(bytes.fromhex(array_execs[str(index)])).decode()
            assert command == f"snakemake_exec_{index}"

    def test_submission_is_always_a_stdin_script(self, tmp_path):
        """The payload goes in the batch script, never in the sbatch argv.

        That is what removed the E2BIG retry: an argument list too long for
        --wrap was the only reason to fall back to script mode.
        """
        executor = self._build_executor(tmp_path)
        jobs = self._make_jobs(n=3)

        with patch("snakemake_executor_plugin_slurm.subprocess.Popen") as mock_popen:
            proc = MagicMock()
            proc.communicate.return_value = ("222222", "")
            proc.returncode = 0
            mock_popen.return_value = proc
            executor.run_array_jobs(jobs)

        assert mock_popen.call_count == 1
        call_str = mock_popen.call_args_list[0][0][0]
        assert "/dev/stdin" in call_str
        assert "--wrap=" not in call_str
        assert "--slurm-jobstep-array-execs" not in call_str
        script = proc.communicate.call_args.kwargs["input"]
        assert script.startswith("#!/bin/sh")
        assert "SLURM_ARRAY_TASK_ID" in script

    def test_non_empty_wildcards_in_comment_triggers_warning(
        self, tmp_path, mock_popen_success
    ):
        """
        When wildcards are non-empty, a warning
        about comment limitations is logged.
        """
        executor = self._build_executor(tmp_path)
        jobs = self._make_jobs(n=2)

        with patch(
            "snakemake_executor_plugin_slurm.get_job_wildcards",
            side_effect=["sample_A", "sample_B"],
        ):
            executor.run_array_jobs(jobs)

        executor.logger.warning.assert_called()
        warning_msgs = " ".join(str(c) for c in executor.logger.warning.call_args_list)
        assert "wildcard" in warning_msgs.lower()

    def test_no_wildcards_comment_is_plain_rule_name(
        self, tmp_path, mock_popen_success
    ):
        """Empty wildcards → comment is 'rule_<name>'; no wildcard warning."""
        executor = self._build_executor(tmp_path)
        jobs = self._make_jobs(n=2)

        # Real get_job_wildcards returns "" for jobs with empty wildcards dict
        executor.run_array_jobs(jobs)

        popen_call_str = mock_popen_success.call_args_list[0][0][0]
        assert "rule_myrule" in popen_call_str

        # No wildcard-specific warning should have been issued
        for c in executor.logger.warning.call_args_list:
            assert "wildcard" not in str(c).lower()

    def test_failed_nodes_exclusion_propagated_to_sbatch_call(
        self, tmp_path, mock_popen_success
    ):
        """_failed_nodes set is propagated as --exclude=<node> in the sbatch call."""
        executor = self._build_executor(tmp_path)
        executor._failed_nodes = {"bad_node01"}
        jobs = self._make_jobs(n=2)
        executor.run_array_jobs(jobs)

        popen_call_str = mock_popen_success.call_args_list[0][0][0]
        assert "--exclude=bad_node01" in popen_call_str

    def test_exception_after_a_chunk_does_not_fail_the_submitted_chunk(self, tmp_path):
        """Tasks already registered stay registered when a later chunk raises."""
        executor = self._build_executor(tmp_path, array_limit=2)
        jobs = self._make_jobs(n=4)

        proc = MagicMock()
        proc.communicate.side_effect = [("987654", ""), RuntimeError("sbatch gone")]
        proc.returncode = 0

        with patch("snakemake_executor_plugin_slurm.subprocess.Popen") as popen:
            popen.return_value = proc
            executor.run_array_jobs(jobs)

        submitted = [
            c[0][0].job
            for c in executor._report_job_submission_threadsafe.call_args_list
        ]
        failed = [
            c[0][0].job for c in executor._report_job_error_threadsafe.call_args_list
        ]
        assert submitted == jobs[:2]
        assert failed == jobs[2:]


class TestStatusLookupIds:
    """Edge-case unit tests for _status_lookup_ids."""

    def test_plain_numeric_id_returns_single_entry(self):
        """A plain numeric job ID returns only itself — no parent appended."""
        assert _status_lookup_ids("12345") == ["12345"]

    def test_array_task_appends_parent_id(self):
        """'<jobid>_<taskid>' with all-numeric parts appends the parent ID."""
        assert _status_lookup_ids("12345_3") == ["12345_3", "12345"]

    def test_non_numeric_parent_not_treated_as_array(self):
        """Non-numeric parent prevents parent-ID fallback."""
        assert _status_lookup_ids("abc_123") == ["abc_123"]

    def test_non_numeric_task_not_treated_as_array(self):
        """Non-numeric task index prevents parent-ID fallback."""
        assert _status_lookup_ids("abc_1") == ["abc_1"]

    def test_multiple_underscores_first_split_only(self):
        """Only the first underscore is used; extra parts make task non-numeric."""
        # parent="12345" (digits), task="3_extra" (not digits) → no fallback
        result = _status_lookup_ids("12345_3_extra")
        assert result == ["12345_3_extra"]


class _NoopAsyncContext:
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False


def _make_check_executor():
    """Return an Executor stub wired for check_active_jobs tests."""
    executor = Executor.__new__(Executor)
    executor.logger = MagicMock()
    executor.run_uuid = "run-uuid"
    executor.status_rate_limiter = _NoopAsyncContext()
    executor.get_status_command = lambda: "sacct"
    executor.next_seconds_between_status_checks = 40
    executor._status_query_calls = 0
    executor._status_query_failures = 0
    executor._status_query_total_seconds = 0.0
    executor._status_query_min_seconds = None
    executor._status_query_max_seconds = 0.0
    executor._status_query_cycle_rows = []
    executor._preemption_warning = False
    executor._failed_nodes = set()
    executor.report_job_success = MagicMock()
    executor.report_job_error = MagicMock()
    executor.workflow = SimpleNamespace(
        executor_settings=SimpleNamespace(
            status_attempts=1,
            init_seconds_before_status_checks=40,
            keep_successful_logs=False,
            requeue=False,
        )
    )
    return executor


def _run_check(executor, active_jobs):
    """Drain check_active_jobs into a list synchronously."""

    async def _collect():
        remaining = []
        async for job in executor.check_active_jobs(active_jobs):
            remaining.append(job)
        return remaining

    return asyncio.run(_collect())


class TestCheckActiveArrayJobs:
    """Tests for check_active_jobs status resolution with array tasks."""

    def _patch_all(self, monkeypatch, status_dict):
        """Patch all external dependencies used by check_active_jobs."""

        async def _mock_query(command, logger):
            return (status_dict, 0.01)

        monkeypatch.setattr(
            "snakemake_executor_plugin_slurm.query_job_status", _mock_query
        )
        monkeypatch.setattr(
            "snakemake_executor_plugin_slurm.query_job_status_sacct",
            lambda run_uuid: "mock_sacct_cmd",
        )
        monkeypatch.setattr(
            "snakemake_executor_plugin_slurm.get_min_job_age", lambda: 300
        )
        monkeypatch.setattr(
            "snakemake_executor_plugin_slurm.is_query_tool_available",
            lambda tool: True,
        )

    def test_task_level_status_takes_precedence_over_parent(
        self, monkeypatch, tmp_path
    ):
        """Task-specific 'COMPLETED' wins over parent-array 'FAILED'."""
        executor = _make_check_executor()
        self._patch_all(monkeypatch, {"123_2": "COMPLETED", "123": "FAILED"})

        log = tmp_path / "123_2.log"
        log.write_text("content")
        active_job = SimpleNamespace(external_jobid="123_2", aux={"slurm_logfile": log})

        remaining = _run_check(executor, [active_job])

        assert remaining == []
        executor.report_job_success.assert_called_once()
        executor.report_job_error.assert_not_called()

    def test_all_tasks_of_array_resolved_via_parent_status(self, monkeypatch, tmp_path):
        """Multiple array tasks all resolved via parent 'COMPLETED' in one cycle."""
        executor = _make_check_executor()
        self._patch_all(monkeypatch, {"123": "COMPLETED"})

        active_jobs = [
            SimpleNamespace(
                external_jobid=f"123_{i}",
                aux={"slurm_logfile": tmp_path / f"123_{i}.log"},
            )
            for i in range(1, 4)
        ]

        remaining = _run_check(executor, active_jobs)

        assert remaining == []
        assert executor.report_job_success.call_count == 3
        executor.report_job_error.assert_not_called()

    def test_non_terminal_status_keeps_job_active(self, monkeypatch, tmp_path):
        """A status not in the terminal set (e.g. 'RUNNING') keeps the job active."""
        executor = _make_check_executor()
        self._patch_all(monkeypatch, {"123_1": "RUNNING"})

        active_job = SimpleNamespace(
            external_jobid="123_1",
            aux={"slurm_logfile": tmp_path / "123_1.log"},
        )

        remaining = _run_check(executor, [active_job])

        assert remaining == [active_job]
        executor.report_job_success.assert_not_called()
        executor.report_job_error.assert_not_called()
