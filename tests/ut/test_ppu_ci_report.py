# SPDX-License-Identifier: Apache-2.0
"""Exercise the real controller and reporter at the GitHub API boundary."""

import copy
import importlib
import json
from pathlib import Path

import pytest
from test_ppu_ci_command import CONTROL, HEAD, FakeGitHub, event, executor, launch

NAMES = (
    "PPU wheel build",
    "PPU pin UT / main-verified",
    "PPU E2E smoke / E2E qwen3.8-27b",
    "PPU CI gate",
)


@pytest.fixture
def reporter():
    class Reporter:
        def __getattr__(self, name):
            assert (
                Path(__file__).resolve().parents[2] / "scripts/ci/ppu_ci_report.py"
            ).exists(), "PPU reporter is missing"
            return getattr(importlib.import_module("scripts.ci.ppu_ci_report"), name)

    return Reporter()


def jobs(attempt=1):
    return [
        {
            "id": i + attempt * 100,
            "name": "PPU / " + name,
            "status": "completed",
            "conclusion": "success",
            "run_attempt": attempt,
        }
        for i, name in enumerate(NAMES)
    ]


class ReportAPI(FakeGitHub):
    def __init__(self):
        super().__init__()
        self.attempts = {1: jobs()}

    def pages(self, path, key=None):
        if path.startswith("actions/runs/200/attempts/"):
            return copy.deepcopy(self.attempts[int(path.split("/")[4])])
        return super().pages(path, key)


@pytest.fixture
def setup():
    from scripts.ci import ppu_ci_command as command

    api = ReportAPI()
    context = {
        "run_id": 100,
        "control_sha": CONTROL,
        "default_branch": "main",
        "trusted_users": "",
    }
    request = launch(command, api, context)
    run = executor(api, request)
    command.validate_execution(api, api.dispatches[0]["inputs"], run, context)
    run.update(status="completed", conclusion="success")
    return api, request, run, command, context


def snapshot(run):
    return {"action": "completed", "workflow_run": copy.deepcopy(run)}


def gate(api):
    return next(c for c in api.checks.values() if c["name"] == "PPU CI gate")


def test_report_uses_bound_head_not_control_sha(reporter, setup):
    api, _, run, _, _ = setup
    api.pr["head"]["sha"] = "d" * 40
    reporter.report(snapshot(run), api)
    assert gate(api)["conclusion"] == "success"
    assert all(c["head_sha"] == HEAD for c in api.checks.values())
    assert gate(api)["details_url"].endswith("/runs/200")


@pytest.mark.parametrize(
    "state",
    [
        "failure",
        "cancelled",
        "timed_out",
        "skipped",
        "neutral",
        "action_required",
        "missing",
    ],
)
def test_non_success_stage_never_passes_gate(reporter, setup, state):
    api, _, run, _, _ = setup
    if state == "missing":
        api.attempts[1].pop(2)
    else:
        api.attempts[1][2]["conclusion"] = state
    reporter.report(snapshot(run), api)
    assert gate(api)["status"] == "completed"
    assert gate(api)["conclusion"] != "success"


def test_retry_reuses_successful_build_across_attempts(reporter, setup):
    api, _, run, command, context = setup
    run.update(conclusion="failure")
    api.attempts[1][2]["conclusion"] = "failure"
    api.attempts[1][3]["conclusion"] = "failure"
    reporter.report(snapshot(run), api)
    api.comment.update(id=18, body="/ci retry")
    command.handle_event(event(api), api, context)
    assert gate(api)["status"] == "queued"
    run.update(run_attempt=2, conclusion="success")
    api.attempts[2] = jobs(2)[2:]
    reporter.report(snapshot(run), api)
    assert gate(api)["conclusion"] == "success"


def test_old_callback_cannot_complete_queued_retry(reporter, setup):
    api, _, run, command, context = setup
    run.update(conclusion="failure")
    old = snapshot(run)
    api.comment.update(id=18, body="/ci retry")
    command.handle_event(event(api), api, context)
    before = copy.deepcopy(api.checks)
    reporter.report(old, api)
    assert api.checks == before


def test_rebuilt_wheels_invalidate_older_test_successes(reporter, setup):
    api, _, run, _, _ = setup
    run.update(run_attempt=2)
    api.attempts[2] = [jobs(2)[0], jobs(2)[3]]
    reporter.report(snapshot(run), api)
    assert gate(api)["conclusion"] != "success"


@pytest.mark.parametrize(
    "mutation", ["workflow", "app", "title", "branch", "sha", "repository", "duplicate"]
)
def test_untrusted_callback_cannot_write_checks(reporter, setup, mutation):
    from scripts.ci.ppu_ci_common import Rejected

    api, request, run, _, _ = setup
    if mutation == "workflow":
        run["workflow_id"] = 999
    elif mutation == "app":
        api.checks[request["id"]]["app"]["id"] = 999
    elif mutation == "title":
        run["display_title"] += " extra"
    elif mutation == "branch":
        run["head_branch"] = "evil"
    elif mutation == "sha":
        run["head_sha"] = HEAD
    elif mutation == "repository":
        run["repository"]["full_name"] = "evil/fork"
    else:
        run["id"] = 201
        api.runs[201] = run
    before = copy.deepcopy(api.checks)
    with pytest.raises(Rejected):
        reporter.report(snapshot(run), api)
    assert api.checks == before


def test_cancelled_execution_never_succeeds_even_with_green_jobs(reporter, setup):
    api, _, run, _, _ = setup
    run.update(conclusion="cancelled")
    reporter.report(snapshot(run), api)
    assert gate(api)["conclusion"] == "cancelled"


def test_missing_validator_binding_never_succeeds(reporter, setup):
    api, request, run, _, _ = setup
    record = json.loads(api.checks[request["id"]]["output"]["summary"])
    record.pop("validated", None)
    api.checks[request["id"]]["output"]["summary"] = json.dumps(record)
    reporter.report(snapshot(run), api)
    assert gate(api)["conclusion"] != "success"


def test_duplicate_job_names_fail_closed(reporter, setup):
    api, _, run, _, _ = setup
    api.attempts[1].append(copy.deepcopy(api.attempts[1][0]))
    reporter.report(snapshot(run), api)
    assert gate(api)["conclusion"] != "success"


def test_started_callback_updates_link_but_not_success(reporter, setup):
    api, _, run, _, _ = setup
    run.update(status="in_progress", conclusion=None)
    reporter.report({"action": "in_progress", "workflow_run": copy.deepcopy(run)}, api)
    assert gate(api)["status"] == "in_progress"
    assert gate(api).get("conclusion") != "success"


def test_cancelled_request_can_retry_with_fresh_authorization(reporter, setup):
    api, request, run, command, context = setup
    run.update(status="in_progress", conclusion=None)
    api.comment.update(id=18, body="/ci cancel")
    command.handle_event(event(api), api, context)
    run.update(status="completed", conclusion="cancelled")
    api.comment.update(id=19, body="/ci retry")
    command.handle_event(event(api), api, context)
    assert api.retries == [200]
    run.update(status="in_progress", conclusion=None, run_attempt=2)
    record = command.validate_execution(api, api.dispatches[0]["inputs"], run, context)
    assert not record.get("cancelled")


def test_old_same_sha_callback_does_not_overwrite_new_request(reporter, setup):
    api, _, run, command, context = setup
    old = snapshot(run)
    reporter.report(old, api)
    context["run_id"] = 101
    api.runs[101] = {**copy.deepcopy(api.runs[100]), "id": 101}
    api.comment.update(id=18, body="/ci run")
    command.handle_event(event(api), api, context)
    assert len(api.checks) == 10
    before = copy.deepcopy(api.checks)
    reporter.report(old, api)
    assert api.checks == before


def test_cancel_does_not_invalidate_completed_results(reporter, setup):
    api, _, run, command, context = setup
    reporter.report(snapshot(run), api)
    before = copy.deepcopy(api.checks)
    api.comment.update(id=18, body="/ci cancel")
    command.handle_event(event(api), api, context)
    assert not api.cancellations
    assert api.checks == before
    assert "No active" in api.messages[-1]["body"]


def test_jobs_api_failure_invalidates_old_gate(reporter, setup, monkeypatch):
    api, _, run, _, _ = setup
    reporter.report(snapshot(run), api)
    original = api.pages

    def fail_jobs(path, key=None):
        if "/attempts/" in path:
            raise OSError("GitHub is unavailable")
        return original(path, key)

    monkeypatch.setattr(api, "pages", fail_jobs)
    with pytest.raises(OSError):
        reporter.report(snapshot(run), api)
    assert gate(api)["status"] != "completed"
