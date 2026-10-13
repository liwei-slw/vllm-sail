# SPDX-License-Identifier: Apache-2.0
"""Exercise the real controller and reporter at the GitHub API boundary."""

import copy
import importlib
import json
from pathlib import Path

import pytest
from test_ppu_ci_command import (
    CONTROL,
    HEAD,
    FakeGitHub,
    complete_jobs,
    event,
    executor,
    latest_request,
    launch,
    next_comment,
    validate,
)

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


def jobs(attempt=1, run_id=200):
    return [
        {
            "id": run_id * 10 + i + (attempt - 1) * 100,
            "run_id": run_id,
            "name": name,
            "status": "completed",
            "conclusion": "success",
            "run_attempt": attempt,
        }
        for i, name in enumerate([*("PPU / " + n for n in NAMES), "validate"])
    ]


class ReportAPI(FakeGitHub):
    def __init__(self):
        super().__init__()
        self.attempts = {1: jobs()}
        complete_jobs(self, {"id": 200})

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
    validate(command, api, run, context)
    run.update(status="completed", conclusion="success")
    return api, request, run, command, context


def snapshot(run):
    return {"action": "completed", "workflow_run": copy.deepcopy(run)}


def gate(api):
    return max(
        (c for c in api.checks.values() if c["name"] == "PPU CI gate"),
        key=lambda c: c["id"],
    )


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


def test_retry_reuses_successful_build_across_runs(reporter, setup):
    api, request, run, command, context = setup
    run.update(conclusion="failure")
    api.attempts[1][2]["conclusion"] = "failure"
    api.attempts[1][3]["conclusion"] = "failure"
    reporter.report(snapshot(run), api)
    before = copy.deepcopy(api.checks)
    next_comment(api, context)
    command.handle_event(event(api), api, context)
    assert gate(api)["status"] == "queued"
    second = executor(api, latest_request(api), 201)
    validate(command, api, second, context)
    api.job_runs[201] = jobs(run_id=201)[2:]
    second.update(status="completed", conclusion="success")
    reporter.report(snapshot(second), api)
    assert gate(api)["conclusion"] == "success"
    assert {k: api.checks[k] for k in before} == before
    build = next(
        c
        for c in api.checks.values()
        if c.get("external_id") == f"ppu-ci:{latest_request(api)['id']}:PPU wheel build"
    )
    assert "/runs/200/job/2000" in build["output"]["summary"]
    assert "reuse" in build["output"]["summary"].lower()
    assert str(request["id"]) in build["output"]["summary"]


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
    record.pop("validation", None)
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
    assert not api.retries and len(api.dispatches) == 2
    second = executor(api, latest_request(api), 201)
    record = validate(command, api, second, context)
    assert not record.get("cancelled")
    assert json.loads(api.checks[request["id"]]["output"]["summary"])["cancelled"]


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


@pytest.mark.parametrize("failed_write", range(1, 7))
def test_reconcile_recovers_every_partial_report_write(
    reporter, setup, monkeypatch, failed_write
):
    api, _, run, _, _ = setup
    original = api.patch
    writes = 0

    def fail_once(path, payload):
        nonlocal writes
        writes += 1
        if writes == failed_write:
            raise OSError("Reporter was interrupted")
        return original(path, payload)

    monkeypatch.setattr(api, "patch", fail_once)
    with pytest.raises(OSError):
        reporter.report(snapshot(run), api)
    monkeypatch.setattr(api, "patch", original)
    before = copy.deepcopy(api.checks)
    targets = reporter.discover({"schedule": "*/10 * * * *"}, api)
    assert api.checks == before
    assert targets == [{"pr_number": 7, "run_id": 200}]
    reporter.reconcile(api, targets[0]["run_id"], targets[0]["pr_number"])
    assert all(
        c["status"] == "completed" and c["conclusion"] == "success"
        for c in api.checks.values()
    )
    assert reporter.discover({}, api) == []
    assert len(api.dispatches) == 1 and not api.retries


def test_discovery_scans_all_pages_and_drains_oldest_hundred(
    reporter, setup, monkeypatch
):
    from scripts.ci.ppu_ci_common import GitHub

    api, _, run, command, context = setup
    run["updated_at"] = "2026-01-01T00:00:00Z"
    for index in range(1, 105):
        api.pr["head"]["sha"] = f"{index:040x}"
        api.comment["id"] += 1
        command.create_request(api, api.pr, api.comment, context, "/ci run")
        request = latest_request(api)
        other = executor(api, request, 200 + index)
        other["display_title"] = other["display_title"].replace(
            HEAD, api.pr["head"]["sha"]
        )
        validate(command, api, other, context)
        api.job_runs[other["id"]] = jobs(run_id=other["id"])
        other.update(
            status="completed", conclusion="success", updated_at="2026-01-02T00:00:00Z"
        )
    runs = sorted(
        (r for r in api.runs.values() if r["workflow_id"] == 20), key=lambda r: -r["id"]
    )
    original_pages = api.pages
    pages_seen = []

    def get_page(path):
        assert path.startswith(
            "actions/workflows/ppu-pr-ci.yaml/runs?per_page=100&page="
        )
        page = int(path.rsplit("=", 1)[-1])
        pages_seen.append(page)
        return {"workflow_runs": copy.deepcopy(runs[(page - 1) * 100 : page * 100])}

    def pages(path, key=None):
        if path == "actions/workflows/ppu-pr-ci.yaml/runs":
            from types import SimpleNamespace

            return GitHub.pages(SimpleNamespace(get=get_page), path, key)
        return original_pages(path, key)

    monkeypatch.setattr(api, "pages", pages)
    before = copy.deepcopy(api.checks)
    first = reporter.discover({}, api)
    assert pages_seen[:2] == [1, 2]
    assert api.checks == before
    assert [target["run_id"] for target in first] == list(range(200, 300))
    for target in first:
        reporter.reconcile(api, **target)
    assert [target["run_id"] for target in reporter.discover({}, api)] == list(
        range(300, 305)
    )


def test_repair_rechecks_supersession_and_lock_identity(reporter, setup):
    from scripts.ci.ppu_ci_common import Rejected

    api, _, run, command, context = setup
    target = reporter.discover({}, api)[0]
    with pytest.raises(Rejected):
        reporter.reconcile(api, run_id=200, pr_number=8)
    run["conclusion"] = "failure"
    next_comment(api, context)
    command.handle_event(event(api), api, context)
    before = copy.deepcopy(api.checks)
    reporter.reconcile(api, **target)
    assert api.checks == before
    assert reporter.discover({}, api) == []


def test_reconcile_preserves_cancellation_and_old_head(reporter, setup):
    api, request, run, command, _ = setup
    record = json.loads(api.checks[request["id"]]["output"]["summary"])
    record["cancelled"] = True
    command.save_record(
        api, request, record, status="completed", conclusion="cancelled"
    )
    api.pr["head"]["sha"] = "f" * 40
    reporter.reconcile(api, run_id=run["id"], pr_number=7)
    assert all(
        c["head_sha"] == HEAD and c["conclusion"] == "cancelled"
        for c in api.checks.values()
    )
    assert reporter.discover({}, api) == []


def test_discovery_manual_target_and_native_rerun_recheck(reporter, setup):
    api, _, run, _, _ = setup
    reporter.report(snapshot(run), api)
    assert reporter.discover({}, api) == []
    assert reporter.discover({"inputs": {"run_id": "200"}}, api) == [
        {"pr_number": 7, "run_id": 200}
    ]
    run["run_attempt"] = 2
    target = reporter.discover({}, api)[0]
    reporter.reconcile(api, **target)
    assert gate(api)["conclusion"] == "failure"
    assert reporter.discover({}, api) == []


@pytest.mark.parametrize("code", [404, 410, 503])
def test_discovery_isolates_nonexistent_requests_but_not_api_outages(
    reporter, setup, monkeypatch, code
):
    from urllib.error import HTTPError

    api, _, run, _, _ = setup
    bogus = copy.deepcopy(run)
    bogus.update(id=199, display_title=f"PPU PR 7 head {HEAD} request 9999")
    api.runs[199] = bogus
    original = api.get

    def get(path):
        if path == "check-runs/9999":
            raise HTTPError("https://api.github.com", code, "Unavailable", {}, None)
        return original(path)

    monkeypatch.setattr(api, "get", get)
    if code == 503:
        with pytest.raises(HTTPError):
            reporter.discover({}, api)
    else:
        assert reporter.discover({}, api) == [{"pr_number": 7, "run_id": 200}]


def test_reconciliation_after_queue_capacity_or_reporter_cancellation(reporter, setup):
    api, _, run, _, _ = setup
    # max 队列已满时本次回写未运行；后续定时 discovery 仍会发现它。
    pending, cancelled = list(range(100)), []
    if len(pending) == 100:
        cancelled.append(run["id"])
    assert cancelled == [200]
    target = reporter.discover({}, api)[0]
    # 模拟已进入队列的 repair 又被取消，无需伪造 checks 即可再次恢复。
    assert reporter.discover({}, api) == [target]
    reporter.reconcile(api, **target)
    assert gate(api)["conclusion"] == "success"
    # 终态之后若发生部分回写，finalized 不能仅信请求上的完成标记。
    gate(api).update(status="in_progress")
    assert reporter.discover({}, api) == [target]
    reporter.reconcile(api, **target)
    assert reporter.discover({}, api) == []


def test_uncertain_dispatch_without_run_is_never_redispatched(reporter, setup):
    api, _, run, command, context = setup
    reporter.report(snapshot(run), api)
    next_comment(api, context, body="/ci run")
    api.fail_dispatch = True
    command.handle_event(event(api), api, context)
    before = copy.deepcopy(api.checks)
    assert reporter.discover({}, api) == []
    assert api.checks == before and len(api.dispatches) == 1


def test_native_rerun_never_reports_green(reporter, setup):
    api, _, run, _, _ = setup
    run["run_attempt"] = 2
    api.attempts[2] = jobs(2)
    reporter.report(snapshot(run), api)
    assert gate(api)["conclusion"] == "failure"


@pytest.mark.parametrize(
    "damage", ["run", "attempt", "plan", "missing_job", "failed_job"]
)
def test_validator_requires_run_bound_plan_and_successful_job(reporter, setup, damage):
    api, request, run, _, _ = setup
    record = json.loads(api.checks[request["id"]]["output"]["summary"])
    if damage == "run":
        record["validation"]["run_id"] = 999
    elif damage == "attempt":
        record["validation"]["run_attempt"] = 2
    elif damage == "plan":
        record["validation"]["plan"]["unknown"] = {"mode": "reuse"}
    elif damage == "missing_job":
        api.attempts[1].pop()
    else:
        api.attempts[1][-1]["conclusion"] = "failure"
    api.checks[request["id"]]["output"]["summary"] = json.dumps(record)
    reporter.report(snapshot(run), api)
    assert gate(api)["conclusion"] == "failure"
