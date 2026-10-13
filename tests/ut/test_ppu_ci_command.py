# SPDX-License-Identifier: Apache-2.0
"""Exercise the real CI controller with an in-memory GitHub API boundary."""

from __future__ import annotations

import copy
import importlib
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
HEAD, BASE, CONTROL = "a" * 40, "b" * 40, "c" * 40
REPO = "t-head/vllm-sail"


@pytest.fixture
def controller():
    class Controller:
        def __getattr__(self, name):
            path = ROOT / "scripts/ci/ppu_ci_command.py"
            assert path.is_file(), "The trusted PPU comment controller is missing"
            return getattr(importlib.import_module("scripts.ci.ppu_ci_command"), name)

    return Controller()


class FakeGitHub:
    repo = REPO

    def __init__(self):
        self.pr = {
            "number": 7,
            "state": "open",
            "draft": False,
            "user": {"login": "author"},
            "labels": [{"name": "ready"}],
            "head": {
                "sha": HEAD,
                "ref": "main",
                "repo": {"id": 1, "full_name": "author/fork"},
            },
            "base": {"sha": BASE, "ref": "main", "repo": {"id": 2, "full_name": REPO}},
        }
        self.comment = {
            "id": 17,
            "body": "/ci run",
            "user": {"login": "author", "type": "User"},
            "issue_url": f"https://api.github.com/repos/{REPO}/issues/7",
        }
        self.permissions = {"author": "read", "maintainer": "write"}
        self.reviews = []
        self.decision = "REVIEW_REQUIRED"
        self.behind = 0
        self.checks = {}
        self.runs = {
            100: {
                "id": 100,
                "workflow_id": 10,
                "path": ".github/workflows/run-ci-command.yml",
                "event": "issue_comment",
                "head_sha": CONTROL,
                "head_branch": "main",
                "repository": {"full_name": REPO},
                "run_attempt": 1,
                "status": "in_progress",
                "conclusion": None,
            }
        }
        self.messages = []
        self.dispatches = []
        self.retries = []
        self.cancellations = []
        self.pr_reads = 0
        self.change_on_read = None
        self.fail_dispatch = False

    def get(self, path):
        if path == "":
            return {"default_branch": "main", "full_name": REPO}
        if path == "pulls/7":
            self.pr_reads += 1
            result = copy.deepcopy(self.pr)
            if self.change_on_read == self.pr_reads:
                result["head"]["sha"] = "d" * 40
            return result
        if path == f"issues/comments/{self.comment['id']}":
            return copy.deepcopy(self.comment)
        if path.startswith("collaborators/"):
            return {"permission": self.permissions.get(path.split("/")[1], "read")}
        if path.startswith("compare/"):
            return {"behind_by": self.behind}
        if path.startswith("check-runs/"):
            return copy.deepcopy(self.checks[int(path.split("/")[1])])
        if path.startswith("actions/workflows/"):
            filename = path.split("/")[2]
            return {
                "id": {"run-ci-command.yml": 10, "ppu-pr-ci.yaml": 20}[filename],
                "path": f".github/workflows/{filename}",
            }
        if path.startswith("actions/runs/"):
            return copy.deepcopy(self.runs[int(path.split("/")[2])])
        raise AssertionError(f"Unexpected GET {path}")

    def pages(self, path, key=None):
        if path.startswith("commits/") and "/check-runs" in path:
            sha = path.split("/")[1]
            return [
                copy.deepcopy(c) for c in self.checks.values() if c["head_sha"] == sha
            ]
        if path == "pulls/7/reviews":
            return copy.deepcopy(self.reviews)
        if path == "issues/7/comments":
            return copy.deepcopy(self.messages)
        if path.startswith("actions/workflows/"):
            filename = path.split("/")[2]
            return [
                copy.deepcopy(r)
                for r in self.runs.values()
                if r["path"] == f".github/workflows/{filename}"
            ]
        raise AssertionError(f"Unexpected list {path}")

    def review_decision(self, number):
        assert number == 7
        return self.decision

    def post(self, path, payload=None):
        if path == "check-runs":
            check = {
                "id": len(self.checks) + 1,
                "app": {"id": 15368},
                **copy.deepcopy(payload),
            }
            self.checks[check["id"]] = check
            return copy.deepcopy(check)
        if path.endswith("/dispatches"):
            if self.fail_dispatch:
                raise OSError("Dispatch response was lost")
            self.dispatches.append(copy.deepcopy(payload))
            return None
        if path == "issues/7/comments":
            self.messages.append({"user": {"login": "github-actions[bot]"}, **payload})
            return {}
        if path.endswith("/rerun-failed-jobs"):
            self.retries.append(int(path.split("/")[2]))
            return None
        if path.endswith("/cancel"):
            self.cancellations.append(int(path.split("/")[2]))
            return None
        raise AssertionError(f"Unexpected POST {path}")

    def patch(self, path, payload):
        check_id = int(path.split("/")[1])
        self.checks[check_id].update(copy.deepcopy(payload))
        return copy.deepcopy(self.checks[check_id])


@pytest.fixture
def api():
    return FakeGitHub()


@pytest.fixture
def context():
    return {
        "run_id": 100,
        "control_sha": CONTROL,
        "default_branch": "main",
        "trusted_users": "",
    }


def event(api):
    return {
        "action": "created",
        "issue": {"number": 7, "pull_request": {}},
        "comment": copy.deepcopy(api.comment),
    }


def launch(controller, api, context):
    controller.handle_event(event(api), api, context)
    return next(c for c in api.checks.values() if c["name"] == "PPU CI request")


def executor(api, request):
    run = {
        "id": 200,
        "workflow_id": 20,
        "event": "workflow_dispatch",
        "path": ".github/workflows/ppu-pr-ci.yaml",
        "head_sha": CONTROL,
        "head_branch": "main",
        "repository": {"full_name": REPO},
        "display_title": f"PPU PR 7 head {HEAD} request {request['id']}",
        "run_attempt": 1,
        "status": "in_progress",
        "conclusion": None,
    }
    api.runs[200] = run
    return run


@pytest.mark.parametrize(
    "body,expected",
    [
        ("/ci run", "/ci run"),
        ("  /ci retry\n", "/ci retry"),
        ("/ci cancel", "/ci cancel"),
        ("/ci run --allow-stale", "/ci run --allow-stale"),
        ("/ci run all", None),
        ("/ci run nightly", None),
        ("/ci run; id", None),
        ("```\n/ci run\n```", None),
        ("please /ci run", None),
        ("/ci run\nthanks", None),
    ],
)
def test_exact_commands(controller, body, expected):
    assert controller.parse_command(body) == expected


@pytest.mark.parametrize(
    "permission,draft,labels,actor,trusted,allowed",
    [
        ("read", False, ["ready"], "author", "", True),
        ("read", False, ["ready-run-all-tests"], "author", "", True),
        ("read", False, [], "author", "", False),
        ("read", True, ["ready"], "author", "", False),
        ("read", False, ["ready"], "stranger", "", False),
        ("triage", False, [], "author", "", False),
        ("write", True, [], "maintainer", "", True),
        ("maintain", False, [], "maintainer", "", True),
        ("admin", False, [], "maintainer", "", True),
        ("read", True, [], "Maintainer", "maintainer", True),
    ],
)
def test_authorization_matrix(
    controller, api, permission, draft, labels, actor, trusted, allowed
):
    api.pr.update(draft=draft, labels=[{"name": name} for name in labels])
    api.permissions[actor] = permission
    if allowed:
        controller.authorize(api, api.pr, actor, trusted)
    else:
        with pytest.raises(controller.Rejected):
            controller.authorize(api, api.pr, actor, trusted)


@pytest.mark.parametrize(
    "decision,states,allowed",
    [
        ("APPROVED", ["APPROVED"], True),
        ("REVIEW_REQUIRED", ["APPROVED"], False),
        ("APPROVED", ["APPROVED", "DISMISSED"], False),
        ("APPROVED", ["APPROVED", "CHANGES_REQUESTED"], False),
        ("APPROVED", ["APPROVED", "COMMENTED"], True),
    ],
)
def test_current_trusted_approval(controller, api, decision, states, allowed):
    api.pr["labels"] = []
    api.decision = decision
    api.reviews = [
        {"user": {"login": "maintainer"}, "state": state} for state in states
    ]
    if allowed:
        controller.authorize(api, api.pr, "author", "")
    else:
        with pytest.raises(controller.Rejected):
            controller.authorize(api, api.pr, "author", "")


def test_command_creates_bound_checks_before_dispatch(controller, api, context):
    request = launch(controller, api, context)
    record = json.loads(request["output"]["summary"])
    assert record["head_sha"] == HEAD and record["control_sha"] == CONTROL
    assert record["base_sha"] == BASE and record["comment_id"] == 17
    assert record["controller_run_id"] == 100 and record["pr_number"] == 7
    assert len(api.checks) == 5
    assert all(
        c["head_sha"] == HEAD and c["status"] == "queued" for c in api.checks.values()
    )
    assert api.dispatches == [
        {
            "ref": "main",
            "inputs": {
                "pr_number": "7",
                "head_sha": HEAD,
                "base_sha": BASE,
                "request_check_id": str(request["id"]),
            },
        }
    ]
    controller.handle_event(event(api), api, context)
    assert len(api.dispatches) == 1


@pytest.mark.parametrize(
    "mutation",
    ["closed", "unauthorized", "edited", "issue", "stale", "race", "unknown"],
)
def test_rejected_event_never_dispatches(controller, api, context, mutation):
    ev = event(api)
    if mutation == "closed":
        api.pr["state"] = "closed"
    elif mutation == "unauthorized":
        api.pr["labels"] = []
    elif mutation == "edited":
        ev["action"] = "edited"
    elif mutation == "issue":
        ev["issue"].pop("pull_request")
    elif mutation == "stale":
        api.behind = 3
    elif mutation == "race":
        api.change_on_read = 2
    elif mutation == "unknown":
        ev["comment"]["body"] = "/ci run all"
    controller.handle_event(ev, api, context)
    assert not api.dispatches and not api.checks


@pytest.mark.parametrize("side", ["head", "base"])
def test_stable_pr_ignores_repository_metadata(controller, api, context, side):
    original = copy.deepcopy(api.pr)
    api.pr[side]["repo"]["pushed_at"] = "2026-10-13T08:00:00Z"
    assert controller.stable_pr(api, original, "author", context) == api.pr


@pytest.mark.parametrize("side", ["head", "base"])
@pytest.mark.parametrize("field", ["sha", "ref", "repository", "repository_id"])
def test_stable_pr_rejects_identity_changes(controller, api, context, side, field):
    api.pr[side]["repo"]["id"] = 100
    original = copy.deepcopy(api.pr)
    if field == "repository":
        api.pr[side]["repo"]["full_name"] = "other/repo"
    elif field == "repository_id":
        api.pr[side]["repo"]["id"] = 101
    else:
        api.pr[side][field] = "d" * 40 if field == "sha" else "other-branch"
    with pytest.raises(controller.Rejected, match="head or base changed"):
        controller.stable_pr(api, original, "author", context)


def test_stable_pr_rechecks_authorization(controller, api, context):
    original = copy.deepcopy(api.pr)
    api.pr["labels"] = []
    with pytest.raises(controller.Rejected):
        controller.stable_pr(api, original, "author", context)


def test_allow_stale_runs_but_warns(controller, api, context):
    api.behind = 2
    api.comment["body"] = "/ci run --allow-stale"
    launch(controller, api, context)
    assert api.dispatches
    assert "behind" in api.messages[-1]["body"]


def test_api_failure_is_not_an_authorization(controller, api, context, monkeypatch):
    monkeypatch.setattr(api, "get", lambda _: (_ for _ in ()).throw(OSError("403")))
    with pytest.raises(OSError):
        controller.handle_event(event(api), api, context)
    assert not api.dispatches


def test_lost_dispatch_response_keeps_request_pending(controller, api, context):
    api.fail_dispatch = True
    controller.handle_event(event(api), api, context)
    assert len(api.checks) == 5
    assert all(c["status"] == "queued" for c in api.checks.values())
    assert "confirm" in api.messages[-1]["body"].lower()


def test_executor_validates_record_and_uses_pr_sha(controller, api, context):
    request = launch(controller, api, context)
    run = executor(api, request)
    inputs = api.dispatches[0]["inputs"]
    record = controller.validate_execution(api, inputs, run, context)
    assert record["head_sha"] == HEAD != run["head_sha"]


@pytest.mark.parametrize(
    "mutation",
    [
        "app",
        "controller",
        "repository",
        "head",
        "base",
        "control",
        "branch",
        "label",
        "input",
        "replay",
    ],
)
def test_executor_fails_closed_before_ppu(controller, api, context, mutation):
    request = launch(controller, api, context)
    run = executor(api, request)
    inputs = copy.deepcopy(api.dispatches[0]["inputs"])
    if mutation == "app":
        api.checks[request["id"]]["app"]["id"] = 999
    elif mutation == "controller":
        api.runs[100]["path"] = ".github/workflows/ut.yaml"
    elif mutation == "repository":
        run["repository"]["full_name"] = "evil/repo"
    elif mutation == "head":
        api.pr["head"]["sha"] = "e" * 40
    elif mutation == "base":
        api.pr["base"]["sha"] = "e" * 40
    elif mutation == "control":
        run["head_sha"] = "e" * 40
    elif mutation == "branch":
        run["head_branch"] = "untrusted"
    elif mutation == "label":
        api.pr["labels"] = []
    elif mutation == "input":
        inputs["head_sha"] = "e" * 40
    elif mutation == "replay":
        api.runs[199] = {**copy.deepcopy(run), "id": 199}
    with pytest.raises(controller.Rejected):
        controller.validate_execution(api, inputs, run, context)


def test_ready_notification_is_idempotent_and_never_dispatches(
    controller, api, context
):
    ev = {"action": "labeled", "label": {"name": "ready"}, "pull_request": api.pr}
    controller.notify(ev, api, context)
    controller.notify(ev, api, context)
    assert len(api.messages) == 1 and "/ci run" in api.messages[0]["body"]
    assert not api.dispatches and not api.checks


def test_retry_requires_an_execution_for_current_head(controller, api, context):
    api.comment["body"] = "/ci retry"
    controller.handle_event(event(api), api, context)
    assert not api.retries and not api.dispatches
    assert "/ci run" in api.messages[-1]["body"]


def test_retry_and_cancel_only_touch_bound_execution(controller, api, context):
    request = launch(controller, api, context)
    run = executor(api, request)
    run.update(status="completed", conclusion="failure")
    api.comment.update(id=18, body="/ci retry")
    controller.handle_event(event(api), api, context)
    assert api.retries == [200]
    assert all(c["status"] == "queued" for c in api.checks.values())
    run.update(status="in_progress", run_attempt=2)
    api.comment.update(id=19, body="/ci cancel")
    controller.handle_event(event(api), api, context)
    assert api.cancellations == [200]


@pytest.mark.parametrize("code", [400, 401, 403, 404, 409, 422, 429])
def test_rejected_retry_write_can_be_retried(
    controller, api, context, monkeypatch, code
):
    from urllib.error import HTTPError

    request = launch(controller, api, context)
    run = executor(api, request)
    run.update(status="completed", conclusion="failure")
    original_post = api.post

    def reject(path, payload=None):
        if path.endswith("/rerun-failed-jobs"):
            raise HTTPError("https://api.github.com", code, "rejected", {}, None)
        return original_post(path, payload)

    monkeypatch.setattr(api, "post", reject)
    api.comment.update(id=18, body="/ci retry")
    controller.handle_event(event(api), api, context)
    record = json.loads(api.checks[request["id"]]["output"]["summary"])
    assert record["min_attempt"] == 1
    assert all(c["status"] == "completed" for c in api.checks.values())
    assert all(c["conclusion"] == "failure" for c in api.checks.values())
    assert "not accepted" in api.messages[-1]["body"]
    monkeypatch.setattr(api, "post", original_post)
    api.comment["id"] = 19
    controller.handle_event(event(api), api, context)
    assert api.retries == [200]


@pytest.mark.parametrize("failure_step", ["record", "check"])
def test_retry_preparation_failure_does_not_wedge_request(
    controller, api, context, monkeypatch, failure_step
):
    request = launch(controller, api, context)
    executor(api, request).update(status="completed", conclusion="failure")
    original_patch = api.patch
    failed = []
    target = request["id"] if failure_step == "record" else request["id"] + 1

    def fail_once(path, payload):
        if path == f"check-runs/{target}" and not failed:
            failed.append(path)
            original_patch(path, payload)
            raise OSError("check response lost before rerun was sent")
        return original_patch(path, payload)

    monkeypatch.setattr(api, "patch", fail_once)
    api.comment.update(id=18, body="/ci retry")
    controller.handle_event(event(api), api, context)
    assert not api.retries
    record = json.loads(api.checks[request["id"]]["output"]["summary"])
    assert record["min_attempt"] == 1
    api.comment["id"] = 19
    controller.handle_event(event(api), api, context)
    assert api.retries == [200]


@pytest.mark.parametrize("accepted", [False, True])
@pytest.mark.parametrize("error_kind", ["connection", "http408", "http503"])
def test_uncertain_retry_write_reconciles_without_duplicate(
    controller, api, context, monkeypatch, accepted, error_kind
):
    from urllib.error import HTTPError

    request = launch(controller, api, context)
    run = executor(api, request)
    run.update(status="completed", conclusion="failure")
    original_post = api.post
    writes = []

    def lose_response(path, payload=None):
        if path.endswith("/rerun-failed-jobs"):
            writes.append(path)
            if accepted:
                original_post(path, payload)
                run.update(run_attempt=2, status="in_progress", conclusion=None)
            if error_kind.startswith("http"):
                raise HTTPError(
                    "https://api.github.com", int(error_kind[4:]), "uncertain", {}, None
                )
            raise OSError("rerun response lost")
        return original_post(path, payload)

    monkeypatch.setattr(api, "post", lose_response)
    api.comment.update(id=18, body="/ci retry")
    controller.handle_event(event(api), api, context)
    record = json.loads(api.checks[request["id"]]["output"]["summary"])
    assert record["min_attempt"] == 2
    assert all(c["status"] == "queued" for c in api.checks.values())
    assert ("confirmed" if accepted else "uncertain") in api.messages[-1]["body"]
    api.comment["id"] = 19
    controller.handle_event(event(api), api, context)
    assert len(writes) == 1
    if not accepted:
        assert "inspect" in api.messages[-1]["body"]
    run.update(run_attempt=2, status="completed", conclusion="failure")
    monkeypatch.setattr(api, "post", original_post)
    api.comment["id"] = 20
    controller.handle_event(event(api), api, context)
    assert api.retries == ([200, 200] if accepted else [200])


def test_http_transport_accepts_compare_and_never_retries_writes(monkeypatch):
    import io

    from scripts.ci.ppu_ci_common import GitHub

    urls = []

    def respond(request, timeout):
        urls.append(request.full_url)
        return io.BytesIO(b'{"behind_by":0}')

    monkeypatch.setattr("urllib.request.urlopen", respond)
    client = GitHub(REPO, "test-token")
    assert client.get(f"compare/main...{HEAD}") == {"behind_by": 0}
    assert urls == [f"https://api.github.com/repos/{REPO}/compare/main...{HEAD}"]
    calls = []

    def unavailable(request, timeout):
        calls.append(request)
        raise OSError("response lost")

    monkeypatch.setattr("urllib.request.urlopen", unavailable)
    with pytest.raises(OSError):
        client.post("actions/workflows/ppu-pr-ci.yaml/dispatches", {})
    assert len(calls) == 1


@pytest.mark.parametrize(
    "path", ["/evil", "../evil", "foo/../evil", "https://evil/a", "foo/%2e%2e/evil"]
)
def test_http_transport_rejects_unsafe_paths(path, monkeypatch):
    from scripts.ci.ppu_ci_common import GitHub, Rejected

    def unexpected(*args, **kwargs):
        pytest.fail("Unsafe API path reached the transport")

    monkeypatch.setattr("urllib.request.urlopen", unexpected)
    with pytest.raises(Rejected):
        GitHub(REPO, "test-token").get(path)


def test_active_automatic_run_is_deduplicated(controller, api, context):
    api.runs[50] = {
        "id": 50,
        "event": "pull_request",
        "status": "in_progress",
        "path": ".github/workflows/build-ppu-wheels.yaml",
        "repository": {"full_name": REPO},
        "pull_requests": [{"number": 7, "head": {"sha": HEAD}}],
    }
    controller.handle_event(event(api), api, context)
    assert not api.dispatches and not api.checks
    assert "/runs/50" in api.messages[-1]["body"]


@pytest.mark.parametrize("target", [100, 200])
def test_workflow_id_is_verified(controller, api, context, target):
    request = launch(controller, api, context)
    run = executor(api, request)
    api.runs[target]["workflow_id"] = 999
    with pytest.raises(controller.Rejected):
        controller.validate_execution(api, api.dispatches[0]["inputs"], run, context)


def test_cancel_after_push_only_cancels_this_pr(controller, api, context):
    request = launch(controller, api, context)
    run = executor(api, request)
    api.runs[201] = {
        **copy.deepcopy(run),
        "id": 201,
        "display_title": run["display_title"].replace("PR 7", "PR 8"),
    }
    api.pr["head"]["sha"] = "d" * 40
    api.comment.update(id=18, body="/ci cancel")
    controller.handle_event(event(api), api, context)
    assert api.cancellations == [200]


def test_cancel_queued_dispatch_blocks_late_execution(controller, api, context):
    request = launch(controller, api, context)
    api.comment.update(id=18, body="/ci cancel")
    controller.handle_event(event(api), api, context)
    run = executor(api, request)
    with pytest.raises(controller.Rejected):
        controller.validate_execution(api, api.dispatches[0]["inputs"], run, context)


def test_retry_respects_revoked_authorization(controller, api, context):
    request = launch(controller, api, context)
    executor(api, request).update(status="completed", conclusion="failure")
    api.pr["labels"] = []
    api.comment.update(id=18, body="/ci retry")
    controller.handle_event(event(api), api, context)
    assert not api.retries


@pytest.mark.parametrize("revoke_retry_actor", [False, True])
def test_retry_validator_uses_current_requester(
    controller, api, context, revoke_retry_actor
):
    request = launch(controller, api, context)
    run = executor(api, request)
    api.pr["labels"] = []
    with pytest.raises(controller.Rejected):
        controller.validate_execution(api, api.dispatches[0]["inputs"], run, context)
    run.update(status="completed", conclusion="failure")
    api.comment.update(
        id=18, body="/ci retry", user={"login": "maintainer", "type": "User"}
    )
    controller.handle_event(event(api), api, context)
    assert api.retries == [200]
    run.update(status="in_progress", conclusion=None, run_attempt=2)
    if revoke_retry_actor:
        api.permissions["maintainer"] = "read"
        with pytest.raises(controller.Rejected):
            controller.validate_execution(
                api, api.dispatches[0]["inputs"], run, context
            )
    else:
        record = controller.validate_execution(
            api, api.dispatches[0]["inputs"], run, context
        )
        assert record["validated"] is True
        assert record["actor"] == "author"


def test_duplicate_delivery_after_push_cannot_test_new_head(controller, api, context):
    launch(controller, api, context)
    api.pr["head"]["sha"] = "d" * 40
    controller.handle_event(event(api), api, context)
    assert len(api.dispatches) == 1


def test_duplicate_retry_comment_cannot_retry_again_after_new_attempt(
    controller, api, context
):
    request = launch(controller, api, context)
    run = executor(api, request)
    run.update(status="completed", conclusion="failure")
    api.comment.update(id=18, body="/ci retry")
    controller.handle_event(event(api), api, context)
    run["run_attempt"] = 2
    controller.handle_event(event(api), api, context)
    assert api.retries == [200]
