# SPDX-License-Identifier: Apache-2.0
"""Trusted PPU comment controller; never execute this from a PR revision."""

from __future__ import annotations

import json
import os
import re
import sys
from urllib.parse import quote

from scripts.ci.ppu_ci_common import (
    ACTIONS_APP_ID,
    CONTEXTS,
    EXECUTOR,
    REQUEST_NAME,
    TITLE,
    GitHub,
    Rejected,
    behind_base,
    bound_checks,
    checks,
    context_from_env,
    execution_identity,
    matching_runs,
    number,
    request_record,
    require,
    run_url,
    sha,
)

COMMANDS = {"/ci run", "/ci run --allow-stale", "/ci retry", "/ci cancel"}
READY = {"ready", "ready-run-all-tests"}
HELP = (
    "PPU CI is authorized. Comment `/ci run` to test the current head, "
    "`/ci retry` to retry its failed jobs, or `/ci cancel` to cancel its PPU run. "
    "New commits require a new `/ci run`. CPU checks remain automatic. "
    "`/ci run all` and `/ci run nightly` are not supported."
)


def parse_command(body):
    body = body.strip()
    return body if body in COMMANDS else None


def trusted(api, actor, configured):
    users = {u.lower() for u in re.split(r"[\s,]+", configured) if u}
    if actor.lower() in users:
        return True
    permission = api.get(f"collaborators/{quote(actor, safe='')}/permission")[
        "permission"
    ]
    return permission in {"write", "maintain", "admin"}


def authorize(api, pr, actor, configured):
    require(
        pr["state"] == "open" and pr["base"]["repo"]["full_name"] == api.repo,
        "CI commands require an open PR in this repository.",
    )
    if trusted(api, actor, configured):
        return
    require(
        actor.lower() == pr["user"]["login"].lower() and not pr["draft"],
        "Only the authorized PR author or a maintainer can request PPU CI.",
    )
    if READY.intersection(label["name"] for label in pr["labels"]):
        return
    require(
        api.review_decision(pr["number"]) == "APPROVED",
        "Ask a maintainer for a ready label or a trusted approval first.",
    )
    latest = {}
    for review in api.pages(f"pulls/{pr['number']}/reviews"):
        if review["state"] in {"APPROVED", "CHANGES_REQUESTED", "DISMISSED"}:
            latest[review["user"]["login"]] = review["state"]
    require(
        any(
            state == "APPROVED" and trusted(api, user, configured)
            for user, state in latest.items()
        ),
        "A current trusted approval is required.",
    )


def reply(api, pr_number, message):
    api.post(f"issues/{number(pr_number)}/comments", {"body": message})


def requests_for_head(api, pr):
    result = []
    for check in checks(api, pr["head"]["sha"]):
        if (
            check.get("name") == REQUEST_NAME
            and check.get("app", {}).get("id") == ACTIONS_APP_ID
        ):
            record = request_record(api, check)
            if record["pr_number"] == pr["number"]:
                result.append((check, record))
    return sorted(result, key=lambda item: item[1]["controller_run_id"], reverse=True)


def save_record(api, check, record, **updates):
    return api.patch(
        f"check-runs/{check['id']}",
        {
            "output": {
                "title": REQUEST_NAME,
                "summary": json.dumps(record, sort_keys=True),
            },
            **updates,
        },
    )


def stable_pr(api, pr, actor, context):
    current = api.get(f"pulls/{pr['number']}")
    require(
        current["head"] == pr["head"] and current["base"] == pr["base"],
        "PR head or base changed. Please send the command again.",
    )
    authorize(api, current, actor, context["trusted_users"])
    return current


def run_command(api, pr, comment, context, command):
    existing = requests_for_head(api, pr)
    for check, record in existing:
        if record["comment_id"] == comment["id"]:
            reply(
                api,
                pr["number"],
                f"This request was already recorded: {check['details_url']}",
            )
            return
        runs = matching_runs(api, check, record)
        if check["status"] != "completed" or any(
            r["status"] != "completed" for r in runs
        ):
            link = run_url(api, runs[0]["id"]) if runs else check["details_url"]
            reply(api, pr["number"], f"PPU CI is already queued or active: {link}")
            return
    for run in api.pages(
        "actions/workflows/build-ppu-wheels.yaml/runs?event=pull_request",
        "workflow_runs",
    ):
        if (
            run["event"] == "pull_request"
            and run["status"] != "completed"
            and run["repository"]["full_name"] == api.repo
            and any(
                p["number"] == pr["number"] and p["head"]["sha"] == pr["head"]["sha"]
                for p in run.get("pull_requests", [])
            )
        ):
            reply(
                api,
                pr["number"],
                f"Automatic PPU CI is already active: {run_url(api, run['id'])}",
            )
            return
    behind = behind_base(api, pr)
    require(
        behind == 0 or command == "/ci run --allow-stale",
        "The PR is behind its base branch. Update it or explicitly use `/ci run --allow-stale`.",
    )
    stable_pr(api, pr, comment["user"]["login"], context)
    record = {
        "repository": api.repo,
        "pr_number": pr["number"],
        "head_sha": sha(pr["head"]["sha"]),
        "base_sha": sha(pr["base"]["sha"]),
        "control_sha": sha(context["control_sha"]),
        "default_branch": context["default_branch"],
        "controller_run_id": number(context["run_id"]),
        "comment_id": number(comment["id"]),
        "actor": comment["user"]["login"],
        "command": command,
        "min_attempt": 1,
    }
    check = api.post(
        "check-runs",
        {
            "name": REQUEST_NAME,
            "head_sha": record["head_sha"],
            "status": "queued",
            "external_id": f"ppu-ci-request:{record['controller_run_id']}:{record['comment_id']}",
            "details_url": run_url(api, record["controller_run_id"]),
            "output": {
                "title": REQUEST_NAME,
                "summary": json.dumps(record, sort_keys=True),
            },
        },
    )
    for name in CONTEXTS:
        api.post(
            "check-runs",
            {
                "name": name,
                "head_sha": record["head_sha"],
                "status": "queued",
                "external_id": f"ppu-ci:{check['id']}:{name}",
                "details_url": check["details_url"],
            },
        )
    try:
        api.post(
            f"actions/workflows/{EXECUTOR}/dispatches",
            {
                "ref": context["default_branch"],
                "inputs": {
                    "pr_number": str(pr["number"]),
                    "head_sha": record["head_sha"],
                    "base_sha": record["base_sha"],
                    "request_check_id": str(check["id"]),
                },
            },
        )
    except OSError:
        reply(
            api,
            pr["number"],
            "Dispatch acceptance is uncertain; the checks remain queued. "
            f"A maintainer must confirm the run before retrying: {check['details_url']}",
        )
        return
    warning = (
        f" Warning: this head is {behind} commits behind its base; merge protection is unchanged."
        if behind
        else ""
    )
    reply(
        api,
        pr["number"],
        f"PPU CI queued for `{record['head_sha']}`.{warning} {check['details_url']}",
    )


def cancel_pr(api, pr, comment, context):
    requests = {
        check["id"]: (check, record) for check, record in requests_for_head(api, pr)
    }
    for run in api.pages(
        f"actions/workflows/{EXECUTOR}/runs?event=workflow_dispatch", "workflow_runs"
    ):
        title = TITLE.fullmatch(run.get("display_title", ""))
        if title and int(title[1]) == pr["number"] and run["status"] != "completed":
            check = api.get(f"check-runs/{number(title[3])}")
            record = request_record(api, check)
            execution_identity(api, run, check, record)
            requests[check["id"]] = check, record
    stable_pr(api, pr, comment["user"]["login"], context)
    require(requests, "No on-demand PPU request exists for this PR.")
    cancelled_any = False
    for check, record in requests.values():
        runs = matching_runs(api, check, record)
        if check["status"] == "completed" and all(
            r["status"] == "completed" for r in runs
        ):
            continue
        if comment["id"] in record.get("handled_comments", []):
            continue
        cancelled_any = True
        record["cancelled"] = True
        record.setdefault("handled_comments", []).append(comment["id"])
        save_record(api, check, record, status="completed", conclusion="cancelled")
        for status in bound_checks(api, check, record):
            api.patch(
                f"check-runs/{status['id']}",
                {"status": "completed", "conclusion": "cancelled"},
            )
        for run in runs:
            if run["status"] != "completed":
                api.post(f"actions/runs/{run['id']}/cancel")
    require(cancelled_any, "No active on-demand PPU request exists for this PR.")
    reply(api, pr["number"], "Cancellation requested for this PR's on-demand PPU CI.")


def retry_or_cancel(api, pr, comment, context, command):
    if command == "/ci cancel":
        cancel_pr(api, pr, comment, context)
        return
    requests = requests_for_head(api, pr)
    require(requests, "No PPU execution exists for this head. Use `/ci run` first.")
    check, record = requests[0]
    runs = matching_runs(api, check, record)
    require(
        runs,
        "The dispatch is still queued; confirm its status before sending another command.",
    )
    run = runs[0]
    stable_pr(api, pr, comment["user"]["login"], context)
    if comment["id"] in record.get("handled_comments", []):
        return
    require(
        run["status"] == "completed"
        and run["conclusion"]
        in {"failure", "cancelled", "timed_out", "action_required", "startup_failure"},
        f"No completed failed execution to retry: {run_url(api, run['id'])}",
    )
    require(run["run_attempt"] >= record["min_attempt"], "A retry is already queued.")
    record["min_attempt"] = run["run_attempt"] + 1
    record["run_id"] = run["id"]
    record["cancelled"] = False
    record["retry_actor"] = comment["user"]["login"]
    record.setdefault("handled_comments", []).append(comment["id"])
    save_record(api, check, record, status="queued")
    for status in bound_checks(api, check, record):
        api.patch(f"check-runs/{status['id']}", {"status": "queued"})
    api.post(f"actions/runs/{run['id']}/rerun-failed-jobs")
    reply(api, pr["number"], f"Failed jobs queued for retry: {run_url(api, run['id'])}")


def handle_event(event, api, context):
    if event.get("action") != "created" or "pull_request" not in event.get("issue", {}):
        return
    command = parse_command(event["comment"]["body"])
    pr_number = number(event["issue"]["number"])
    if command is None:
        if event["comment"]["body"].strip().startswith("/ci "):
            reply(api, pr_number, "Unsupported CI command. " + HELP)
        return
    try:
        comment = api.get(f"issues/comments/{number(event['comment']['id'])}")
        require(comment == event["comment"], "The comment changed; post a new command.")
        require(
            comment["issue_url"]
            == f"https://api.github.com/repos/{api.repo}/issues/{pr_number}"
            and comment["user"]["type"] == "User",
            "Invalid command source.",
        )
        pr = api.get(f"pulls/{pr_number}")
        authorize(api, pr, comment["user"]["login"], context["trusted_users"])
        marker = f"<!-- ppu-ci-command:{comment['id']} -->"
        if any(
            c.get("user", {}).get("login") == "github-actions[bot]"
            and marker in c["body"]
            for c in api.pages(f"issues/{pr_number}/comments")
        ):
            reply(
                api,
                pr_number,
                "This comment has already been processed. Post a new command if needed.",
            )
            return
        # Reserve a receipt before dispatch; it prevents replay, never grants access.
        reply(
            api,
            pr_number,
            marker + f"\nProcessing `{command}`: {run_url(api, context['run_id'])}",
        )
        if command.startswith("/ci run"):
            run_command(api, pr, comment, context, command)
        else:
            retry_or_cancel(api, pr, comment, context, command)
    except Rejected as error:
        reply(api, pr_number, str(error))


def validate_execution(api, inputs, run, context):
    check = api.get(f"check-runs/{number(inputs['request_check_id'])}")
    record = request_record(api, check)
    require(
        set(inputs) == {"pr_number", "head_sha", "base_sha", "request_check_id"},
        "Unexpected dispatch inputs.",
    )
    require(
        all(str(record[k]) == inputs[k] for k in ("pr_number", "head_sha", "base_sha")),
        "Dispatch input binding mismatch.",
    )
    execution_identity(api, run, check, record)
    require(
        context["control_sha"] == record["control_sha"]
        and context["default_branch"] == record["default_branch"],
        "Control revision changed.",
    )
    runs = matching_runs(api, check, record)
    require(
        runs
        and runs[0]["id"] == run["id"]
        and record.get("run_id", run["id"]) == run["id"],
        "A request cannot be replayed.",
    )
    require(not record.get("cancelled"), "This request was cancelled.")
    require(
        run["run_attempt"] >= record["min_attempt"], "This attempt has been superseded."
    )
    pr = api.get(f"pulls/{record['pr_number']}")
    require(
        pr["head"]["sha"] == record["head_sha"]
        and pr["base"]["sha"] == record["base_sha"],
        "PR head or base changed; send `/ci run` again.",
    )
    actor = record.get("retry_actor", record["actor"])
    authorize(api, pr, actor, context["trusted_users"])
    require(
        record["command"] == "/ci run --allow-stale" or behind_base(api, pr) == 0,
        "The PR is now behind its base branch.",
    )
    stable_pr(api, pr, actor, context)
    bound_checks(api, check, record)
    record["run_id"] = run["id"]
    record["validated"] = True
    save_record(api, check, record, status="in_progress")
    return record


def notify(event, api, context):
    if (
        event.get("action") != "labeled"
        or event.get("label", {}).get("name") not in READY
    ):
        return
    pr = api.get(f"pulls/{number(event['pull_request']['number'])}")
    if (
        pr["state"] != "open"
        or pr["draft"]
        or not READY.intersection(x["name"] for x in pr["labels"])
    ):
        return
    marker = "<!-- ppu-ci-authorized -->"
    if any(
        c.get("user", {}).get("login") == "github-actions[bot]" and marker in c["body"]
        for c in api.pages(f"issues/{pr['number']}/comments")
    ):
        return
    reply(api, pr["number"], marker + "\n" + HELP)


def main():
    api = GitHub(os.environ["GITHUB_REPOSITORY"], os.environ["GH_TOKEN"])
    with open(os.environ["GITHUB_EVENT_PATH"]) as stream:
        event = json.load(stream)
    context = context_from_env(api)
    mode = sys.argv[1]
    if mode == "command":
        handle_event(event, api, context)
    elif mode == "notify":
        notify(event, api, context)
    elif mode == "validate":
        record = validate_execution(
            api, event["inputs"], api.get(f"actions/runs/{context['run_id']}"), context
        )
        with open(os.environ["GITHUB_OUTPUT"], "a") as stream:
            stream.write(
                f"head_sha={record['head_sha']}\npr_number={record['pr_number']}\n"
            )
    else:
        raise Rejected("Unknown controller mode.")


if __name__ == "__main__":
    main()
