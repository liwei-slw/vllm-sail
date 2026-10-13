# SPDX-License-Identifier: Apache-2.0
"""Report PR checks using only trusted requests and the GitHub jobs API."""

from __future__ import annotations

import json
import os
import sys
from urllib.error import HTTPError

from scripts.ci.ppu_ci_command import requests_for_head, save_record
from scripts.ci.ppu_ci_common import (
    CONTEXTS,
    EXECUTOR,
    STAGES,
    TITLE,
    GitHub,
    Rejected,
    bound_checks,
    execution_evidence,
    execution_identity,
    matching_runs,
    number,
    outcome,
    request_record,
    require,
    run_url,
)


def identify(event, api, known_runs=None):
    run = api.get(f"actions/runs/{number(event['workflow_run']['id'])}")
    title = TITLE.fullmatch(run.get("display_title", ""))
    require(title, "Unexpected executor run title.")
    check = api.get(f"check-runs/{number(title[3])}")
    record = request_record(api, check)
    execution_identity(api, run, check, record)
    require(
        record["default_branch"] == api.get("")["default_branch"],
        "Default branch changed.",
    )
    runs = matching_runs(api, check, record, known_runs)
    require(
        runs
        and runs[0]["id"] == run["id"]
        and record.get("run_id", run["id"]) == run["id"],
        "Replayed execution.",
    )
    return run, check, record


def aggregate(api, run, check, record):
    stages, sources, jobs = execution_evidence(api, check, record, run)
    result = {name: stages[stage] for stage, name in STAGES.items()}
    result[CONTEXTS[-1]] = outcome(jobs.get("PPU / " + CONTEXTS[-1], {}))
    if any(value != "success" for value in result.values()):
        result[CONTEXTS[-1]] = "failure"
    summaries = {}
    for stage, source in sources.items():
        mode = record["validation"]["plan"][stage]["mode"]
        link = f"{run_url(api, source['run_id'])}/job/{source['job_id']}"
        summaries[STAGES[stage]] = (
            f"{mode}: request {source['request_id']}, run {source['run_id']}, attempt 1, job {source['job_id']}. {link}"
        )
    return result, summaries


def report(event, api, expected_pr=None):
    require(
        event.get("action") in {"requested", "in_progress", "completed"},
        "Unexpected report event.",
    )
    run, check, record = identify(event, api)
    require(
        expected_pr is None or number(expected_pr) == record["pr_number"],
        "Report lock identity mismatch.",
    )
    if event["workflow_run"]["run_attempt"] != run["run_attempt"]:
        return
    latest = requests_for_head(
        api, {"number": record["pr_number"], "head": {"sha": record["head_sha"]}}
    )
    if not latest or latest[0][0]["id"] != check["id"]:
        return
    statuses = {c["name"]: c for c in bound_checks(api, check, record)}
    link = run_url(api, run["id"])
    gate = statuses[CONTEXTS[-1]]
    api.patch(
        f"check-runs/{gate['id']}", {"status": "in_progress", "details_url": link}
    )
    if run["status"] != "completed" and not record.get("cancelled"):
        for name in CONTEXTS[:-1]:
            api.patch(
                f"check-runs/{statuses[name]['id']}",
                {"status": "in_progress", "details_url": link},
            )
        return
    summaries = {}
    try:
        result, summaries = aggregate(api, run, check, record)
    except (Rejected, KeyError, TypeError) as error:
        result = dict.fromkeys(CONTEXTS, "failure")
        summaries = dict.fromkeys(CONTEXTS, str(error))
    if record.get("cancelled") or run["conclusion"] == "cancelled":
        result = dict.fromkeys(CONTEXTS, "cancelled")
    elif run["conclusion"] != "success":
        result[CONTEXTS[-1]] = "failure"
    for name in CONTEXTS:
        api.patch(
            f"check-runs/{statuses[name]['id']}",
            {
                "status": "completed",
                "conclusion": result[name],
                "details_url": link,
                "output": {
                    "title": name,
                    "summary": f"Attempt {run['run_attempt']}: {result[name]}. {summaries.get(name, link)}",
                },
            },
        )
    record["run_id"] = run["id"]
    record["reported"] = {**report_identity(run), "conclusions": result}
    save_record(api, check, record, status="completed", conclusion=result[CONTEXTS[-1]])


def report_identity(run):
    return {
        "run_id": run["id"],
        "run_attempt": run["run_attempt"],
        "updated_at": run.get("updated_at"),
        "conclusion": run["conclusion"],
    }


def finalized(api, run, check, record):
    reported = record.get("reported", {})
    if any(reported.get(key) != value for key, value in report_identity(run).items()):
        return False
    results = reported.get("conclusions", {})
    if set(results) != set(CONTEXTS):
        return False
    if record.get("cancelled") and any(
        value != "cancelled" for value in results.values()
    ):
        return False
    if (
        check["status"] != "completed"
        or check.get("conclusion") != results[CONTEXTS[-1]]
    ):
        return False
    return all(
        c["status"] == "completed"
        and c.get("conclusion") == results[c["name"]]
        and c.get("details_url") == run_url(api, run["id"])
        for c in bound_checks(api, check, record)
    )


def discover(event, api):
    # 只读发现；完整分页后按终止时间排序，不向设备派发，也不重发 uncertain 请求。
    selected = event.get("workflow_run", {}).get("id") or event.get("inputs", {}).get(
        "run_id"
    )
    if selected:
        run, _, record = identify({"workflow_run": {"id": number(selected)}}, api)
        return [{"pr_number": record["pr_number"], "run_id": run["id"]}]
    runs = api.pages(f"actions/workflows/{EXECUTOR}/runs", "workflow_runs")
    pending = sorted(
        (r for r in runs if r["status"] == "completed"),
        key=lambda r: (r.get("updated_at", ""), r["id"]),
    )
    targets, heads = [], {}
    for run in pending:
        try:
            run, check, record = identify({"workflow_run": run}, api, runs)
            key = (record["pr_number"], record["head_sha"])
            if key not in heads:
                heads[key] = requests_for_head(
                    api, {"number": key[0], "head": {"sha": key[1]}}
                )
            latest = heads[key]
            if not latest or latest[0][0]["id"] != check["id"]:
                continue
            if not finalized(api, run, check, record):
                targets.append({"pr_number": record["pr_number"], "run_id": run["id"]})
        except HTTPError as error:
            if error.code not in {404, 410}:
                raise
            print(
                f"Ignoring missing source for run {run['id']}: {error}", file=sys.stderr
            )
        except Rejected as error:
            print(f"Ignoring untrusted run {run['id']}: {error}", file=sys.stderr)
        if len(targets) == 100:
            break
    return targets


def reconcile(api, run_id, pr_number):
    # 此函数仅从已取得该 PR 共享锁的 job 调用，重新读取全部执行证据。
    run = api.get(f"actions/runs/{number(run_id)}")
    report(
        {
            "action": "completed" if run["status"] == "completed" else "in_progress",
            "workflow_run": run,
        },
        api,
        expected_pr=pr_number,
    )


def main():
    api = GitHub(os.environ["GITHUB_REPOSITORY"], os.environ["GH_TOKEN"])
    with open(os.environ["GITHUB_EVENT_PATH"]) as stream:
        event = json.load(stream)
    if len(sys.argv) > 1 and sys.argv[1] == "discover":
        targets = discover(event, api)
        with open(os.environ["GITHUB_OUTPUT"], "a") as stream:
            stream.write(
                f"matrix={json.dumps({'include': targets}, separators=(',', ':'))}\n"
            )
            stream.write(f"has_targets={str(bool(targets)).lower()}\n")
    elif os.environ.get("SOURCE_RUN_ID"):
        reconcile(api, os.environ["SOURCE_RUN_ID"], os.environ["REPORT_PR_NUMBER"])
    else:
        report(event, api)


if __name__ == "__main__":
    main()
