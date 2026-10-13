# SPDX-License-Identifier: Apache-2.0
"""Report PR checks using only trusted requests and the GitHub jobs API."""

from __future__ import annotations

import json
import os
import sys

from scripts.ci.ppu_ci_command import requests_for_head, save_record
from scripts.ci.ppu_ci_common import (
    CONTEXTS,
    TITLE,
    GitHub,
    Rejected,
    bound_checks,
    execution_identity,
    matching_runs,
    number,
    request_record,
    require,
    run_url,
)


def identify(event, api):
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
    runs = matching_runs(api, check, record)
    require(
        runs
        and runs[0]["id"] == run["id"]
        and record.get("run_id", run["id"]) == run["id"],
        "Replayed execution.",
    )
    return run, check, record


def aggregate(api, run):
    effective = {}
    for attempt in range(1, number(run["run_attempt"]) + 1):
        seen = set()
        for job in api.pages(
            f"actions/runs/{run['id']}/attempts/{attempt}/jobs", "jobs"
        ):
            name = job["name"]
            if name not in {"PPU / " + c for c in CONTEXTS}:
                continue
            require(name not in seen, "Duplicate required job name.")
            require(job["run_attempt"] == attempt, "Job attempt mismatch.")
            seen.add(name)
            effective[name.removeprefix("PPU / ")] = job
    result = {}
    for name in CONTEXTS:
        job = effective.get(name, {})
        conclusion = (
            job.get("conclusion") if job.get("status") == "completed" else "failure"
        )
        result[name] = (
            conclusion
            if conclusion in {"success", "cancelled", "timed_out"}
            else "failure"
        )
    build_attempt = effective.get(CONTEXTS[0], {}).get("run_attempt", 0)
    for name in CONTEXTS[1:3]:
        if effective.get(name, {}).get("run_attempt", 0) < build_attempt:
            result[name] = "failure"
    if any(result[name] != "success" for name in CONTEXTS):
        result[CONTEXTS[-1]] = "failure"
    return result


def report(event, api):
    require(
        event.get("action") in {"requested", "in_progress", "completed"},
        "Unexpected report event.",
    )
    run, check, record = identify(event, api)
    if (
        event["workflow_run"]["run_attempt"] != run["run_attempt"]
        or run["run_attempt"] < record["min_attempt"]
    ):
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
    try:
        result = aggregate(api, run)
    except Rejected:
        result = dict.fromkeys(CONTEXTS, "failure")
    if not record.get("validated"):
        result = dict.fromkeys(CONTEXTS, "failure")
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
                    "summary": f"Attempt {run['run_attempt']}: {result[name]}. {link}",
                },
            },
        )
    record["run_id"] = run["id"]
    save_record(api, check, record, status="completed", conclusion=result[CONTEXTS[-1]])


def main():
    api = GitHub(os.environ["GITHUB_REPOSITORY"], os.environ["GH_TOKEN"])
    with open(os.environ["GITHUB_EVENT_PATH"]) as stream:
        event = json.load(stream)
    if len(sys.argv) > 1 and sys.argv[1] == "identify":
        _, _, record = identify(event, api)
        with open(os.environ["GITHUB_OUTPUT"], "a") as stream:
            stream.write(f"pr_number={record['pr_number']}\n")
    else:
        report(event, api)


if __name__ == "__main__":
    main()
