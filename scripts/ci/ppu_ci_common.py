# SPDX-License-Identifier: Apache-2.0
"""GitHub API and immutable identities shared by the trusted PPU control plane."""

from __future__ import annotations

import json
import os
import re
import urllib.request
from urllib.parse import quote, unquote, urlsplit

REPOSITORY = "t-head/vllm-sail"
ACTIONS_APP_ID = 15368
REQUEST_NAME = "PPU CI request"
EXECUTOR = "ppu-pr-ci.yaml"
CONTROLLER = "run-ci-command.yml"
CONTEXTS = (
    "PPU wheel build",
    "PPU pin UT / main-verified",
    "PPU E2E smoke / E2E qwen3.8-27b",
    "PPU CI gate",
)
TITLE = re.compile(r"PPU PR ([1-9][0-9]*) head ([0-9a-f]{40}) request ([1-9][0-9]*)")


class Rejected(ValueError):
    """A request is unsafe, unauthorized, stale, or not actionable."""


def require(condition, message):
    if not condition:
        raise Rejected(message)


def sha(value):
    require(
        isinstance(value, str) and re.fullmatch(r"[0-9a-f]{40}", value),
        "Invalid commit SHA.",
    )
    return value


def number(value):
    require(re.fullmatch(r"[1-9][0-9]*", str(value)), "Invalid positive identifier.")
    return int(value)


def run_url(api, run_id):
    return f"https://github.com/{api.repo}/actions/runs/{number(run_id)}"


class GitHub:
    """Small, paginated API client. Never retry potentially accepted writes."""

    def __init__(self, repo, token):
        require(
            repo == REPOSITORY,
            "CI commands are enabled only in the upstream repository.",
        )
        self.repo = repo
        self.token = token

    def request(self, method, path, payload=None, *, graphql=False):
        parsed = urlsplit(path)
        require(
            not parsed.scheme
            and not parsed.netloc
            and not parsed.fragment
            and not path.startswith("/")
            and not {".", ".."}.intersection(unquote(parsed.path).split("/")),
            "Invalid API path.",
        )
        url = (
            "https://api.github.com/graphql"
            if graphql
            else f"https://api.github.com/repos/{self.repo}/{path}"
        )
        request = urllib.request.Request(
            url,
            data=None if payload is None else json.dumps(payload).encode(),
            method=method,
            headers={
                "Authorization": f"Bearer {self.token}",
                "Accept": "application/vnd.github+json",
                "Content-Type": "application/json",
                "X-GitHub-Api-Version": "2022-11-28",
                "User-Agent": "vllm-sail-ppu-ci",
            },
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            data = response.read()
        return json.loads(data) if data else None

    def get(self, path):
        return self.request("GET", path)

    def post(self, path, payload=None):
        return self.request("POST", path, payload)

    def patch(self, path, payload):
        return self.request("PATCH", path, payload)

    def pages(self, path, key=None):
        result = []
        separator = "&" if "?" in path else "?"
        page = 1
        while True:
            data = self.get(f"{path}{separator}per_page=100&page={page}")
            items = data[key] if key else data
            require(isinstance(items, list), "Invalid paginated API response.")
            result.extend(items)
            if len(items) < 100:
                return result
            page += 1

    def review_decision(self, pr_number):
        owner, name = self.repo.split("/")
        data = self.request(
            "POST",
            "",
            {
                "query": "query($owner:String!,$name:String!,$number:Int!){repository(owner:$owner,name:$name){pullRequest(number:$number){reviewDecision}}}",
                "variables": {
                    "owner": owner,
                    "name": name,
                    "number": number(pr_number),
                },
            },
            graphql=True,
        )
        require(
            not data.get("errors"), "Could not determine the current review decision."
        )
        return data["data"]["repository"]["pullRequest"]["reviewDecision"]


def context_from_env(api):
    repository = api.get("")
    return {
        "run_id": number(os.environ["GITHUB_RUN_ID"]),
        "control_sha": sha(os.environ["GITHUB_WORKFLOW_SHA"]),
        "default_branch": repository["default_branch"],
        "trusted_users": os.environ.get("CI_TRUSTED_USERS", ""),
    }


def checks(api, head):
    return api.pages(f"commits/{sha(head)}/check-runs?filter=all", "check_runs")


def request_record(api, check):
    require(check.get("app", {}).get("id") == ACTIONS_APP_ID, "Unexpected request app.")
    require(check.get("name") == REQUEST_NAME, "Not a PPU CI request.")
    try:
        record = json.loads(check["output"]["summary"])
        require(
            record["repository"] == api.repo == REPOSITORY,
            "Request repository mismatch.",
        )
        for key in ("head_sha", "base_sha", "control_sha"):
            sha(record[key])
        for key in ("controller_run_id", "comment_id", "pr_number"):
            number(record[key])
        require(check["head_sha"] == record["head_sha"], "Request head mismatch.")
        expected = (
            f"ppu-ci-request:{record['controller_run_id']}:{record['comment_id']}"
        )
        require(check["external_id"] == expected, "Request identity mismatch.")
        require(
            check["details_url"] == run_url(api, record["controller_run_id"]),
            "Request provenance mismatch.",
        )
    except (KeyError, TypeError, json.JSONDecodeError) as error:
        raise Rejected("Malformed CI request.") from error
    source = api.get(f"actions/runs/{record['controller_run_id']}")
    workflow_identity(api, source, CONTROLLER)
    require(
        source["event"] == "issue_comment"
        and source["path"] == f".github/workflows/{CONTROLLER}"
        and source["head_sha"] == record["control_sha"]
        and source["head_branch"] == record["default_branch"]
        and source["repository"]["full_name"] == api.repo,
        "Untrusted request controller.",
    )
    return record


def workflow_identity(api, run, filename):
    workflow = api.get(f"actions/workflows/{filename}")
    require(
        workflow["path"] == run["path"] == f".github/workflows/{filename}"
        and run.get("workflow_id") == workflow["id"],
        "Workflow identity mismatch.",
    )


def execution_identity(api, run, check, record):
    workflow_identity(api, run, EXECUTOR)
    expected = (
        f"PPU PR {record['pr_number']} head {record['head_sha']} request {check['id']}"
    )
    require(
        run["event"] == "workflow_dispatch"
        and run["path"] == f".github/workflows/{EXECUTOR}"
        and run["repository"]["full_name"] == api.repo
        and run["head_branch"] == record["default_branch"]
        and run["head_sha"] == record["control_sha"]
        and run["display_title"] == expected,
        "Execution does not match its trusted request.",
    )


def bound_checks(api, request, record):
    prefix = f"ppu-ci:{request['id']}:"
    found = [
        c
        for c in checks(api, record["head_sha"])
        if c.get("app", {}).get("id") == ACTIONS_APP_ID
        and c.get("name") in CONTEXTS
        and c.get("external_id") == prefix + c["name"]
    ]
    require(
        len(found) == len(CONTEXTS) and {c["name"] for c in found} == set(CONTEXTS),
        "Required check identities are missing or duplicated.",
    )
    return found


def matching_runs(api, check, record):
    expected = (
        f"PPU PR {record['pr_number']} head {record['head_sha']} request {check['id']}"
    )
    found = []
    for run in api.pages(
        f"actions/workflows/{EXECUTOR}/runs?event=workflow_dispatch", "workflow_runs"
    ):
        if run.get("display_title") == expected:
            execution_identity(api, run, check, record)
            found.append(run)
    return sorted(found, key=lambda run: run["id"])


def behind_base(api, pr):
    comparison = f"{quote(pr['base']['ref'], safe='')}...{sha(pr['head']['sha'])}"
    behind = api.get(f"compare/{comparison}")["behind_by"]
    require(type(behind) is int and behind >= 0, "Invalid branch comparison.")
    return behind
