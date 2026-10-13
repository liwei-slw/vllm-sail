# Contributing to vLLM SAIL

Contributions to documentation, tests, PPU kernels and performance are welcome.
Check existing issues and pull requests before starting. Discuss substantial
API, dependency or architecture changes in an issue first.

## Development environment

The CPU test suite runs with Python 3.10–3.13 and does not require torch, vLLM,
the SAIL SDK or a device. From a checkout of this repository:

```bash
python3 -m venv .venv-test
source .venv-test/bin/activate
python -m pip install -r requirements/dev.txt
python -m pytest tests/ut -q
```

The package is imported directly from the checkout for these tests. The expected
`vllm_sail._version is missing` warning can appear before the package is installed.
Optional integration tests skip when their dependencies are unavailable.

For native compilation or device testing, use the separate
[PPU installation instructions](docs/getting_started/installation.md).

## Coding guidelines

- Keep changes focused and describe the behavior they change.
- Prefer explicit interfaces and shared helpers when they remove real duplication.
- Preserve lazy imports and entry-point idempotency. Core imports and CPU tests
  must remain independent of optional SDK packages.
- Use the existing registration APIs and platform hooks before adding a patch.
  Read the [runtime patch guide](vllm_sail/patch/README.md) for metadata and copied
  source rules.
- Edit plugin-owned HGGC kernels directly in `csrc/plugin/`. For upstream
  kernels, edit generation inputs and regenerate; follow
  [HGGC kernel development](docs/developer_guide/kernels.md).
- Preserve upstream attribution and license headers. New Python files should
  use `# SPDX-License-Identifier: Apache-2.0`.
- Keep user documentation focused on reproducible PPU workflows. Personal
  run logs, implementation phase records and private environment notes belong
  outside the repository.

See [Architecture](docs/developer_guide/architecture.md) for module boundaries.

## Checks before review

Run the CPU tests and formatting checks:

```bash
python -m pytest tests/ut -q
pre-commit run --files path/to/changed_file.py
```

For staged files or a full repository check:

```bash
pre-commit run
pre-commit run --all-files --hook-stage manual
```

Add tests for behavior changes and keep formatting changes focused. For native
or runtime changes, follow [Verification](docs/user_guide/verification_guide.md)
and report any checks that were not run. Verify links and commands in docs.

## Pull request workflow

1. Fork the repository and create a topic branch for the change.
2. Implement the change and update the relevant tests and documentation.
3. Run the applicable checks and inspect the diff.
4. Push the topic branch to your fork and open a pull request.
5. Address review feedback; merge through the platform after review.

Use vLLM-style PR titles: `[Type][Scope] Concise behavior-oriented summary`,
with an optional scope. For example, `[Bugfix][MoE] Honor explicit backend
selection` or `[Doc] Clarify native build prerequisites`.

Complete the [PR template](.github/pull_request_template.md), link related issues
and include test commands and results. Performance changes need baseline and
candidate measurements under the same workload. Avoid force pushes and direct
pushes to shared branches; use follow-up commits for review changes.

## Requesting PPU CI

CPU unit tests, upstream source contracts and repository security checks remain
automatic. The comment interface controls only the PPU wheel build, pin unit
tests, Qwen smoke test and PPU gate. During rollout, the existing automatic PPU
PR workflow remains enabled until the on-demand path has been verified on the
default branch. Commands reuse an active automatic run instead of duplicating it.

Post one of these commands as the entire text of a **new PR comment**:

| Command | Behavior |
| --- | --- |
| `/ci run` | Build and test the current PR head using the existing PPU PR chain. |
| `/ci run --allow-stale` | Run even if the head is behind its actual base branch; merge protection is unchanged. |
| `/ci retry` | Retry failed jobs from an existing execution of the current head. Successful prerequisites may be reused. |
| `/ci cancel` | Cancel this PR's on-demand PPU requests, including older heads; CPU checks and other PRs are unaffected. |

An ordinary author needs an open, non-Draft PR and either a `ready` or
`ready-run-all-tests` label, or a current trusted approval with the PR's overall
review decision `APPROVED`. Historical or dismissed approvals do not authorize a
run. Maintainers with `write`, `maintain` or `admin` permission, and users listed
in the optional `CI_TRUSTED_USERS` repository variable (comma- or whitespace-
separated GitHub logins), can run open PRs directly, including Drafts. Permissions
are checked again when processing a command. `author_association` is not sufficient.

Adding a ready label only posts help; it does not launch tests. Approval is
checked when a command is processed. `/ci run all`, `/ci run nightly`, edited
comments, code blocks and arbitrary arguments are not supported. Manual operator
qualification and historical-wheel debugging remain separate and do not provide
PR gate evidence. Do not dispatch `PPU PR CI` directly: it requires a trusted
request created by the comment controller.

Each request binds the repository, PR number, head and base SHAs, and the trusted
control workflow revision. Checks belong to that head only. A new commit needs a
new `/ci run`; `/ci retry` never transfers failures or success from an old head.
Once automatic PPU triggering is retired, a newly pushed head has missing /
**Expected** PPU checks until explicitly tested. An old green result does not
satisfy the new head's gate. Branch synchronization, approvals and required checks
are never bypassed. The rollout still retains automatic PPU triggering for now.

Commands are serialized per PR. A duplicate or active request returns its link.
If dispatch acceptance is uncertain, checks remain queued: ask a maintainer to
inspect the linked controller/executor run instead of repeatedly dispatching.
Cancellation invalidates the request before asking Actions to stop it. Missing,
skipped, cancelled or timed-out required jobs cannot pass the gate. Reporter
failures leave the gate non-successful; inspect the `PPU CI report` run and rerun
that trusted reporting workflow after resolving the API failure.

### Maintainer deployment checks

The comment, dispatch and reporting workflows must be present on the default
branch. Create the ready labels before advertising them. Verify that organization
policy permits the controller's `actions: write`, `checks: write` and
`issues: write`; a policy API 403 is not proof that workflow tokens have these
permissions. If policy blocks them, ask the organization maintainer to resolve it;
do not substitute a personal token or weaken branch protection.

PPU workflows explicitly use GitHub's `cache-mode: none`, propagated through the
reusable chain and enforced by scoped cache tokens. Jobs verify the effective
`ACTIONS_CACHE_MODE` before executing PR code. This prevents default-branch cache
writes from dispatch-based PR execution. Test jobs have read-only repository and
Actions permissions, no inherited secrets and no persisted checkout credentials.

Before retiring the automatic PPU trigger in a separate PR, verify on GitHub:
authorization and notification without scheduling, fork PR execution, the exact
head SHA and all four checks, retry with reused prerequisites, cancellation and
cache enforcement. Local CPU tests cannot establish platform or device success.
After the switch, also verify that a new push does not start PPU work and that its
head cannot reuse the old green checks. Keep all existing required checks.

For workflow linting, actionlint 1.7.12 predates the official `cache-mode` key.
Until actionlint supports it, the compatibility exception is restricted to
`unexpected key "cache-mode" for "workflow" section`; the pipeline contract tests
must still verify `none` and reusable-workflow propagation. Do not ignore other
syntax errors or security findings. Existing self-hosted runner-label warnings
may be configured separately.

## Reporting issues

Use the bug or feature request template in the repository's Issues tab. Bug
reports should include a minimal reproducer, expected and actual results, the
full relevant error and `collect_env` output. Remove credentials, private host
addresses and confidential model paths from logs before posting.

Keep discussions respectful and focused on the work. Explain technical
disagreements with examples, measurements or source evidence.

## License

Contributions are made under the project's [Apache License 2.0](LICENSE).
Retain the applicable notices when adapting third-party code.
