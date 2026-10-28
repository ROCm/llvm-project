# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Post the RockCI baseline-update Adaptive Card to the Teams workflow webhook.

Reads TheRock submodule gitlinks at the promoted rev so the card lists every
component SHA that the merged baseline pins.
"""

from __future__ import annotations

import base64
import json
import os
import re
import sys
from datetime import datetime, timezone
import urllib.error
import urllib.parse
import urllib.request

THEROCK_REPOSITORY = "ROCm/TheRock"
API_URL = os.environ.get("GITHUB_API_URL", "https://api.github.com")


def api_get(path: str) -> dict:
    request = urllib.request.Request(f"{API_URL}/{path}")
    request.add_header("Accept", "application/vnd.github+json")
    request.add_header("X-GitHub-Api-Version", "2022-11-28")
    token = os.environ.get("GH_TOKEN", "")
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


def commit_url(git_url: str, sha: str) -> str:
    repo = re.sub(r"\.git$", "", git_url.strip())
    repo = re.sub(r"^git@github\.com:", "https://github.com/", repo)
    return f"{repo}/commit/{sha}"


def gitmodule_paths(rev: str) -> list[tuple[str, str]]:
    quoted = urllib.parse.quote(rev)
    payload = api_get(f"repos/{THEROCK_REPOSITORY}/contents/.gitmodules?ref={quoted}")
    text = base64.b64decode(payload["content"]).decode("utf-8", errors="replace")
    entries: list[tuple[str, str]] = []
    path = url = ""
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("[submodule"):
            if path:
                entries.append((path, url))
            path = url = ""
        elif line.startswith("path"):
            path = line.split("=", 1)[1].strip()
        elif line.startswith("url"):
            url = line.split("=", 1)[1].strip()
    if path:
        entries.append((path, url))
    return sorted(entries)


def submodule_shas(rev: str) -> list[tuple[str, str, str]]:
    quoted_rev = urllib.parse.quote(rev)
    resolved: list[tuple[str, str, str]] = []
    for path, url in gitmodule_paths(rev):
        payload = api_get(
            f"repos/{THEROCK_REPOSITORY}/contents/"
            f"{urllib.parse.quote(path)}?ref={quoted_rev}"
        )
        if payload.get("type") != "submodule":
            continue
        resolved.append(
            (path, payload.get("sha", ""), payload.get("submodule_git_url", url))
        )
    return resolved


def text_block(text: str, **extra) -> dict:
    block = {"type": "TextBlock", "text": text, "wrap": True}
    block.update(extra)
    return block


def indented(text: str) -> dict:
    """Teams flattens nested markdown lists, so indent with a spacer column."""
    return {
        "type": "ColumnSet",
        "spacing": "None",
        "columns": [
            {"type": "Column", "width": "8px", "items": []},
            {"type": "Column", "width": "stretch", "items": [text_block(text)]},
        ],
    }


def build_payload(
    title: str,
    therock_rev: str,
    submodules: list[tuple[str, str, str]],
    pr_url: str,
    status: str,
) -> dict:
    rock_link = commit_url(f"https://github.com/{THEROCK_REPOSITORY}", therock_rev)
    lines = [
        f"- {path}: [{sha}]({commit_url(url, sha)})" for path, sha, url in submodules
    ]
    body = [
        text_block(title, weight="Bolder", size="Large"),
        text_block(f"- Rock main sha: [{therock_rev}]({rock_link})"),
    ]
    if lines:
        body.append(indented("\n\n".join(lines)))
    body.append(text_block(f"- PR which did the baseline update: {pr_url}"))
    body.append(text_block(f"- PR Status: **{status}**"))
    return {
        "type": "message",
        "summary": title,
        "attachments": [
            {
                "contentType": "application/vnd.microsoft.card.adaptive",
                "contentUrl": None,
                "content": {
                    "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
                    "type": "AdaptiveCard",
                    "version": "1.5",
                    "msteams": {"width": "Full"},
                    "body": body,
                },
            }
        ],
    }


def post_payload(webhook_url: str, payload: dict) -> None:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        webhook_url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            print(f"Teams webhook HTTP {response.status}")
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        print(f"Teams webhook HTTP {exc.code}", file=sys.stderr)
        if error_body.strip():
            print(error_body, file=sys.stderr)
        raise SystemExit(1) from exc


def main() -> None:
    webhook_url = os.environ.get("TEAMS_WEBHOOK_URL", "")
    if not webhook_url:
        print("TEAMS_WEBHOOK_URL is empty; skipping Teams post.")
        return

    therock_rev = os.environ["THEROCK_REV"]
    base_branch = os.environ.get("BASE_BRANCH", "amd-staging")
    status = os.environ.get("PR_STATUS", "success")
    raw_date = os.environ.get("CARD_DATE", "").strip()
    if len(raw_date) >= 10 and raw_date[4] == "-" and raw_date[7] == "-":
        card_date = raw_date[:10]
    else:
        card_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    pr_url = os.environ.get("PR_URL", "")

    title = (
        f"{base_branch} - PSDB/Nightly - rock main branch baseline-update {card_date}"
    )

    payload = build_payload(
        title, therock_rev, submodule_shas(therock_rev), pr_url, status
    )
    print(json.dumps(payload, indent=2))
    post_payload(webhook_url, payload)


if __name__ == "__main__":
    main()
