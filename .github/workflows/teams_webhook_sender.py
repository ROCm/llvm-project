#!/usr/bin/env python3
"""Post nightly build status to a Teams Workflow webhook (Power Automate).

Legacy Office 365 Connectors accepted {"title", "text"}. Microsoft retired those
connectors; "Send webhook alerts to a channel" expects an Adaptive Card inside
a message envelope. See:
https://learn.microsoft.com/en-us/microsoftteams/platform/webhooks-and-connectors/how-to/add-incoming-webhook
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from datetime import datetime
from zoneinfo import ZoneInfo


def build_adaptive_card_payload(title: str, body: str) -> dict:
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
                    "version": "1.4",
                    "msteams": {"width": "Full"},
                    "body": [
                        {
                            "type": "TextBlock",
                            "text": title,
                            "weight": "Bolder",
                            "size": "ExtraLarge",
                            "wrap": True,
                        },
                        {
                            "type": "TextBlock",
                            "text": body,
                            "wrap": True,
                        },
                    ],
                },
            }
        ],
    }


def normalize_table_text(text: str) -> str:
    # Older extractor output used literal "\\n" between markdown table rows.
    return text.replace("\\n", "\n")


def infer_status(results: dict | None) -> str:
    if not results:
        return "failure"
    failure_table = results.get("failure_table", "")
    if failure_table and failure_table != "No failures found":
        return "failure"
    return "success"


def build_body(
    *,
    repository: str,
    branch: str,
    status: str,
    run_url: str,
    results: dict | None,
) -> str:
    lines = [
        f"**Repository:** {repository}",
        f"**Branch:** {branch}",
        f"**Status:** {status}",
        "",
        f"🔗 [Build Link]({run_url})",
    ]

    if results:
        lines.extend(
            [
                "",
                "**Manifest, Artifacts, and Logs:**",
                "",
                normalize_table_text(results.get("manifest_artifacts_table", "")),
                "",
                "**Details:**",
                "",
                normalize_table_text(results.get("submodule_table", "")),
            ]
        )
        failure_table = results.get("failure_table", "")
        if failure_table and failure_table != "No failures found":
            lines.extend(
                [
                    "",
                    "**Failure Jobs:**",
                    "",
                    normalize_table_text(failure_table),
                ]
            )
    else:
        lines.extend(
            [
                "",
                "**Details:** Unable to fetch detailed results. Please check logs.",
            ]
        )

    return "\n".join(lines)


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
            body = response.read().decode("utf-8", errors="replace")
            print(f"Teams webhook HTTP {response.status}")
            if body.strip():
                print(body)
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        print(f"Teams webhook HTTP {exc.code}", file=sys.stderr)
        if error_body.strip():
            print(error_body, file=sys.stderr)
        raise SystemExit(1) from exc


def main() -> None:
    webhook_url = os.environ["TEAMS_WEBHOOK_URL"]
    results_path = os.environ.get("JSON_RESULT_FILE", "results.json")
    repository = os.environ["GITHUB_REPOSITORY"]
    branch = os.environ.get("GITHUB_REF_NAME", os.environ.get("GITHUB_REF", ""))
    run_id = os.environ["RUN_ID"]
    tz_name = os.environ.get("TZ", "America/Chicago")

    run_url = f"https://github.com/{repository}/actions/runs/{run_id}"
    start_date = datetime.now(ZoneInfo(tz_name)).strftime("%Y-%m-%d")

    results = None
    if os.path.isfile(results_path):
        with open(results_path, encoding="utf-8") as handle:
            results = json.load(handle)

    status = infer_status(results)
    icon = "✅" if status == "success" else "❌"
    title = f"{start_date} - {icon} Build {status}"
    body = build_body(
        repository=repository,
        branch=branch,
        status=status,
        run_url=run_url,
        results=results,
    )

    payload = build_adaptive_card_payload(title, body)
    print("Notification payload:")
    print(json.dumps(payload, indent=2))
    post_payload(webhook_url, payload)


if __name__ == "__main__":
    main()
