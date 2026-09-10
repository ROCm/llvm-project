# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Post a RockCI SHA-PR Adaptive Card to the Teams workflow webhook."""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request


def build_payload(title: str, facts: list[dict[str, str]], details: str) -> dict:
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
                    "body": [
                        {
                            "type": "TextBlock",
                            "text": title,
                            "weight": "Bolder",
                            "size": "Large",
                            "wrap": True,
                        },
                        {"type": "FactSet", "facts": facts},
                        {
                            "type": "TextBlock",
                            "text": details,
                            "wrap": True,
                        },
                    ],
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

    title = os.environ["NOTIFY_TITLE"]
    details = os.environ.get("NOTIFY_DETAILS", "")
    facts = json.loads(os.environ.get("NOTIFY_FACTS_JSON", "[]"))
    payload = build_payload(title, facts, details)
    print(json.dumps(payload, indent=2))
    post_payload(webhook_url, payload)


if __name__ == "__main__":
    main()
