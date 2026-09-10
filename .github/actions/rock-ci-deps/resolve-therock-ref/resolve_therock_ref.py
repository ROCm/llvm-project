# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Reads the promoted TheRock SHA from the RockCI config JSON."""

import base64
import json
import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request

SHA_PATTERN = re.compile(r"[0-9a-fA-F]{40}")


def fail(message: str):
    print(f"::error::{message}")
    sys.exit(1)


def main() -> None:
    repository = os.environ["CONFIG_REPOSITORY"]
    config_path = os.environ["CONFIG_PATH"]
    config_ref = os.environ["CONFIG_REF"]
    api_url = os.environ.get("GITHUB_API_URL", "https://api.github.com")

    url = (
        f"{api_url}/repos/{repository}/contents/"
        f"{urllib.parse.quote(config_path)}"
        f"?ref={urllib.parse.quote(config_ref)}"
    )
    request = urllib.request.Request(url)
    request.add_header("Accept", "application/vnd.github+json")
    request.add_header("X-GitHub-Api-Version", "2022-11-28")
    token = os.environ.get("GH_TOKEN", "")
    if token:
        request.add_header("Authorization", f"Bearer {token}")

    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            payload = json.load(response)
    except urllib.error.HTTPError as error:
        fail(
            f"Could not read {config_path} from {repository}@{config_ref}: "
            f"HTTP {error.code} {error.reason}"
        )

    try:
        config = json.loads(base64.b64decode(payload["content"]).decode())
    except (KeyError, ValueError) as error:
        fail(f"{config_path} at {config_ref} is not valid JSON: {error}")

    ref = config.get("therock_ref", "")
    if not SHA_PATTERN.fullmatch(ref):
        fail(f"{config_path} must contain a full 40-character therock_ref.")

    ref = ref.lower()
    print(f"Resolved TheRock ref {ref} from {repository}@{config_ref}")
    with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as output:
        output.write(f"ref={ref}\n")


if __name__ == "__main__":
    main()
