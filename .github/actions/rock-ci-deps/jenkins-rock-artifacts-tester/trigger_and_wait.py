#!/usr/bin/env python3
"""Trigger Jenkins rock-artifacts-tester and wait for its final result."""

import base64
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request


def required(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        sys.exit(f"Missing required environment variable {name}.")
    return value


def optional_int(name: str, default: int) -> int:
    value = os.environ.get(name, "").strip()
    return int(value) if value else default


def write_output(key: str, value: str) -> None:
    path = os.environ.get("JENKINS_OUTPUT_FILE", "").strip()
    if path:
        with open(path, "a", encoding="utf-8") as output:
            output.write(f"{key}={value}\n")


def auth_header(user: str, token: str) -> str:
    return "Basic " + base64.b64encode(f"{user}:{token}".encode()).decode()


def request(url: str, authorization: str, method: str = "GET"):
    call = urllib.request.Request(url, method=method)
    call.add_header("Authorization", authorization)
    try:
        return urllib.request.urlopen(call, timeout=60)
    except urllib.error.HTTPError as error:
        body = error.read().decode(errors="replace")
        sys.exit(f"{method} {url} failed: HTTP {error.code}: {body}")


def request_json(url: str, authorization: str, description: str) -> dict:
    with request(url, authorization) as response:
        try:
            return json.loads(response.read().decode())
        except json.JSONDecodeError as error:
            sys.exit(f"Failed to read {description}: {error}")


def wait_for_build_url(
    queue_url: str, authorization: str, poll_seconds: int, deadline: float
) -> str:
    while True:
        item = request_json(
            f"{queue_url}/api/json", authorization, "the Jenkins queue item"
        )
        if item.get("cancelled"):
            sys.exit("The queued Jenkins build was cancelled.")
        if item.get("executable"):
            return item["executable"]["url"].rstrip("/")
        if time.monotonic() > deadline:
            sys.exit("Timed out waiting for the Jenkins build to leave the queue.")
        print(
            f"Still queued: {item.get('why') or 'waiting for an executor'}",
            flush=True,
        )
        time.sleep(poll_seconds)


def wait_for_result(
    build_url: str, authorization: str, poll_seconds: int, deadline: float
) -> str:
    while True:
        build = request_json(
            f"{build_url}/api/json", authorization, "the Jenkins build"
        )
        if not build.get("building") and build.get("result"):
            return build["result"]
        if time.monotonic() > deadline:
            sys.exit(f"Timed out waiting for {build_url} to finish.")
        print("Jenkins build still running.", flush=True)
        time.sleep(poll_seconds)


def main() -> None:
    host = required("JENKINS_HOST").rstrip("/")
    job = required("JENKINS_JOB")
    run_id = required("ARTIFACTS_RUN_ID")
    poll_seconds = optional_int("POLL_INTERVAL_SECONDS", 60)
    deadline = time.monotonic() + optional_int("TIMEOUT_MINUTES", 300) * 60
    authorization = auth_header(required("JENKINS_USER"), required("JENKINS_TOKEN"))

    parameters = urllib.parse.urlencode(
        {
            "theRockArtifactsRunId": run_id,
            "gpuTarget": required("GPU_TARGET"),
        }
    )
    with request(
        f"{host}/job/{job}/buildWithParameters?{parameters}",
        authorization,
        method="POST",
    ) as response:
        queue_url = response.headers.get("Location", "").rstrip("/")
    if not queue_url:
        sys.exit("Jenkins accepted the trigger but returned no queue URL.")
    print(f"Queued {job} for GitHub Actions run {run_id}: {queue_url}", flush=True)

    build_url = wait_for_build_url(queue_url, authorization, poll_seconds, deadline)
    write_output("build_url", build_url)
    print(f"Jenkins build: {build_url}", flush=True)

    result = wait_for_result(build_url, authorization, poll_seconds, deadline)
    write_output("result", result)
    print(f"Jenkins build result: {result}", flush=True)
    if result != "SUCCESS":
        sys.exit(f"{job} finished as {result}: {build_url}")


if __name__ == "__main__":
    main()

