#!/usr/bin/env python3
"""Trigger Jenkins rock-artifacts-tester and wait for its final result."""

import os
import sys
import time

import requests
from requests.auth import HTTPBasicAuth


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


def response_json(response: requests.Response, description: str) -> dict:
    if response.status_code != 200:
        sys.exit(
            f"Failed to read {description}: "
            f"HTTP {response.status_code}: {response.text}"
        )
    return response.json()


def wait_for_build_url(
    session: requests.Session, queue_url: str, poll_seconds: int, deadline: float
) -> str:
    while True:
        item = response_json(
            session.get(f"{queue_url}/api/json", timeout=60),
            "the Jenkins queue item",
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
    session: requests.Session, build_url: str, poll_seconds: int, deadline: float
) -> str:
    while True:
        build = response_json(
            session.get(f"{build_url}/api/json", timeout=60),
            "the Jenkins build",
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

    session = requests.Session()
    session.auth = HTTPBasicAuth(required("JENKINS_USER"), required("JENKINS_TOKEN"))
    response = session.post(
        f"{host}/job/{job}/buildWithParameters",
        params={
            "theRockArtifactsRunId": run_id,
            "gpuTarget": required("GPU_TARGET"),
        },
        timeout=60,
    )
    if response.status_code not in (200, 201):
        sys.exit(
            f"Jenkins trigger failed: HTTP {response.status_code}: {response.text}"
        )

    queue_url = response.headers.get("Location", "").rstrip("/")
    if not queue_url:
        sys.exit("Jenkins accepted the trigger but returned no queue URL.")
    print(f"Queued {job} for GitHub Actions run {run_id}: {queue_url}", flush=True)

    build_url = wait_for_build_url(session, queue_url, poll_seconds, deadline)
    write_output("build_url", build_url)
    print(f"Jenkins build: {build_url}", flush=True)

    result = wait_for_result(session, build_url, poll_seconds, deadline)
    write_output("result", result)
    print(f"Jenkins build result: {result}", flush=True)
    if result != "SUCCESS":
        sys.exit(f"{job} finished as {result}: {build_url}")


if __name__ == "__main__":
    main()
