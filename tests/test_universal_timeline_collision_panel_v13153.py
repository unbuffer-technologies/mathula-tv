from __future__ import annotations

import socket
import threading
import time
from pathlib import Path
from urllib.request import urlopen

from mathula_tv.timeline_collision_panel import (
    discover_timeline_collision_reviews,
    ensure_universal_timeline_collision_panel,
    save_timeline_collision_resolution,
    timeline_collision_job_paths,
    upsert_timeline_collision_case,
    wait_for_timeline_collision_resolutions,
)


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _case(collision_id: str, shortfall_ms: int = 200) -> dict:
    return {
        "collision_id": collision_id,
        "reason": "test_collision",
        "shortfall_ms": shortfall_ms,
        "available_duration_ms": 1000,
        "measured_duration_ms": 1000 + shortfall_ms,
        "triplet": [],
    }


def _upsert(jobs_root: Path, job_id: str, collision_id: str) -> dict:
    paths = timeline_collision_job_paths(jobs_root, job_id)
    return upsert_timeline_collision_case(
        review_path=paths["review"],
        panel_path=paths["panel"],
        resolutions_path=paths["resolutions"],
        job_id=job_id,
        case=_case(collision_id),
    )


def test_universal_dashboard_discovers_multiple_jobs(tmp_path: Path) -> None:
    jobs_root = tmp_path / "jobs"
    _upsert(jobs_root, "job-a", "block_0002")
    _upsert(jobs_root, "job-b", "block_0044")

    reviews = discover_timeline_collision_reviews(jobs_root, include_resolved=False)
    assert {item["job_id"] for item in reviews} == {"job-a", "job-b"}

    port = _free_port()
    url = ensure_universal_timeline_collision_panel(
        jobs_root=jobs_root,
        port=port,
    )
    with urlopen(url, timeout=2) as response:
        page = response.read().decode("utf-8")
    assert "job-a" in page
    assert "job-b" in page
    assert "universal timeline collision panel" in page.lower()


def test_case_revision_rejects_stale_resolution(tmp_path: Path) -> None:
    jobs_root = tmp_path / "jobs"
    first = _upsert(jobs_root, "job-a", "block_0002")
    case = first["cases"][0]
    paths = timeline_collision_job_paths(jobs_root, "job-a")
    save_timeline_collision_resolution(
        resolutions_path=paths["resolutions"],
        review_path=paths["review"],
        panel_path=paths["panel"],
        job_id="job-a",
        collision_id="block_0002",
        case_token=case["case_token"],
        strategy="allow_overlap",
        reviewed_by="Reviewer",
        overlap_before_ms=100,
        overlap_after_ms=100,
    )

    reopened = _upsert(jobs_root, "job-a", "block_0002")
    new_case = reopened["cases"][0]
    assert new_case["case_revision"] == case["case_revision"] + 1
    assert new_case["case_token"] != case["case_token"]
    assert new_case["status"] == "open"


def test_waiting_dub_resumes_after_panel_decision(tmp_path: Path) -> None:
    jobs_root = tmp_path / "jobs"
    review = _upsert(jobs_root, "job-a", "block_0002")
    case = review["cases"][0]
    paths = timeline_collision_job_paths(jobs_root, "job-a")
    port = _free_port()
    result: dict = {}
    errors: list[BaseException] = []

    def waiter() -> None:
        try:
            result.update(
                wait_for_timeline_collision_resolutions(
                    review=review,
                    jobs_root=jobs_root,
                    port=port,
                    poll_interval_seconds=0.05,
                    timeout_seconds=5,
                )
            )
        except BaseException as exc:  # test captures worker failure
            errors.append(exc)

    thread = threading.Thread(target=waiter)
    thread.start()
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        try:
            with urlopen(f"http://127.0.0.1:{port}/health", timeout=0.2):
                break
        except OSError:
            time.sleep(0.05)

    save_timeline_collision_resolution(
        resolutions_path=paths["resolutions"],
        review_path=paths["review"],
        panel_path=paths["panel"],
        job_id="job-a",
        collision_id="block_0002",
        case_token=case["case_token"],
        strategy="manual_text",
        reviewed_by="Reviewer",
        manual_text="Umbhalo omfushane.",
    )
    thread.join(timeout=5)

    assert not thread.is_alive()
    assert not errors
    assert result["block_0002"]["manual_text"] == "Umbhalo omfushane."
