"""Universal human review service for direct-dub timeline collisions.

Each job keeps its own immutable collision evidence and reviewed direct-dub
resolution files.  The web service is global: it scans every job beneath the
Mathula TV jobs root, displays all unresolved cases in one dashboard, and saves
resolutions back into the owning job.

`dub-azure` may start the localhost service automatically, wait for the exact
case revision to be resolved, and then retry its cached render in the same CLI
process.  The authoritative translation artifact is never rewritten.
"""

from __future__ import annotations

import hashlib
import html
import json
import threading
import time
import uuid
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
from urllib.error import URLError
from urllib.parse import parse_qs, urlparse
from urllib.request import urlopen

from .atomic_io import atomic_write_json, read_json

REVIEW_SCHEMA_VERSION = "mathula.timeline-collision-review.v2"
RESOLUTION_SCHEMA_VERSION = "mathula.timeline-collision-resolutions.v2"
PANEL_VERSION = "mathula.timeline-collision-panel.v3-universal-auto-resume"
UNIVERSAL_SERVICE_VERSION = "mathula.timeline-collision-service.v2-auto-resume"
ALLOWED_STRATEGIES = {"allow_overlap", "manual_text", "select_variants"}
DEFAULT_PANEL_HOST = "127.0.0.1"
DEFAULT_PANEL_PORT = 8765

_SERVER_LOCK = threading.Lock()
_SERVER_THREADS: dict[tuple[str, str, int], threading.Thread] = {}
_SERVER_ERRORS: dict[tuple[str, str, int], BaseException] = {}


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_read(path: Path, default: Mapping[str, Any]) -> dict[str, Any]:
    if not path.is_file():
        return dict(default)
    try:
        value = read_json(path)
    except Exception:
        return dict(default)
    return dict(value) if isinstance(value, Mapping) else dict(default)


def _safe_job_id(value: str) -> str:
    job_id = str(value or "").strip()
    if not job_id or any(
        character
        not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
        for character in job_id
    ):
        raise ValueError("Invalid job ID")
    return job_id


def _job_direct_dub_root(jobs_root: Path, job_id: str) -> Path:
    root = Path(jobs_root).resolve()
    candidate = (root / _safe_job_id(job_id) / "direct_dub").resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError("Job path escapes the configured jobs root") from exc
    return candidate


def timeline_collision_job_paths(jobs_root: Path, job_id: str) -> dict[str, Path]:
    root = _job_direct_dub_root(jobs_root, job_id)
    return {
        "root": root,
        "review": root / "timeline_collision_review.json",
        "panel": root / "timeline_collision_panel.html",
        "resolutions": root / "timeline_collision_resolutions.json",
    }


def load_timeline_collision_review(path: Path) -> dict[str, Any]:
    return _safe_read(
        path,
        {
            "schema_version": REVIEW_SCHEMA_VERSION,
            "status": "clear",
            "cases": [],
        },
    )


def load_timeline_collision_resolutions(path: Path) -> dict[str, Any]:
    return _safe_read(
        path,
        {
            "schema_version": RESOLUTION_SCHEMA_VERSION,
            "resolutions": [],
        },
    )


def resolution_map(path: Path) -> dict[str, dict[str, Any]]:
    payload = load_timeline_collision_resolutions(path)
    values: dict[str, dict[str, Any]] = {}
    for item in payload.get("resolutions") or []:
        if not isinstance(item, Mapping):
            continue
        collision_id = str(item.get("collision_id") or "").strip()
        if collision_id:
            values[collision_id] = dict(item)
    return values


def _case_digest(case: Mapping[str, Any]) -> str:
    ignored = {
        "status",
        "updated_at",
        "created_at",
        "saved_resolution",
        "case_token",
        "case_revision",
    }
    payload = {key: value for key, value in case.items() if key not in ignored}
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _new_case_token(job_id: str, collision_id: str, revision: int) -> str:
    nonce = uuid.uuid4().hex
    return f"{job_id}:{collision_id}:r{revision}:{nonce}"


def _current_case(review: Mapping[str, Any], collision_id: str) -> dict[str, Any] | None:
    for item in review.get("cases") or []:
        if not isinstance(item, Mapping):
            continue
        if str(item.get("collision_id") or "") == collision_id:
            return dict(item)
    return None


def upsert_timeline_collision_case(
    *,
    review_path: Path,
    panel_path: Path,
    resolutions_path: Path,
    job_id: str,
    case: Mapping[str, Any],
) -> dict[str, Any]:
    """Insert or reopen one case and assign a revision-specific wait token.

    A saved resolution is valid only for the exact case token it reviewed.  If
    the renderer retries and the same collision still fails, this function
    advances the revision so the waiting process cannot accidentally reuse the
    unsuccessful decision.
    """

    job_id = _safe_job_id(job_id)
    collision_id = str(case.get("collision_id") or "").strip()
    if not collision_id:
        raise ValueError("Timeline collision case requires collision_id")

    payload = load_timeline_collision_review(review_path)
    cases = [
        dict(item) for item in payload.get("cases") or [] if isinstance(item, Mapping)
    ]
    resolutions = resolution_map(resolutions_path)
    incoming_digest = _case_digest(case)

    existing_index: int | None = None
    existing: dict[str, Any] | None = None
    for index, value in enumerate(cases):
        if str(value.get("collision_id") or "") == collision_id:
            existing_index = index
            existing = value
            break

    if existing is None:
        revision = 1
        case_token = _new_case_token(job_id, collision_id, revision)
        created_at = _utcnow()
    else:
        revision = int(existing.get("case_revision") or 1)
        case_token = str(existing.get("case_token") or "")
        created_at = str(existing.get("created_at") or _utcnow())
        saved = resolutions.get(collision_id)
        saved_token = str((saved or {}).get("case_token") or "")
        previous_digest = str(existing.get("case_digest") or "")
        resolution_was_consumed = bool(saved_token and saved_token == case_token)
        evidence_changed = bool(previous_digest and previous_digest != incoming_digest)
        if not case_token or resolution_was_consumed or evidence_changed:
            revision += 1 if case_token else 0
            case_token = _new_case_token(job_id, collision_id, revision)

    replacement = {
        **dict(case),
        "collision_id": collision_id,
        "job_id": job_id,
        "status": "open",
        "case_revision": revision,
        "case_token": case_token,
        "case_digest": incoming_digest,
        "created_at": created_at,
        "updated_at": _utcnow(),
    }
    saved_resolution = resolutions.get(collision_id)
    if saved_resolution:
        replacement["previous_resolution"] = saved_resolution
        if str(saved_resolution.get("case_token") or "") == case_token:
            replacement["saved_resolution"] = saved_resolution
            replacement["status"] = "resolution_saved"

    if existing_index is None:
        cases.append(replacement)
    else:
        cases[existing_index] = replacement

    output = {
        "schema_version": REVIEW_SCHEMA_VERSION,
        "panel_version": PANEL_VERSION,
        "job_id": job_id,
        "status": "review_required",
        "updated_at": _utcnow(),
        "review_path": str(review_path),
        "panel_path": str(panel_path),
        "resolutions_path": str(resolutions_path),
        "cases": cases,
        "open_case_count": sum(
            1
            for item in cases
            if item.get("status") in {"open", "resolution_failed_review_again"}
        ),
        "resolved_case_count": sum(
            1 for item in cases if item.get("status") == "resolution_saved"
        ),
        "resume_mode": "automatic_waiting_dub_process",
    }
    review_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(review_path, output)
    panel_path.write_text(render_timeline_collision_panel(output), encoding="utf-8")
    return output


def save_timeline_collision_resolution(
    *,
    resolutions_path: Path,
    review_path: Path,
    panel_path: Path,
    job_id: str,
    collision_id: str,
    strategy: str,
    reviewed_by: str,
    case_token: str | None = None,
    overlap_before_ms: int | None = None,
    overlap_after_ms: int | None = None,
    manual_text: str | None = None,
    previous_variant_id: str | None = None,
    current_variant_id: str | None = None,
    following_variant_id: str | None = None,
) -> dict[str, Any]:
    job_id = _safe_job_id(job_id)
    collision_id = collision_id.strip()
    strategy = strategy.strip()
    reviewed_by = reviewed_by.strip()
    if not collision_id:
        raise ValueError("collision_id is required")
    if strategy not in ALLOWED_STRATEGIES:
        raise ValueError(f"Unsupported timeline collision strategy: {strategy}")
    if not reviewed_by:
        raise ValueError("reviewed_by is required")

    review = load_timeline_collision_review(review_path)
    current = _current_case(review, collision_id)
    if current is None:
        raise ValueError(f"Timeline collision case does not exist: {collision_id}")
    current_token = str(current.get("case_token") or "")
    supplied_token = str(case_token or current_token)
    if not current_token:
        raise ValueError("Timeline collision case is missing its revision token")
    if supplied_token != current_token:
        raise ValueError(
            "This collision panel is stale; reload it before saving a decision"
        )

    resolution: dict[str, Any] = {
        "collision_id": collision_id,
        "job_id": job_id,
        "case_token": current_token,
        "case_revision": int(current.get("case_revision") or 1),
        "strategy": strategy,
        "reviewed_by": reviewed_by,
        "reviewed_at": _utcnow(),
        "human_review_required": True,
        "scope": "direct_dub_timeline_only",
        "authoritative_translation_changed": False,
    }
    if strategy == "allow_overlap":
        before = int(overlap_before_ms or 0)
        after = int(overlap_after_ms or 0)
        if before < 0 or after < 0:
            raise ValueError("Overlap allowances must be non-negative")
        if before + after <= 0:
            raise ValueError("At least one overlap allowance must be positive")
        if before > 3000 or after > 3000 or before + after > 5000:
            raise ValueError("Manual overlap exceeds the 5-second safety ceiling")
        resolution.update(
            overlap_before_ms=before,
            overlap_after_ms=after,
            max_total_overlap_ms=before + after,
        )
    elif strategy == "manual_text":
        text = " ".join((manual_text or "").split()).strip()
        if not text:
            raise ValueError("manual_text strategy requires non-empty text")
        resolution["manual_text"] = text
    elif strategy == "select_variants":
        variants = {
            "previous": (previous_variant_id or "").strip(),
            "current": (current_variant_id or "").strip(),
            "following": (following_variant_id or "").strip(),
        }
        if not any(variants.values()):
            raise ValueError("Select at least one prepared variant")
        invalid = [
            value
            for value in variants.values()
            if value and value not in {"natural", "concise", "compact"}
        ]
        if invalid:
            raise ValueError(f"Unsupported prepared variant: {invalid[0]}")
        resolution["variants"] = variants

    payload = load_timeline_collision_resolutions(resolutions_path)
    values = [
        dict(item)
        for item in payload.get("resolutions") or []
        if isinstance(item, Mapping)
        and str(item.get("collision_id") or "") != collision_id
    ]
    values.append(resolution)
    output = {
        "schema_version": RESOLUTION_SCHEMA_VERSION,
        "job_id": job_id,
        "updated_at": _utcnow(),
        "resolutions": values,
    }
    resolutions_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(resolutions_path, output)

    for item in review.get("cases") or []:
        if isinstance(item, dict) and str(item.get("collision_id") or "") == collision_id:
            item["saved_resolution"] = resolution
            item["status"] = "resolution_saved"
    if review.get("cases"):
        review["updated_at"] = _utcnow()
        review["open_case_count"] = sum(
            1
            for item in review["cases"]
            if item.get("status") in {"open", "resolution_failed_review_again"}
        )
        review["resolved_case_count"] = sum(
            1 for item in review["cases"] if item.get("status") == "resolution_saved"
        )
        atomic_write_json(review_path, review)
        panel_path.write_text(render_timeline_collision_panel(review), encoding="utf-8")
    return resolution


def clear_timeline_collision_resolution(
    *,
    resolutions_path: Path,
    review_path: Path,
    panel_path: Path,
    collision_id: str,
) -> None:
    payload = load_timeline_collision_resolutions(resolutions_path)
    payload["resolutions"] = [
        item
        for item in payload.get("resolutions") or []
        if str((item or {}).get("collision_id") or "") != collision_id
    ]
    payload["updated_at"] = _utcnow()
    atomic_write_json(resolutions_path, payload)
    review = load_timeline_collision_review(review_path)
    for case in review.get("cases") or []:
        if isinstance(case, dict) and str(case.get("collision_id") or "") == collision_id:
            case.pop("saved_resolution", None)
            case["status"] = "open"
    review["updated_at"] = _utcnow()
    review["open_case_count"] = sum(
        1 for item in review.get("cases") or [] if item.get("status") == "open"
    )
    atomic_write_json(review_path, review)
    panel_path.write_text(render_timeline_collision_panel(review), encoding="utf-8")


def discover_timeline_collision_reviews(
    jobs_root: Path,
    *,
    include_resolved: bool = True,
) -> list[dict[str, Any]]:
    jobs_root = Path(jobs_root)
    results: list[dict[str, Any]] = []
    if not jobs_root.is_dir():
        return results
    for review_path in sorted(
        jobs_root.glob("*/direct_dub/timeline_collision_review.json")
    ):
        try:
            review = load_timeline_collision_review(review_path)
            job_id = _safe_job_id(
                str(review.get("job_id") or review_path.parents[1].name)
            )
        except Exception:
            continue
        paths = timeline_collision_job_paths(jobs_root, job_id)
        review["job_id"] = job_id
        review["review_path"] = str(paths["review"])
        review["panel_path"] = str(paths["panel"])
        review["resolutions_path"] = str(paths["resolutions"])
        cases = [
            dict(item)
            for item in review.get("cases") or []
            if isinstance(item, Mapping)
        ]
        if not include_resolved:
            cases = [
                item
                for item in cases
                if item.get("status")
                in {"open", "resolution_failed_review_again"}
            ]
        review["cases"] = cases
        review["open_case_count"] = sum(
            1
            for item in cases
            if item.get("status") in {"open", "resolution_failed_review_again"}
        )
        if cases or include_resolved:
            results.append(review)
    results.sort(
        key=lambda item: (
            int(item.get("open_case_count") or 0) == 0,
            str(item.get("updated_at") or ""),
        )
    )
    return results


def _variant_rows(block: Mapping[str, Any]) -> str:
    variants = block.get("variants") or []
    rows: list[str] = []
    for variant in variants:
        if not isinstance(variant, Mapping):
            continue
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(variant.get('variant_id') or ''))}</td>"
            f"<td>{html.escape(str(variant.get('spoken_text') or ''))}</td>"
            f"<td>{html.escape(str(variant.get('estimated_duration_ms') or ''))}</td>"
            "</tr>"
        )
    return "".join(rows) or '<tr><td colspan="3">No prepared variants</td></tr>'


def _block_card(role: str, block: Mapping[str, Any]) -> str:
    return f"""
    <section class="block-card">
      <h4>{html.escape(role.title())}: {html.escape(str(block.get('block_id') or 'unknown'))}</h4>
      <p><strong>Speaker:</strong> {html.escape(str(block.get('speaker_id') or ''))}</p>
      <p><strong>Window:</strong> {html.escape(str(block.get('start_ms') or ''))}–{html.escape(str(block.get('end_ms') or ''))} ms</p>
      <p><strong>Source:</strong> {html.escape(str(block.get('source_text') or ''))}</p>
      <p><strong>Selected text:</strong> {html.escape(str(block.get('translated_text') or ''))}</p>
      <table><thead><tr><th>Variant</th><th>Spoken text</th><th>Est. ms</th></tr></thead><tbody>{_variant_rows(block)}</tbody></table>
    </section>
    """


def _case_html(job_id: str, case: Mapping[str, Any]) -> str:
    collision_id = str(case.get("collision_id") or "")
    case_token = str(case.get("case_token") or "")
    triplet = {
        str(item.get("role") or ""): item.get("block") or {}
        for item in case.get("triplet") or []
        if isinstance(item, Mapping)
    }
    shortfall = int(case.get("shortfall_ms") or 0)
    suggested_before = max(0, shortfall // 2)
    suggested_after = max(0, shortfall - suggested_before)
    current_text = str((triplet.get("current") or {}).get("translated_text") or "")
    saved = (
        case.get("saved_resolution")
        if isinstance(case.get("saved_resolution"), Mapping)
        else None
    )
    saved_html = (
        '<div class="saved"><strong>Decision saved.</strong> The waiting '
        "<code>dub-azure</code> process will resume automatically.</div>"
        if saved
        else ""
    )
    cards = "".join(
        _block_card(role, triplet.get(role) or {})
        for role in ("previous", "current", "following")
    )
    hidden = (
        f'<input type="hidden" name="job_id" value="{html.escape(job_id)}">'
        f'<input type="hidden" name="collision_id" value="{html.escape(collision_id)}">'
        f'<input type="hidden" name="case_token" value="{html.escape(case_token)}">'
    )
    return f"""
    <article class="collision" id="{html.escape(job_id)}-{html.escape(collision_id)}">
      <header><h2>{html.escape(collision_id)}</h2><span class="status">{html.escape(str(case.get('status') or 'open'))}</span></header>
      <p><strong>Revision:</strong> {html.escape(str(case.get('case_revision') or 1))}</p>
      <p><strong>Reason:</strong> {html.escape(str(case.get('reason') or 'timeline_collision'))}</p>
      <p><strong>Measured:</strong> {html.escape(str(case.get('measured_duration_ms') or ''))} ms; <strong>available:</strong> {html.escape(str(case.get('available_duration_ms') or ''))} ms; <strong>shortfall:</strong> {shortfall} ms.</p>
      {saved_html}
      <div class="triplet">{cards}</div>
      <div class="options">
        <form method="post" action="/resolve">{hidden}<input type="hidden" name="strategy" value="allow_overlap">
          <h3>Option 1 — allow commission interjection overlap</h3>
          <label>Overlap before (ms)<input name="overlap_before_ms" type="number" min="0" max="3000" value="{suggested_before}"></label>
          <label>Overlap after (ms)<input name="overlap_after_ms" type="number" min="0" max="3000" value="{suggested_after}"></label>
          <label>Reviewed by<input name="reviewed_by" required></label>
          <button>Save overlap decision</button>
        </form>
        <form method="post" action="/resolve">{hidden}<input type="hidden" name="strategy" value="manual_text">
          <h3>Option 2 — shorter direct-dub text override</h3>
          <textarea name="manual_text" rows="5" required>{html.escape(current_text)}</textarea>
          <label>Reviewed by<input name="reviewed_by" required></label>
          <button>Save manual text</button>
        </form>
        <form method="post" action="/resolve">{hidden}<input type="hidden" name="strategy" value="select_variants">
          <h3>Option 3 — force prepared variants across the triplet</h3>
          {''.join(f'<label>{role.title()}<select name="{role}_variant_id"><option value="">Keep automatic</option><option>natural</option><option>concise</option><option>compact</option></select></label>' for role in ('previous','current','following'))}
          <label>Reviewed by<input name="reviewed_by" required></label>
          <button>Save variant choice</button>
        </form>
        <form method="post" action="/clear">{hidden}<button class="secondary">Clear saved decision</button></form>
      </div>
    </article>
    """


def _styles() -> str:
    return """
body{font-family:system-ui,sans-serif;margin:0;background:#f4f5f7;color:#171717}main{max-width:1500px;margin:auto;padding:24px}.top{position:sticky;top:0;background:#f4f5f7eF;backdrop-filter:blur(8px);padding:12px 0;z-index:4}.job{background:#eef3ff;border:1px solid #b9c9ef;border-radius:14px;padding:16px;margin:22px 0}.collision{background:white;border:1px solid #ddd;border-radius:12px;padding:20px;margin:16px 0;box-shadow:0 2px 10px #0001}.collision header{display:flex;justify-content:space-between;align-items:center;gap:12px}.status,.saved,.notice{background:#fff3cd;border:1px solid #ffe69c;border-radius:6px;padding:8px}.saved{background:#d1e7dd;border-color:#a3cfbb}.triplet{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}.block-card{border:1px solid #ddd;border-radius:8px;padding:12px;overflow:auto;background:white}table{border-collapse:collapse;width:100%;font-size:13px}th,td{border:1px solid #ddd;padding:6px;text-align:left;vertical-align:top}.options{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px;margin-top:16px}.options form{border:1px solid #ddd;border-radius:8px;padding:12px;background:#fafafa}label{display:block;margin:8px 0}input,select,textarea{box-sizing:border-box;width:100%;padding:8px;margin-top:4px}button{padding:10px 14px;border:0;border-radius:6px;background:#111;color:white;cursor:pointer}button.secondary{background:#666}code{white-space:pre-wrap;word-break:break-word}.empty{background:white;border-radius:12px;padding:30px;border:1px solid #ddd}@media(max-width:1000px){.triplet,.options{grid-template-columns:1fr}}
"""


def render_timeline_collision_panel(review: Mapping[str, Any]) -> str:
    """Render one job for backward-compatible file artifacts."""
    job_id = str(review.get("job_id") or "")
    body = "".join(
        _case_html(job_id, case)
        for case in review.get("cases") or []
        if isinstance(case, Mapping)
    ) or '<div class="empty">No timeline collisions are awaiting review.</div>'
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Mathula TV timeline collision review</title><style>{_styles()}</style></head><body><main><h1>Mathula TV timeline collision review</h1><p>Job <code>{html.escape(job_id)}</code>. Decisions are direct-dub overlays; the authoritative translation is not rewritten.</p><p class="notice">After saving a decision, the waiting <code>dub-azure</code> process resumes automatically. Do not rerun it.</p>{body}</main></body></html>"""


def render_universal_timeline_collision_panel(
    reviews: Sequence[Mapping[str, Any]],
) -> str:
    pending = sum(int(item.get("open_case_count") or 0) for item in reviews)
    job_sections: list[str] = []
    for review in reviews:
        job_id = str(review.get("job_id") or "")
        cases = [
            item for item in review.get("cases") or [] if isinstance(item, Mapping)
        ]
        if not cases:
            continue
        cases_html = "".join(_case_html(job_id, case) for case in cases)
        job_sections.append(
            f'<section class="job"><h2>Job <code>{html.escape(job_id)}</code></h2>'
            f'<p>{int(review.get("open_case_count") or 0)} open collision(s)</p>'
            f"{cases_html}</section>"
        )
    body = "".join(job_sections) or (
        '<div class="empty"><h2>No pending timeline collisions</h2>'
        "<p>This universal panel is ready. Any running <code>dub-azure</code> job "
        "will appear here automatically when it needs a decision.</p></div>"
    )
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Mathula TV universal timeline collision panel</title><style>{_styles()}</style></head><body><main><div class="top"><h1>Mathula TV universal timeline collision panel</h1><p><strong>{pending}</strong> unresolved collision(s) across all jobs.</p><p class="notice">Save a decision here and the corresponding waiting <code>dub-azure</code> process resumes automatically. No job ID or second dub command is required.</p></div>{body}</main></body></html>"""


def timeline_collision_summary(review: Mapping[str, Any]) -> str:
    lines = [
        f"Timeline collision review for job {review.get('job_id', '')}",
        f"Cases: {len(review.get('cases') or [])}",
    ]
    for case in review.get("cases") or []:
        if not isinstance(case, Mapping):
            continue
        lines.append(
            "- "
            f"{case.get('collision_id')}: {case.get('reason')} | "
            f"shortfall={case.get('shortfall_ms', 'unknown')}ms | "
            f"status={case.get('status', 'open')}"
        )
    return "\n".join(lines)


def universal_timeline_collision_summary(jobs_root: Path) -> str:
    reviews = discover_timeline_collision_reviews(jobs_root, include_resolved=False)
    lines = [
        "Mathula TV universal timeline collision panel",
        f"Jobs with pending cases: {len(reviews)}",
        f"Pending cases: {sum(len(item.get('cases') or []) for item in reviews)}",
    ]
    for review in reviews:
        lines.append(f"- {review.get('job_id')}: {len(review.get('cases') or [])} case(s)")
    return "\n".join(lines)


def _health_payload(jobs_root: Path, host: str, port: int) -> dict[str, Any]:
    return {
        "service": UNIVERSAL_SERVICE_VERSION,
        "panel_version": PANEL_VERSION,
        "jobs_root": str(Path(jobs_root).resolve()),
        "host": host,
        "port": port,
    }


def _server_health(
    *,
    jobs_root: Path,
    host: str,
    port: int,
    timeout_seconds: float = 0.4,
) -> tuple[bool, dict[str, Any] | None]:
    try:
        with urlopen(
            f"http://{host}:{port}/health", timeout=timeout_seconds
        ) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (OSError, URLError, ValueError, json.JSONDecodeError):
        return False, None
    if not isinstance(payload, Mapping):
        return False, None
    same_service = payload.get("service") == UNIVERSAL_SERVICE_VERSION
    same_root = str(payload.get("jobs_root") or "") == str(Path(jobs_root).resolve())
    return bool(same_service and same_root), dict(payload)


def serve_universal_timeline_collision_panel(
    *,
    jobs_root: Path,
    host: str = DEFAULT_PANEL_HOST,
    port: int = DEFAULT_PANEL_PORT,
    announce: bool = True,
) -> None:
    if host not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("Timeline collision panel may only bind to localhost")
    jobs_root = Path(jobs_root).resolve()

    class Handler(BaseHTTPRequestHandler):
        def _send_json(self, payload: Mapping[str, Any], status: int = 200) -> None:
            data = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _redirect(self, location: str = "/") -> None:
            self.send_response(303)
            self.send_header("Location", location)
            self.end_headers()

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path == "/health":
                self._send_json(_health_payload(jobs_root, host, port))
                return
            if parsed.path == "/api/dashboard":
                self._send_json(
                    {
                        "service": UNIVERSAL_SERVICE_VERSION,
                        "reviews": discover_timeline_collision_reviews(
                            jobs_root, include_resolved=True
                        ),
                    }
                )
                return
            if parsed.path != "/":
                self.send_error(404)
                return
            page = render_universal_timeline_collision_panel(
                discover_timeline_collision_reviews(jobs_root, include_resolved=False)
            ).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(page)))
            self.end_headers()
            self.wfile.write(page)

        def do_POST(self) -> None:  # noqa: N802
            length = int(self.headers.get("Content-Length") or 0)
            data = parse_qs(self.rfile.read(length).decode("utf-8"))
            one = lambda key, default="": (data.get(key) or [default])[0]
            job_id = one("job_id")
            collision_id = one("collision_id")
            try:
                paths = timeline_collision_job_paths(jobs_root, job_id)
                if self.path == "/resolve":
                    save_timeline_collision_resolution(
                        resolutions_path=paths["resolutions"],
                        review_path=paths["review"],
                        panel_path=paths["panel"],
                        job_id=job_id,
                        collision_id=collision_id,
                        case_token=one("case_token"),
                        strategy=one("strategy"),
                        reviewed_by=one("reviewed_by"),
                        overlap_before_ms=int(one("overlap_before_ms", "0") or 0),
                        overlap_after_ms=int(one("overlap_after_ms", "0") or 0),
                        manual_text=one("manual_text"),
                        previous_variant_id=one("previous_variant_id"),
                        current_variant_id=one("current_variant_id"),
                        following_variant_id=one("following_variant_id"),
                    )
                elif self.path == "/clear":
                    clear_timeline_collision_resolution(
                        resolutions_path=paths["resolutions"],
                        review_path=paths["review"],
                        panel_path=paths["panel"],
                        collision_id=collision_id,
                    )
                else:
                    self.send_error(404)
                    return
            except Exception as exc:
                page = (
                    "<h1>Could not save decision</h1><pre>"
                    f"{html.escape(str(exc))}</pre><p><a href='/'>Back</a></p>"
                ).encode("utf-8")
                self.send_response(400)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(page)))
                self.end_headers()
                self.wfile.write(page)
                return
            self._redirect(f"/#{html.escape(job_id)}-{html.escape(collision_id)}")

        def log_message(self, format: str, *args: Any) -> None:
            return

    server = ThreadingHTTPServer((host, port), Handler)
    if announce:
        print(f"Universal timeline collision panel: http://{host}:{port}")
        print("This panel watches every Mathula TV job. Press Ctrl-C to stop it.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def ensure_universal_timeline_collision_panel(
    *,
    jobs_root: Path,
    host: str = DEFAULT_PANEL_HOST,
    port: int = DEFAULT_PANEL_PORT,
    startup_timeout_seconds: float = 5.0,
) -> str:
    """Reuse an existing universal service or start one in a daemon thread."""

    jobs_root = Path(jobs_root).resolve()
    healthy, payload = _server_health(jobs_root=jobs_root, host=host, port=port)
    if healthy:
        return f"http://{host}:{port}"
    if payload is not None:
        raise RuntimeError(
            f"Port {port} is occupied by a different timeline panel or jobs root"
        )

    key = (str(jobs_root), host, int(port))
    with _SERVER_LOCK:
        thread = _SERVER_THREADS.get(key)
        if thread is None or not thread.is_alive():
            _SERVER_ERRORS.pop(key, None)

            def worker() -> None:
                try:
                    serve_universal_timeline_collision_panel(
                        jobs_root=jobs_root,
                        host=host,
                        port=port,
                        announce=False,
                    )
                except BaseException as exc:  # surfaced synchronously below
                    _SERVER_ERRORS[key] = exc

            thread = threading.Thread(
                target=worker,
                name=f"mathula-timeline-panel-{port}",
                daemon=True,
            )
            _SERVER_THREADS[key] = thread
            thread.start()

    deadline = time.monotonic() + max(0.5, startup_timeout_seconds)
    while time.monotonic() < deadline:
        healthy, payload = _server_health(jobs_root=jobs_root, host=host, port=port)
        if healthy:
            return f"http://{host}:{port}"
        error = _SERVER_ERRORS.get(key)
        if error is not None:
            # Another process may have won the bind race. Check once more before
            # exposing its bind error.
            time.sleep(0.1)
            healthy, _ = _server_health(jobs_root=jobs_root, host=host, port=port)
            if healthy:
                return f"http://{host}:{port}"
            raise RuntimeError(
                f"Could not start universal timeline panel on {host}:{port}: {error}"
            ) from error
        time.sleep(0.1)
    raise RuntimeError(
        f"Universal timeline panel did not become ready on {host}:{port}"
    )


def _expected_case_tokens(review: Mapping[str, Any]) -> dict[str, str]:
    expected: dict[str, str] = {}
    for case in review.get("cases") or []:
        if not isinstance(case, Mapping):
            continue
        if case.get("status") not in {"open", "resolution_failed_review_again"}:
            continue
        collision_id = str(case.get("collision_id") or "").strip()
        token = str(case.get("case_token") or "").strip()
        if collision_id and token:
            expected[collision_id] = token
    return expected


def wait_for_timeline_collision_resolutions(
    *,
    review: Mapping[str, Any],
    jobs_root: Path,
    host: str = DEFAULT_PANEL_HOST,
    port: int = DEFAULT_PANEL_PORT,
    poll_interval_seconds: float = 1.0,
    timeout_seconds: float | None = None,
    progress: Callable[[str], None] | None = None,
) -> dict[str, dict[str, Any]]:
    """Block until every open case in ``review`` has a matching saved decision."""

    job_id = _safe_job_id(str(review.get("job_id") or ""))
    paths = timeline_collision_job_paths(jobs_root, job_id)
    expected = _expected_case_tokens(review)
    if not expected:
        return {}

    panel_url = ensure_universal_timeline_collision_panel(
        jobs_root=jobs_root,
        host=host,
        port=port,
    )
    message = (
        f"Timeline collision review required for job {job_id}. "
        f"Panel: {panel_url} — waiting for {len(expected)} decision(s)."
    )
    if progress is not None:
        progress(message)
    else:
        print(message)
        print("Save the decision in the panel; dub-azure will continue automatically.")

    started = time.monotonic()
    while True:
        values = resolution_map(paths["resolutions"])
        matched = {
            collision_id: values[collision_id]
            for collision_id, token in expected.items()
            if collision_id in values
            and str(values[collision_id].get("case_token") or "") == token
        }
        if len(matched) == len(expected):
            done = (
                f"Timeline collision decision received for job {job_id}; "
                "resuming cached dub automatically."
            )
            if progress is not None:
                progress(done)
            else:
                print(done)
            return matched
        if timeout_seconds is not None and time.monotonic() - started >= timeout_seconds:
            missing = sorted(set(expected) - set(matched))
            raise TimeoutError(
                "Timed out waiting for timeline collision decision(s): "
                + ", ".join(missing)
            )
        time.sleep(max(0.1, poll_interval_seconds))


# Backward-compatible per-job server.  New code should use the universal service.
def serve_timeline_collision_panel(
    *,
    job_id: str,
    review_path: Path,
    panel_path: Path,
    resolutions_path: Path,
    host: str = DEFAULT_PANEL_HOST,
    port: int = DEFAULT_PANEL_PORT,
) -> None:
    jobs_root = Path(review_path).resolve().parents[2]
    serve_universal_timeline_collision_panel(
        jobs_root=jobs_root,
        host=host,
        port=port,
        announce=True,
    )


__all__ = [
    "ALLOWED_STRATEGIES",
    "DEFAULT_PANEL_HOST",
    "DEFAULT_PANEL_PORT",
    "PANEL_VERSION",
    "REVIEW_SCHEMA_VERSION",
    "RESOLUTION_SCHEMA_VERSION",
    "UNIVERSAL_SERVICE_VERSION",
    "clear_timeline_collision_resolution",
    "discover_timeline_collision_reviews",
    "ensure_universal_timeline_collision_panel",
    "load_timeline_collision_resolutions",
    "load_timeline_collision_review",
    "render_timeline_collision_panel",
    "render_universal_timeline_collision_panel",
    "resolution_map",
    "save_timeline_collision_resolution",
    "serve_timeline_collision_panel",
    "serve_universal_timeline_collision_panel",
    "timeline_collision_job_paths",
    "timeline_collision_summary",
    "universal_timeline_collision_summary",
    "upsert_timeline_collision_case",
    "wait_for_timeline_collision_resolutions",
]
