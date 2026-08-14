"""Human review panel for direct-dub timeline collisions.

The panel is intentionally independent of the translation artifact.  It stores
small, auditable direct-dub overrides that are applied only during Azure timing
and rendering.  Source text and the authoritative translation stay unchanged.
"""

from __future__ import annotations

import html
import json
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import parse_qs

from .atomic_io import atomic_write_json, read_json

REVIEW_SCHEMA_VERSION = "mathula.timeline-collision-review.v1"
RESOLUTION_SCHEMA_VERSION = "mathula.timeline-collision-resolutions.v1"
PANEL_VERSION = "mathula.timeline-collision-panel.v1"
ALLOWED_STRATEGIES = {"allow_overlap", "manual_text", "select_variants"}


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


def upsert_timeline_collision_case(
    *,
    review_path: Path,
    panel_path: Path,
    resolutions_path: Path,
    job_id: str,
    case: Mapping[str, Any],
) -> dict[str, Any]:
    collision_id = str(case.get("collision_id") or "").strip()
    if not collision_id:
        raise ValueError("Timeline collision case requires collision_id")

    payload = load_timeline_collision_review(review_path)
    cases = [dict(item) for item in payload.get("cases") or [] if isinstance(item, Mapping)]
    replacement = {
        **dict(case),
        "collision_id": collision_id,
        "status": "open",
        "updated_at": _utcnow(),
    }
    for index, existing in enumerate(cases):
        if str(existing.get("collision_id") or "") == collision_id:
            replacement.setdefault("created_at", existing.get("created_at") or _utcnow())
            cases[index] = replacement
            break
    else:
        replacement.setdefault("created_at", _utcnow())
        cases.append(replacement)

    resolutions = resolution_map(resolutions_path)
    for item in cases:
        item_id = str(item.get("collision_id") or "")
        if item_id in resolutions:
            item["saved_resolution"] = resolutions[item_id]
            item["status"] = "resolution_failed_review_again"

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
        "resume_command": f'python -m mathula_tv.cli dub-azure "{job_id}" --live-operation',
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
    overlap_before_ms: int | None = None,
    overlap_after_ms: int | None = None,
    manual_text: str | None = None,
    previous_variant_id: str | None = None,
    current_variant_id: str | None = None,
    following_variant_id: str | None = None,
) -> dict[str, Any]:
    collision_id = collision_id.strip()
    strategy = strategy.strip()
    reviewed_by = reviewed_by.strip()
    if not collision_id:
        raise ValueError("collision_id is required")
    if strategy not in ALLOWED_STRATEGIES:
        raise ValueError(f"Unsupported timeline collision strategy: {strategy}")
    if not reviewed_by:
        raise ValueError("reviewed_by is required")

    resolution: dict[str, Any] = {
        "collision_id": collision_id,
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
        invalid = [value for value in variants.values() if value and value not in {"natural", "concise", "compact"}]
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

    review = load_timeline_collision_review(review_path)
    for case in review.get("cases") or []:
        if isinstance(case, dict) and str(case.get("collision_id") or "") == collision_id:
            case["saved_resolution"] = resolution
            case["status"] = "resolution_saved"
    if review.get("cases"):
        review["updated_at"] = _utcnow()
        review["open_case_count"] = sum(
            1 for item in review["cases"] if item.get("status") == "open"
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
    atomic_write_json(review_path, review)
    panel_path.write_text(render_timeline_collision_panel(review), encoding="utf-8")


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


def render_timeline_collision_panel(review: Mapping[str, Any]) -> str:
    job_id = str(review.get("job_id") or "")
    cases_html: list[str] = []
    for case in review.get("cases") or []:
        if not isinstance(case, Mapping):
            continue
        collision_id = str(case.get("collision_id") or "")
        triplet = {
            str(item.get("role") or ""): item.get("block") or {}
            for item in case.get("triplet") or []
            if isinstance(item, Mapping)
        }
        shortfall = int(case.get("shortfall_ms") or 0)
        suggested_before = max(0, shortfall // 2)
        suggested_after = max(0, shortfall - suggested_before)
        current_text = str((triplet.get("current") or {}).get("translated_text") or "")
        saved = case.get("saved_resolution") if isinstance(case.get("saved_resolution"), Mapping) else None
        saved_html = (
            f'<div class="saved">Saved resolution: <code>{html.escape(json.dumps(saved, ensure_ascii=False))}</code></div>'
            if saved
            else ""
        )
        cards = "".join(
            _block_card(role, triplet.get(role) or {})
            for role in ("previous", "current", "following")
        )
        cases_html.append(
            f"""
            <article class="collision">
              <header><h2>{html.escape(collision_id)}</h2><span class="status">{html.escape(str(case.get('status') or 'open'))}</span></header>
              <p><strong>Reason:</strong> {html.escape(str(case.get('reason') or 'timeline_collision'))}</p>
              <p><strong>Measured:</strong> {html.escape(str(case.get('measured_duration_ms') or ''))} ms; <strong>available:</strong> {html.escape(str(case.get('available_duration_ms') or ''))} ms; <strong>shortfall:</strong> {shortfall} ms.</p>
              {saved_html}
              <div class="triplet">{cards}</div>
              <div class="options">
                <form method="post" action="/resolve">
                  <input type="hidden" name="collision_id" value="{html.escape(collision_id)}">
                  <input type="hidden" name="strategy" value="allow_overlap">
                  <h3>Option 1 — allow commission interjection overlap</h3>
                  <label>Overlap before (ms)<input name="overlap_before_ms" type="number" min="0" max="3000" value="{suggested_before}"></label>
                  <label>Overlap after (ms)<input name="overlap_after_ms" type="number" min="0" max="3000" value="{suggested_after}"></label>
                  <label>Reviewed by<input name="reviewed_by" required></label>
                  <button>Save overlap decision</button>
                </form>
                <form method="post" action="/resolve">
                  <input type="hidden" name="collision_id" value="{html.escape(collision_id)}">
                  <input type="hidden" name="strategy" value="manual_text">
                  <h3>Option 2 — shorter direct-dub text override</h3>
                  <textarea name="manual_text" rows="5" required>{html.escape(current_text)}</textarea>
                  <label>Reviewed by<input name="reviewed_by" required></label>
                  <button>Save manual text</button>
                </form>
                <form method="post" action="/resolve">
                  <input type="hidden" name="collision_id" value="{html.escape(collision_id)}">
                  <input type="hidden" name="strategy" value="select_variants">
                  <h3>Option 3 — force prepared variants across the triplet</h3>
                  {''.join(f'<label>{role.title()}<select name="{role}_variant_id"><option value="">Keep automatic</option><option>natural</option><option>concise</option><option>compact</option></select></label>' for role in ('previous','current','following'))}
                  <label>Reviewed by<input name="reviewed_by" required></label>
                  <button>Save variant choice</button>
                </form>
                <form method="post" action="/clear"><input type="hidden" name="collision_id" value="{html.escape(collision_id)}"><button class="secondary">Clear saved decision</button></form>
              </div>
            </article>
            """
        )

    body = "".join(cases_html) or "<p>No timeline collisions are awaiting review.</p>"
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Mathula TV timeline collision review</title>
<style>
body{{font-family:system-ui,sans-serif;margin:0;background:#f4f5f7;color:#171717}}main{{max-width:1500px;margin:auto;padding:24px}}header{{display:flex;justify-content:space-between;align-items:center;gap:12px}}.collision{{background:white;border:1px solid #ddd;border-radius:12px;padding:20px;margin:20px 0;box-shadow:0 2px 10px #0001}}.status,.saved{{background:#fff3cd;border:1px solid #ffe69c;border-radius:6px;padding:8px}}.triplet{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}}.block-card{{border:1px solid #ddd;border-radius:8px;padding:12px;overflow:auto}}table{{border-collapse:collapse;width:100%;font-size:13px}}th,td{{border:1px solid #ddd;padding:6px;text-align:left;vertical-align:top}}.options{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px;margin-top:16px}}form{{border:1px solid #ddd;border-radius:8px;padding:12px}}label{{display:block;margin:8px 0}}input,select,textarea{{box-sizing:border-box;width:100%;padding:8px;margin-top:4px}}button{{padding:10px 14px;border:0;border-radius:6px;background:#111;color:white;cursor:pointer}}button.secondary{{background:#666}}code{{white-space:pre-wrap;word-break:break-word}}@media(max-width:1000px){{.triplet,.options{{grid-template-columns:1fr}}}}
</style></head><body><main>
<h1>Mathula TV timeline collision review</h1>
<p>Job <code>{html.escape(job_id)}</code>. Decisions are direct-dub overlays; the authoritative translation is not rewritten.</p>
{body}
<p>After saving decisions, rerun:</p><pre>python -m mathula_tv.cli dub-azure "{html.escape(job_id)}" --live-operation</pre>
</main></body></html>"""


def timeline_collision_summary(review: Mapping[str, Any]) -> str:
    lines = [
        f"Timeline collision review for job {review.get('job_id', '')}",
        f"Panel: {review.get('panel_path', '')}",
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


def serve_timeline_collision_panel(
    *,
    job_id: str,
    review_path: Path,
    panel_path: Path,
    resolutions_path: Path,
    host: str = "127.0.0.1",
    port: int = 8765,
) -> None:
    if host not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("Timeline collision panel may only bind to localhost")

    class Handler(BaseHTTPRequestHandler):
        def _redirect(self) -> None:
            self.send_response(303)
            self.send_header("Location", "/")
            self.end_headers()

        def do_GET(self) -> None:  # noqa: N802
            if self.path != "/":
                self.send_error(404)
                return
            review = load_timeline_collision_review(review_path)
            page = render_timeline_collision_panel(review).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(page)))
            self.end_headers()
            self.wfile.write(page)

        def do_POST(self) -> None:  # noqa: N802
            length = int(self.headers.get("Content-Length") or 0)
            data = parse_qs(self.rfile.read(length).decode("utf-8"))
            one = lambda key, default="": (data.get(key) or [default])[0]
            try:
                if self.path == "/resolve":
                    save_timeline_collision_resolution(
                        resolutions_path=resolutions_path,
                        review_path=review_path,
                        panel_path=panel_path,
                        job_id=job_id,
                        collision_id=one("collision_id"),
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
                        resolutions_path=resolutions_path,
                        review_path=review_path,
                        panel_path=panel_path,
                        collision_id=one("collision_id"),
                    )
                else:
                    self.send_error(404)
                    return
            except Exception as exc:
                page = f"<h1>Could not save decision</h1><pre>{html.escape(str(exc))}</pre><p><a href='/'>Back</a></p>".encode("utf-8")
                self.send_response(400)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(page)))
                self.end_headers()
                self.wfile.write(page)
                return
            self._redirect()

        def log_message(self, format: str, *args: Any) -> None:
            return

    server = ThreadingHTTPServer((host, port), Handler)
    print(f"Timeline collision panel: http://{host}:{port}")
    print("Press Ctrl-C to stop the panel after saving your decision.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


__all__ = [
    "ALLOWED_STRATEGIES",
    "PANEL_VERSION",
    "REVIEW_SCHEMA_VERSION",
    "RESOLUTION_SCHEMA_VERSION",
    "clear_timeline_collision_resolution",
    "load_timeline_collision_resolutions",
    "load_timeline_collision_review",
    "render_timeline_collision_panel",
    "resolution_map",
    "save_timeline_collision_resolution",
    "serve_timeline_collision_panel",
    "timeline_collision_summary",
    "upsert_timeline_collision_case",
]
