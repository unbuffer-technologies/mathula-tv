"""Several copies of the voice service for the length of a job, and none held after it.

Cloud Run does not start a second copy of a GPU service just because requests wait (measured 2026-11-24: four held requests were answered one after another by one copy for four minutes), so a job that
wants its clusters spoken side by side asks for them: ``acquire(n)`` sets the service's minimum number of instances to ``n`` (the Admin API's service-level scaling, no new revision) and ``release()``
sets it back to zero.  Copies cost money for as long as they are held, so every holder is a LEASE in ``<work dir>/voice_scale.json`` (who, how many, since when): the minimum is the largest of the live
leases, a lease is dropped when its process is gone or it is older than ``MAX_AGE_S``, and ``release_if_stale()`` (called at the start of every job) puts the service back to zero when nothing live holds it.

The account that runs this needs ``roles/run.developer`` on the service and ``roles/iam.serviceAccountUser`` on the service's own account (granted 2026-11-24).
"""
from __future__ import annotations

import json
import os
import socket
import time
from pathlib import Path
from typing import Any, Callable, Mapping

MAX_COPIES = 4
MAX_AGE_S = 3 * 3600          # a lease older than this is from a run that died: it is not honoured
LEASE_FILE = "voice_scale.json"
DEFAULTS = {"project": "mathula-ai", "region": "europe-west1", "service": "omnivoice"}


def _target(environ: Mapping[str, str] | None = None) -> str:
    env = os.environ if environ is None else environ
    project = env.get("OMNIVOICE_GCP_PROJECT") or DEFAULTS["project"]
    region = env.get("OMNIVOICE_GCP_REGION") or DEFAULTS["region"]
    service = env.get("OMNIVOICE_GCP_SERVICE") or DEFAULTS["service"]
    return f"https://run.googleapis.com/v2/projects/{project}/locations/{region}/services/{service}"


def _access_token() -> str:
    import google.auth
    import google.auth.transport.requests

    credentials, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
    credentials.refresh(google.auth.transport.requests.Request())
    return str(credentials.token)


def _call(method: str, url: str, body: dict[str, Any] | None = None, *, timeout: float = 60.0) -> dict[str, Any]:
    import requests

    answer = requests.request(method, url, headers={"Authorization": f"Bearer {_access_token()}", "Content-Type": "application/json"}, json=body, timeout=timeout)
    if answer.status_code >= 400:
        raise RuntimeError(f"Cloud Run {method} {answer.status_code}: {answer.text[:300]}")
    return answer.json() if answer.content else {}


def get_min(environ: Mapping[str, str] | None = None, *, call: Callable[..., dict[str, Any]] | None = None) -> int:
    """The service's minimum number of instances now."""
    answer = (call or _call)("GET", _target(environ))
    return int((answer.get("scaling") or {}).get("minInstanceCount") or 0)


def set_min(count: int, environ: Mapping[str, str] | None = None, *, call: Callable[..., dict[str, Any]] | None = None) -> int:
    count = max(0, min(MAX_COPIES, int(count)))
    (call or _call)("PATCH", _target(environ) + "?updateMask=scaling.minInstanceCount", {"scaling": {"minInstanceCount": count}})
    return count


def _alive(pid: int, host: str) -> bool:
    if host != socket.gethostname():
        return True                                    # another machine: only its age can tell
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError as error:
        return getattr(error, "winerror", None) != 87          # Windows says "invalid parameter" for a process that does not exist


def _read(path: Path) -> dict[str, dict[str, Any]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return {str(k): v for k, v in (data.get("leases") or {}).items() if isinstance(v, dict)}
    except (OSError, ValueError):
        return {}


def _write(path: Path, leases: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps({"leases": dict(leases)}, indent=1), encoding="utf-8")
    os.replace(temporary, path)


def live_leases(work_dir: Path | str, *, now: float | None = None) -> dict[str, dict[str, Any]]:
    now = time.time() if now is None else now
    return {k: v for k, v in _read(Path(work_dir) / LEASE_FILE).items()
            if now - float(v.get("at") or 0.0) < MAX_AGE_S and _alive(int(v.get("pid") or 0), str(v.get("host") or ""))}


def acquire(count: int, work_dir: Path | str, *, environ: Mapping[str, str] | None = None, call: Callable[..., dict[str, Any]] | None = None, progress: Callable[[str], None] = print) -> int:
    """Hold ``count`` copies for this process: the minimum becomes the largest of the live leases.  Returns what the minimum is now."""
    path = Path(work_dir) / LEASE_FILE
    leases = live_leases(work_dir)
    leases[str(os.getpid())] = {"pid": os.getpid(), "host": socket.gethostname(), "min": max(1, min(MAX_COPIES, int(count))), "at": time.time()}
    _write(path, leases)
    wanted = max(int(v.get("min") or 0) for v in leases.values())
    set_min(wanted, environ, call=call)
    progress(f"[voice] holding {wanted} cop{'y' if wanted == 1 else 'ies'} of the voice for this job")
    return wanted


def release(work_dir: Path | str, *, environ: Mapping[str, str] | None = None, call: Callable[..., dict[str, Any]] | None = None, progress: Callable[[str], None] = print) -> int:
    """Give up this process's copies: the minimum falls to what the other live leases hold, or zero.  Returns the minimum now."""
    path = Path(work_dir) / LEASE_FILE
    leases = live_leases(work_dir)
    leases.pop(str(os.getpid()), None)
    _write(path, leases)
    wanted = max([int(v.get("min") or 0) for v in leases.values()] or [0])
    set_min(wanted, environ, call=call)
    progress(f"[voice] copies held now: {wanted}")
    return wanted


def release_if_stale(work_dir: Path | str, *, environ: Mapping[str, str] | None = None, call: Callable[..., dict[str, Any]] | None = None, progress: Callable[[str], None] = print) -> bool:
    """At the start of a job: if the service is held above zero but no live lease holds it (a run died with its copies), put it back.  True when it was."""
    held = get_min(environ, call=call)
    if held <= 0:
        return False
    leases = live_leases(work_dir)
    if leases and max(int(v.get("min") or 0) for v in leases.values()) >= held:
        return False
    wanted = max([int(v.get("min") or 0) for v in leases.values()] or [0])
    _write(Path(work_dir) / LEASE_FILE, leases)
    set_min(wanted, environ, call=call)
    progress(f"[voice] the service was held at {held} with no run holding it: now {wanted}")
    return True
