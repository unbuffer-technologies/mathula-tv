"""Find the seams of a finished dub that the ear would catch, without the ear.

    python -m mathula_tv.cut_audit --job <job id> --run <run id> [--track dub.wav]

The dub is made of pieces cut and laid next to each other: spans cut from their lead and follow words, names mixed in, units placed at their start.  A cut made from the aligner's word edges can be
wrong by half a second (they are not exact), and the result is heard, not seen: a sliver of the next word in a pause (prod-y1 0:56, 40 ms at speech level, in 0.9 s of silence), a jump in loudness
between two sentences, a pause that stalls.  This reads the dub itself, the same audio the listener gets, and lists every place where the signal says so, with its time in the video, so that the places to
listen to are known before the listener has to find them.  It looks at the signal only: no model, no recogniser.  It does not decide: a flag says "listen here", and a short word with a real pause either side
is a flag too.

What it looks for, per 10 ms frame, against the local speech level (the 95th percentile of a few seconds around it; quiet is 25 dB under it):

- ``burst_in_pause``: sound of at most 300 ms, within 15 dB of the speech level, with at least 120 ms of quiet on both sides: a leaked sliver of a neighbouring word, a click, a breath cut out of its word.
- ``level_step``: the speech after a pause is 7 dB louder or quieter than the speech before it (the two pieces were not made at the same level).
- ``long_pause``: more than 2 s of quiet between speech of the same unit.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

FRAME_S = 0.01
QUIET_DB = 25.0              # quiet is this far under the local speech level
BURST_MAX_S = 0.30           # a sound this short ...
BURST_QUIET_S = 0.12         # ... with this much quiet either side is not a word of a sentence
BURST_AUDIBLE_DB = 15.0      # and it reaches within this of the speech level (the burst at prod-y1 0:56 was 3.5 dB under it; a breath or the room's own ripple is 20-25 dB under)
STEP_DB = 7.0                # the speech after a pause differs from the speech before it by this much
STEP_SPAN_S = 1.0            # speech compared over this long either side of a pause
STEP_PAUSE_S = 0.10
LONG_PAUSE_S = 2.0
LEVEL_BLOCK = 100            # the local level is taken per second, over a few seconds around it
LEVEL_REACH = 300


def clock(t: float) -> str:
    return f"{int(t // 60)}:{t % 60:04.1f}"


def _frames_db(audio: np.ndarray, rate: int) -> np.ndarray:
    frame = max(1, int(FRAME_S * rate))
    count = len(audio) // frame
    if count == 0:
        return np.zeros(0)
    return 20.0 * np.log10(np.sqrt((audio[: count * frame].reshape(count, frame) ** 2).mean(axis=1)) + 1e-9)


def _local_level(db: np.ndarray) -> np.ndarray:
    """The speech level around every frame: the 95th percentile of the frames within a few seconds, per second."""
    level = np.empty(len(db))
    for start in range(0, len(db), LEVEL_BLOCK):
        window = db[max(0, start - LEVEL_REACH): start + LEVEL_BLOCK + LEVEL_REACH]
        level[start: start + LEVEL_BLOCK] = np.percentile(window, 95)
    return level


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """(first frame, one past the last) of every run of True."""
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return [(int(a), int(b)) for a, b in zip(np.nonzero(edges == 1)[0], np.nonzero(edges == -1)[0])]


def find_seam_flags(audio: np.ndarray, rate: int, unit_starts_s: Sequence[float] = ()) -> list[dict[str, Any]]:
    """The places of ``audio`` (mono, one dub) worth listening to: [{t, time, unit, kind, ...}], in time order.  ``unit_starts_s`` names the unit each flag falls in."""
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    db = _frames_db(audio, rate)
    if len(db) < 50:
        return []
    level = _local_level(db)
    quiet = db < level - QUIET_DB
    flags: list[dict[str, Any]] = []

    def unit_of(t: float) -> int | None:
        found = [i for i, s in enumerate(unit_starts_s) if s <= t + 1e-6]
        return found[-1] if found else None

    def flag(frame: int, kind: str, **detail: Any) -> None:
        t = frame * FRAME_S
        flags.append({"t": round(t, 2), "time": clock(t), "unit": unit_of(t), "kind": kind, **detail})

    sound = _runs(~quiet)
    for k, (a, b) in enumerate(sound):                                    # a short sound with real quiet on both sides
        before = a - sound[k - 1][1] if k > 0 else 0
        after = sound[k + 1][0] - b if k + 1 < len(sound) else 0
        under = float(level[a: b].mean() - db[a: b].max())
        if (b - a) * FRAME_S <= BURST_MAX_S and before * FRAME_S >= BURST_QUIET_S and after * FRAME_S >= BURST_QUIET_S and under <= BURST_AUDIBLE_DB:
            flag(a, "burst_in_pause", length_ms=int(round((b - a) * FRAME_S * 1000)), quiet_before_ms=int(round(before * FRAME_S * 1000)), quiet_after_ms=int(round(after * FRAME_S * 1000)),
                 peak_db_under_speech=round(under, 1))
    span = int(STEP_SPAN_S / FRAME_S)
    for a, b in _runs(quiet):                                             # speech either side of a pause at a different level
        if (b - a) * FRAME_S < STEP_PAUSE_S or a < 10 or b > len(db) - 10:
            continue
        left = db[max(0, a - span): a][~quiet[max(0, a - span): a]]
        right = db[b: b + span][~quiet[b: b + span]]
        if len(left) >= 20 and len(right) >= 20:
            step = float(np.mean(right) - np.mean(left))
            if abs(step) >= STEP_DB:
                flag(b, "level_step", step_db=round(step, 1), pause_ms=int(round((b - a) * FRAME_S * 1000)))
    for a, b in _runs(quiet):                                             # a stall inside a unit
        if (b - a) * FRAME_S < LONG_PAUSE_S or a == 0 or b >= len(db):
            continue
        u0, u1 = unit_of(a * FRAME_S), unit_of(b * FRAME_S - 0.01)
        if u0 is not None and u0 == u1:
            flag(a, "long_pause", length_s=round((b - a) * FRAME_S, 2))
    return sorted(flags, key=lambda f: f["t"])


def audit_track(track: Path | str, items: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """The seam flags of a dub audio file, the units named from the submission's items."""
    import soundfile as sf

    audio, rate = sf.read(str(track), dtype="float32", always_2d=True)
    return find_seam_flags(audio.mean(axis=1), int(rate), [int(i.get("start_ms") or 0) / 1000.0 for i in items])


def report(flags: Sequence[dict[str, Any]]) -> list[str]:
    lines = []
    for f in flags:
        extra = {k: v for k, v in f.items() if k not in ("t", "time", "unit", "kind")}
        lines.append(f"{f['time']:>7}  unit {f['unit']}  {f['kind']:<15} {json.dumps(extra)}")
    return lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="mathula_tv.cut_audit")
    parser.add_argument("--job", required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--track", default=None, help="the dub audio to read (default: the run's Studio audio)")
    parser.add_argument("--clip", default=None)
    args = parser.parse_args(argv)

    from .config import load_settings
    from .console import Console
    from .experience_store import ExperienceStore

    settings = load_settings()
    clip = args.clip or args.job[:8]
    items = json.loads((ExperienceStore(settings.work_dir).repeat_dir(args.run, clip, 1) / "submission.json").read_text(encoding="utf-8"))["items"]
    if args.track:
        track = Path(args.track)
    else:
        console = Console(Path(settings.work_dir))
        answer = console.studio_run(args.job, args.run, make=False)
        if not answer.get("ready"):
            raise SystemExit("the Studio audio of this run is not made yet: finish it first (dub_finish)")
        track = Path(console.audio_dir) / Path(str(answer["url"])).name
    flags = audit_track(track, items)
    print("\n".join(report(flags)) or "no seam flags")
    print(f"{len(flags)} flag(s) in {track.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
