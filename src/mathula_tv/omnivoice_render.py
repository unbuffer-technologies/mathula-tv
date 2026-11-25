"""A submission's audio through the OmniVoice service: the dub as it would sound, unit by unit, on the source timeline.

Each unit is spoken with the same plan the agent's measurements used (``OmniVoiceTTSBackend.plan``: the pronunciation dictionary, held pauses as silence, spans), cloned from the
reference clip cut from its own window of the source audio, and laid at its source start.  Nothing is pushed later: where a unit runs past the next one's start the report says
so, because that is the collision the scorer counts.

``exact=True`` additionally pins each unit's length to its window (OmniVoice's ``duration``), shared between its spans by syllables, with one bounded correction when the result
misses by more than ``tol_s``: a way to fit a draft to its windows without rewording it, at the cost of a pace the model did not choose.  The default is the model's own pace.

Run:  python -m mathula_tv.omnivoice_render --run prod-114 --clip ada0e557 --out /tmp/render
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import soundfile as sf

from . import omnivoice_duration as od
from .omnivoice_tts import OmniVoiceTTSBackend

RATE = 24000


def _syllables(text: str) -> int:
    return max(1, len(re.findall(r"[aeiouAEIOU]", text)))                 # Zulu: one syllable per vowel


def _decode(answer: Mapping[str, Any]) -> np.ndarray:
    audio, rate = sf.read(io.BytesIO(base64.b64decode(answer["audio_b64"])), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if rate != RATE:
        raise ValueError(f"the service answered at {rate} Hz, expected {RATE}")
    return audio


def _pinned(spans: Sequence[str], gaps_ms: Sequence[int], window_s: float) -> list[float]:
    """Each span's length when the whole unit must fill its window: what is left after the silences and the model's 200 ms of padding per span, shared by syllables."""
    speech = max(0.5, window_s - sum(gaps_ms) / 1000.0 - od.EDGE_PAD_S * len(spans))
    total = sum(_syllables(s) for s in spans)
    return [speech * _syllables(s) / total for s in spans]


def render_submission(items: Sequence[Mapping[str, Any]], corpus_root: Path | str, backend: OmniVoiceTTSBackend, *, voices: Mapping[str, str] | None = None,
                      total_s: float | None = None, exact: bool = False, tol_s: float = 0.25) -> dict[str, Any]:
    """The audio of every item placed at its start, and a row per unit: spoken length against its window, and how far it runs into the next unit."""
    ordered = sorted(items, key=lambda it: int(it["start_ms"]))
    rows: list[dict[str, Any]] = []
    pieces: list[tuple[int, np.ndarray]] = []
    for position, item in enumerate(ordered):
        start, end = int(item["start_ms"]), int(item["end_ms"])
        speaker = str(item.get("speaker_id") or "SPEAKER_00")
        group = {"group_id": str(item["item_id"]), "speaker_id": speaker, "start_ms": start, "source_end_ms": end, "source_span_ms": end - start,
                 "parts": [{"type": "text", "segment_id": f"{item['item_id']}_seg", "text": str(item["zulu_text"])}]}
        voice_info = {"selected_voice": (voices or {}).get(speaker, backend.default_voice)}
        spans, gaps = backend.plan(group, voice_info, corpus_root)
        if not spans:
            rows.append({"unit": item["item_id"], "speaker": speaker, "error": "no text"})
            continue
        reference = backend.reference_for(group, corpus_root)
        window_s = (end - start) / 1000.0
        durations = _pinned(spans, gaps, window_s) if exact else None
        natural: dict[str, Any] = {}
        if exact:
            answer = backend.client.synthesize(spans=spans, gaps_ms=gaps, reference=reference, durations_s=durations, speed=backend.speed, lora=backend.lora)
            audio, attempts = _decode(answer), 1
        else:                                            # the natural voice, exactly as the agent measured it (OmniVoiceTTSBackend.speak): nothing pinned, stretched or cut
            audio, rate, _, natural, reference = backend.speak(group, spans, gaps, corpus_root)
            if rate != RATE:
                raise ValueError(f"the service answered at {rate} Hz, expected {RATE}")
            attempts = 1
        if exact and abs(len(audio) / RATE - window_s) > tol_s:                      # one bounded correction, shifted by the miss
            again = _decode(backend.client.synthesize(spans=spans, gaps_ms=gaps, reference=reference, durations_s=_pinned(spans, gaps, window_s - (len(audio) / RATE - window_s)),
                                                       speed=backend.speed, lora=backend.lora))
            attempts = 2
            if abs(len(again) / RATE - window_s) < abs(len(audio) / RATE - window_s):
                audio = again
        following = ordered[position + 1] if position + 1 < len(ordered) else None
        spoken = len(audio) / RATE
        into_next = (start / 1000.0 + spoken - int(following["start_ms"]) / 1000.0) if following else None
        rows.append({"unit": item["item_id"], "speaker": speaker, "source_start_s": round(start / 1000.0, 2), "window_s": round(window_s, 2), "spoken_s": round(spoken, 2),
                     "past_window_end_s": round(spoken - window_s, 2), "runs_into_next_s": None if into_next is None else round(into_next, 2),
                     "next_is_other_speaker": bool(following and str(following.get("speaker_id") or "") != speaker), "spans": len(spans), "attempts": attempts,
                     "reference_s": [round(reference.start_s, 1), round(reference.end_s, 1)],
                     "takes": natural.get("takes"), "cuts": natural.get("overlap"), "joins": natural.get("joins"), "silences": natural.get("silences"), "primers": natural.get("primers")})                  # how each span after the first was cut from its lead: by the service's word times ("words") or by warping ("warp")
        pieces.append((int(start / 1000.0 * RATE), audio))
    length = max([p[0] + len(p[1]) for p in pieces] + [int((total_s or 0) * RATE)] or [0])
    track = np.zeros(length, dtype=np.float32)
    for at, audio in pieces:
        track[at:at + len(audio)] += audio
    return {"track": track, "rows": rows}


def summarise(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    done = [r for r in rows if "error" not in r]
    collisions = [r for r in done if (r.get("runs_into_next_s") or 0) > 0.02]
    return {"units": len(done), "collisions": len(collisions), "cross_speaker_collisions": sum(1 for r in collisions if r["next_is_other_speaker"]),
            "worst_collision_s": max([r["runs_into_next_s"] for r in collisions] or [0]), "over_window": [(r["unit"], r["past_window_end_s"]) for r in done if r["past_window_end_s"] > 0.25],
            "short_of_window_by_over_1s": [(r["unit"], round(-r["past_window_end_s"], 1)) for r in done if r["past_window_end_s"] < -1.0]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="mathula_tv.omnivoice_render")
    parser.add_argument("--run", required=True)
    parser.add_argument("--clip", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--exact", action="store_true", help="pin each unit's length to its window (one bounded correction)")
    parser.add_argument("--total-s", type=float, default=None, help="length of the source in seconds (default: to the end of the last unit)")
    args = parser.parse_args(argv)

    from .config import load_settings
    from .experience_store import ExperienceStore

    store = ExperienceStore(load_settings().work_dir)
    from .dub_pronunciation import register_clip_pronunciation

    register_clip_pronunciation(store, args.clip)             # the job's pronunciation research, as the agent's measurements and the pipeline have it: the same text is planned everywhere
    run_dir = store.repeat_dir(args.run, args.clip, 1)
    items = json.loads((run_dir / "submission.json").read_text(encoding="utf-8"))["items"]
    corpus = Path(store.root) / "corpus" / args.clip
    voices: dict[str, str] = {}
    try:
        resolution = json.loads((corpus / "voice_resolution.json").read_text(encoding="utf-8"))
        assigned = resolution.get("assignments") or {}
        voices = {str(k): str(v.get("selected_voice")) for k, v in assigned.items() if isinstance(v, dict) and v.get("selected_voice")}
    except (OSError, ValueError):
        pass
    backend = OmniVoiceTTSBackend.from_environment()
    result = render_submission(items, corpus, backend, voices=voices, total_s=args.total_s, exact=args.exact)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    track = result["track"]
    sf.write(str(out / "track.flac"), (track / (float(np.abs(track).max()) or 1.0) * 0.9).astype(np.float32), RATE, subtype="PCM_16")
    report = {"run": args.run, "clip": args.clip, "mode": "exact-fit" if args.exact else "natural pace", "summary": summarise(result["rows"]), "units": result["rows"]}
    (out / "report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
