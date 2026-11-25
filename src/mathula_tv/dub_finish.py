"""Finish a dub: the video to post, and everything to post it with, from the run's Studio audio.

    python -m mathula_tv.dub_finish --job <job id> --run <run id> [--opening-cut] [--new-hook]

The listener hears the dub in the Studio (every unit made once, with the draws that were picked); this makes the video from that same audio:

1. the Studio's whole dub (built now if it is not, with the run's own voice settings: ``<run>/<clip>/voice.json``),
2. the dubbed master: the original picture, never retimed or re-encoded, with the dub as its sound,
3. the publication step: the footer hook card over the picture and the editorial package (hook, caption, hashtags, search keywords) from one Gemini call; the whole video is kept unless
   ``opening_cut`` asks for the opening to be cut (a clip of a minute or two has nothing to cut; the editorial call is saved, so a second finish keeps the same hook unless ``new_hook``),
4. the hand-over: ``<job>/output/final_dubbed_<job>.mp4``, ``dubbed_master_<job>.mp4`` and the paste-ready ``tiktok_caption_zu_<job>.txt`` (caption, a blank line, the hashtags), hashtags, cover,
   search keywords and SEO files (dub_publish).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path
from typing import Any, Callable


def _duration(video: Path) -> float:
    out = subprocess.check_output(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(video)])
    return float(out.decode().strip())


def mux_master(source_video: Path, dub_track: Path, output: Path) -> Path:
    """The source picture, copied as it is, with the dub as its sound (stereo AAC at 48 kHz), padded with silence to the picture's length."""
    length = _duration(Path(source_video))
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(source_video), "-i", str(dub_track), "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
                    "-af", f"apad=whole_dur={length},aformat=channel_layouts=stereo", "-t", str(length), "-movflags", "+faststart", str(output)], check=True)
    return Path(output)


def studio_track(console: Any, job_id: str, run_id: str, *, progress: Callable[[str], None] = print, poll_seconds: float = 5.0) -> Path:
    """The whole dub of the run as one audio file (the Studio's), waiting for the units to be made."""
    started = time.time()
    while True:
        answer = console.studio_run(job_id, run_id)
        if answer.get("ready"):
            progress(f"[finish] studio audio ready: {answer['seconds']} s, in {time.time() - started:.0f} s")
            return Path(console.audio_dir) / Path(str(answer["url"])).name
        progress(f"[finish] studio: {answer.get('made')} of {answer.get('total')} units made")
        time.sleep(poll_seconds)


def finish(settings: Any, *, job_id: str, run_id: str, clip_id: str | None = None, opening_cut: bool = False, new_hook: bool = False,
           progress: Callable[[str], None] = print, console: Any = None, provider: Any = None) -> dict[str, Any]:
    from . import native_dub as nd
    from .dub_publish import publish_handoff
    from .dub_render import prepare_render_root
    from .experience_store import ExperienceStore
    from .orchestrator import Orchestrator

    clip_id = clip_id or job_id[:8]
    work_dir = Path(settings.work_dir)
    store = ExperienceStore(work_dir)
    root, info, _voices = prepare_render_root(store, work_dir, run_id=run_id, clip_id=clip_id, repeat=1)
    job = Orchestrator(settings).jobs.load(info["job_id"])
    paths = nd.native_dub_paths(root)

    if console is None:
        from .console import Console

        console = Console(work_dir)
    track = studio_track(console, job_id, run_id, progress=progress)
    source = nd.resolve_job_source_video(job, work_dir / "jobs" / job_id)
    master = mux_master(source, track, Path(paths.video))
    progress(f"[finish] dubbed master: {master.name} ({master.stat().st_size / 1e6:.1f} MB)")

    pass2 = nd.read_json(paths.pass2)
    segments = list(nd.read_json(paths.pass1_sentences).get("records") or [])
    if pass2.get("temporal_mask_mode"):
        _groups, translations = nd._temporal_mask_groups_for_render(pass2, segments)
    else:
        translations = {item["segment_id"]: item["spoken_text"] for item in pass2["translations"]}
    if provider is None:
        import os

        from .foundry_grok import create_native_gpt_provider

        provider = create_native_gpt_provider(dict(os.environ), progress=progress)
    publication = nd.render_native_hook_publication(job=job, job_root=root, master_video=master, segments=segments, translations=translations, provider=provider, force=new_hook,
                                                    apply_opening_cut=opening_cut, progress=progress)
    handoff = publish_handoff(work_dir, job_id=job_id, run_id=run_id, clip_id=clip_id)
    selection = json.loads(paths.publication_selection.read_text(encoding="utf-8")).get("selection") or {}
    result = {"run": run_id, "job": job_id, "opening_cut": bool(opening_cut), "hook": selection.get("hook_text"), "hook_spelling_fixes": selection.get("hook_spelling_fixes") or [],
              "master": str(master), "publication_video": str(publication["output"]["path"]), "handoff": handoff}
    progress(f"[finish] done: {handoff.get('final_video')}")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="mathula_tv.dub_finish")
    parser.add_argument("--job", required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--clip", default=None)
    parser.add_argument("--opening-cut", action="store_true", help="also cut the opening to the hook's anchor (default: keep the whole video: the footer card only)")
    parser.add_argument("--new-hook", action="store_true", help="ask the editorial model again instead of keeping the saved hook, caption and hashtags")
    args = parser.parse_args(argv)

    from .config import load_settings

    print(json.dumps(finish(load_settings(), job_id=args.job, run_id=args.run, clip_id=args.clip, opening_cut=args.opening_cut, new_hook=args.new_hook), indent=2, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
