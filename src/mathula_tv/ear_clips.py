"""Short clips for the ear: the sentences of a dub that contain given words, spoken with the job's own pronunciations.

    python -m mathula_tv.ear_clips --job <job id> --run <run id> --words Jacob,WhatsApp,News [--out folder]

The ear is the only judge of a pronunciation, and a full render of a 12-minute dub to hear three words is the slow way to ask. This says only the
sentences that hold the words (up to ``PER_WORD`` for each, preferring the shortest, each speaker's voice once), through the same dictionary and
the same request the renderer uses, so a clip sounds like the render. Phrases the voice has said before come from the cache and cost nothing.
"""
from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

PER_WORD = 2
GAP_MS = 700
_SENTENCE_END = re.compile(r"(?<=[.?!;:])\s+|~pause\d+\s*")


def sentences(text: str) -> list[str]:
    """The sentences of a unit's text, without the held-pause marks."""
    return [piece.strip() for piece in _SENTENCE_END.split(text) if piece and piece.strip() and not piece.strip().startswith("~pause")]


def word_pattern(word: str) -> "re.Pattern[str]":
    """The word wherever the Zulu text writes it: bare, after a hyphenated prefix (i-MK) or with a Zulu prefix glued on (uJacob).  Group 1 is the prefix, group 2 the word."""
    # a glued prefix is taken only before a capital ("uJacob", "namaPakistan"), as the renderer does: "nhlobo" does not hold "lobo" and "kwehle" does not hold "wehle"
    return re.compile(r"(?<![A-Za-z'])((?:[A-Za-z]{1,6}-)?(?:(?-i:[A-Za-z]{1,4}(?=" + re.escape(word[:1].upper()) + r")))?)(" + re.escape(word) + r")(?![A-Za-z'])", re.IGNORECASE)


def substitute(sentence: str, word: str, spelling: str) -> str:
    """The sentence with the word replaced by a spelling, its Zulu prefix kept: how the voice would be asked to say that spelling in this sentence."""
    return word_pattern(word).sub(lambda match: match.group(1) + spelling, sentence)


def pick(items: Sequence[Mapping[str, Any]], words: Iterable[str], voices: Mapping[str, str], *, per_word: int = PER_WORD) -> list[dict[str, Any]]:
    """For each word, up to ``per_word`` sentences that hold it: one per voice first, shortest first."""
    chosen: list[dict[str, Any]] = []
    for word in words:
        pattern = word_pattern(word)
        found = []
        for item in items:
            voice = voices.get(str(item.get("speaker_id") or ""), "")
            for sentence in sentences(str(item.get("zulu_text") or "")):
                if pattern.search(sentence):
                    found.append({"word": word, "unit": str(item.get("item_id")), "speaker_id": str(item.get("speaker_id") or ""), "voice": voice, "text": sentence})
        found.sort(key=lambda row: len(row["text"]))
        taken: list[dict[str, Any]] = []
        for row in found:                                      # a different voice first: the two voices say the same word differently
            if len(taken) < per_word and all(row["voice"] != other["voice"] for other in taken):
                taken.append(row)
        for row in found:
            if len(taken) < per_word and row not in taken:
                taken.append(row)
        chosen.extend(taken)
    return chosen


def make_clips(
    job_root: Path, items: Sequence[Mapping[str, Any]], words: Sequence[str], out_dir: Path, *, tts: Any,
    synthesize: Callable[..., Any] | None = None, per_word: int = PER_WORD,
) -> list[dict[str, Any]]:
    """Speak the picked sentences with the job's dictionary and return what was made (``path`` of each wav, with its word, unit and text)."""
    from .native_dub import _apply_native_pronunciation_research_to_job_cache, _synthesize_group

    job_root, out_dir = Path(job_root), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    resolution = json.loads((job_root / "direct_dub" / "voice_resolution.json").read_text(encoding="utf-8"))
    assignments = resolution.get("assignments", resolution)
    voices = {speaker: str(a.get("selected_voice") or "") for speaker, a in assignments.items() if isinstance(a, Mapping)}
    _apply_native_pronunciation_research_to_job_cache(job_root, {"accepted_corrections": []})          # the dictionary the renderer reads: the job's lexicon
    speak = synthesize or _synthesize_group
    made = []
    for index, row in enumerate(pick(items, words, voices, per_word=per_word), start=1):
        voice_info = assignments[row["speaker_id"]]
        slug = re.sub(r"[^A-Za-z0-9]+", "_", row["word"]).strip("_")
        path = out_dir / f"{index:02d}_{slug}_{row['voice'].split('-')[-1].replace('Neural', '')}.wav"
        group = {"group_id": f"clip{index:02d}", "speaker_id": row["speaker_id"], "parts": [{"type": "text", "text": row["text"]}]}
        speak(tts=tts, group=group, voice_info=voice_info, output_path=path, preferred_ms=max(2000, 120 * len(row["text"])), maximum_ms=max(4000, 240 * len(row["text"])),
              force=False, job_root=job_root)
        made.append({**row, "path": str(path)})
    return made


def join_clips(clips: Sequence[Mapping[str, Any]], target: Path, *, gap_ms: int = GAP_MS) -> Path | None:
    """The clips one after another with a pause between, as one mp3 (ffmpeg); None when there is nothing to join or no ffmpeg."""
    if not clips:
        return None
    target = Path(target)
    command = ["ffmpeg", "-y", "-loglevel", "error"]
    filters = []
    for index, clip in enumerate(clips):
        command += ["-i", str(clip["path"])]
        filters.append(f"[{index}:a]apad=pad_dur={gap_ms / 1000}[a{index}]")
    chain = ";".join(filters) + ";" + "".join(f"[a{i}]" for i in range(len(clips))) + f"concat=n={len(clips)}:v=0:a=1[out]"
    command += ["-filter_complex", chain, "-map", "[out]", "-codec:a", "libmp3lame", "-b:a", "96k", str(target)]
    try:
        subprocess.run(command, check=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    return target


def main(argv: list[str] | None = None) -> int:
    import argparse

    from .cli import create_tts_backend
    from .config import load_settings
    from .experience_store import ExperienceStore

    parser = argparse.ArgumentParser(prog="mathula_tv.ear_clips")
    parser.add_argument("--job", required=True)
    parser.add_argument("--run", required=True, help="the run whose text holds the sentences")
    parser.add_argument("--words", required=True, help="comma separated words, as written in the Zulu text")
    parser.add_argument("--clip-id")
    parser.add_argument("--out")
    parser.add_argument("--per-word", type=int, default=PER_WORD)
    args = parser.parse_args(argv)
    settings = load_settings()
    job_root = Path(settings.work_dir) / "jobs" / args.job
    clip_id = args.clip_id or args.job[:8]
    submission = json.loads((ExperienceStore(settings.work_dir).repeat_dir(args.run, clip_id, 1) / "submission.json").read_text(encoding="utf-8"))
    out = Path(args.out) if args.out else Path(settings.work_dir) / "ear_clips" / f"{args.job[:8]}_{args.run}"
    words = [w.strip() for w in args.words.split(",") if w.strip()]
    made = make_clips(job_root, submission["items"], words, out, tts=create_tts_backend(settings), per_word=args.per_word)
    index = out / "index.txt"
    index.write_text("".join(f"{i:02d}  {c['word']}  [{c['voice'].split('-')[-1]}]  {c['unit']}: {c['text']}\n" for i, c in enumerate(made, 1)), encoding="utf-8")
    joined = join_clips(made, out / "ear_pack.mp3")
    print(f"{len(made)} clip(s) for {len(words)} word(s) -> {out}" + (f"; one file: {joined}" if joined else ""))
    for clip in made:
        print(f"  {clip['word']:14s} {clip['voice'].split('-')[-1]:14s} {clip['text'][:90]}")
    missing = [w for w in words if not any(c["word"] == w for c in made)]
    if missing:
        print("  no sentence holds:", ", ".join(missing))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
