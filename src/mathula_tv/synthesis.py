from __future__ import annotations

import wave
import re
from pathlib import Path


def select_speaker_references(turns: list[dict], *, min_duration: float = 3.0, preferred_max_duration: float = 15.0, absolute_max_duration: float = 25.0, max_references: int = 1) -> dict[str, list[dict]]:
    selected: dict[str, list[dict]] = {}
    for turn in sorted(turns, key=lambda t: (bool(t.get("overlap")), -float(t["end"] - t["start"]))):
        speaker = turn["speaker"]
        duration=float(turn["end"]-turn["start"])
        if speaker == "UNKNOWN" or turn.get("overlap") or duration < min_duration or float(turn.get("silence_ratio", 0)) > 0.3 or float(turn.get("noise_score", 0)) > 0.7:
            continue
        if len(selected.setdefault(speaker, [])) < max_references:
            end=min(float(turn["end"]),float(turn["start"])+min(preferred_max_duration,absolute_max_duration))
            selected[speaker].append({"speaker":speaker,"start":float(turn["start"]),"end":end,"duration":end-float(turn["start"]),"verified_single_speaker":True})
    return selected


def unresolved_speakers(turns: list[dict], references: dict[str, list[dict]]) -> list[str]:
    return sorted({t["speaker"] for t in turns if t["speaker"] != "UNKNOWN"} - references.keys())


def plan_synthesis_units(turn:dict,min_seconds:float=5,max_seconds:float=15)->list[dict]:
    text=str(turn.get("translated_text","")).strip()
    if not text: raise ValueError(f"Empty translation for {turn.get('segment_id')}")
    duration=float(turn["end"])-float(turn["start"])
    pieces=[x.strip() for x in re.split(r"(?<=[.!?…])\s+",text) if x.strip()]
    if not pieces: pieces=[text]
    expanded=[]
    for piece in pieces:
        estimate=max(1.,duration*len(piece)/max(len(text),1))
        if estimate<=max_seconds: expanded.append(piece); continue
        words=piece.split(); chunks=max(2,int(estimate/max_seconds)+1); size=max(1,(len(words)+chunks-1)//chunks)
        expanded.extend(" ".join(words[i:i+size]) for i in range(0,len(words),size))
    weights=[len(x) for x in expanded]; total=sum(weights); cursor=float(turn["start"]); units=[]
    for i,(piece,weight) in enumerate(zip(expanded,weights),1):
        unit_duration=duration*weight/total; end=float(turn["end"]) if i==len(expanded) else cursor+unit_duration
        units.append({"unit_id":f"{turn['segment_id']}-u{i:03d}","segment_id":turn["segment_id"],"speaker":turn["speaker"],"start":cursor,"end":end,"source_text":turn["source_text"],"translated_text":turn["translated_text"],"unit_text":piece,"delivery_style":turn.get("delivery_style"),"emotional_intensity":turn.get("emotional_intensity"),"attribution_required":turn.get("attribution_required"),"uncertainty_flags":turn.get("uncertainty_flags",[])})
        cursor=end
    return units


def reference_text(words: list[dict], start: float, end: float, speaker: str) -> str:
    selected=[str(w.get("text","")).strip() for w in words if w.get("speaker")==speaker and start <= (float(w["start"])+float(w["end"]))/2 <= end]
    text=" ".join(filter(None,selected)).strip()
    if not text: raise ValueError(f"No exact transcript words found for reference speaker {speaker}")
    return text


def extract_reference_wav(source: Path, output: Path, start: float, end: float) -> None:
    with wave.open(str(source),"rb") as src:
        if src.getnchannels()!=1 or src.getsampwidth()!=2: raise ValueError("Source reference audio must be mono PCM16")
        rate=src.getframerate(); src.setpos(max(0,int(start*rate))); frames=src.readframes(max(1,int((end-start)*rate)))
    output.parent.mkdir(parents=True,exist_ok=True)
    with wave.open(str(output),"wb") as dst: dst.setparams((1,2,rate,0,"NONE","not compressed")); dst.writeframes(frames)


def assemble_timeline(clips: list[dict], output: Path, total_duration: float, rate: int = 24000) -> dict:
    original_duration=total_duration
    furthest=max([total_duration]+[float(c["start"])+float(c.get("duration",0)) for c in clips])
    total_duration=furthest
    frames = bytearray(int(total_duration * rate) * 2)
    overlaps, clipping = [], 0
    occupied = [False] * int(total_duration * rate)
    for clip in clips:
        with wave.open(str(clip["path"]), "rb") as wav:
            if wav.getnchannels() != 1 or wav.getsampwidth() != 2 or wav.getframerate() != rate: raise ValueError("Synthesized clip must be mono 24 kHz PCM16")
            raw = wav.readframes(wav.getnframes())
        start = int(float(clip["start"]) * rate)
        for i in range(0, min(len(raw), len(frames) - start * 2), 2):
            index = start + i // 2
            value = int.from_bytes(raw[i:i+2], "little", signed=True)
            existing = int.from_bytes(frames[index*2:index*2+2], "little", signed=True)
            if occupied[index]: overlaps.append(clip.get("segment_id"))
            mixed = existing + value
            if abs(mixed) > 32767: clipping += 1
            frames[index*2:index*2+2] = max(-32768, min(32767, mixed)).to_bytes(2, "little", signed=True)
            occupied[index] = True
    output.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(output), "wb") as wav: wav.setparams((1, 2, rate, 0, "NONE", "not compressed")); wav.writeframes(frames)
    return {"final_duration":total_duration,"overlaps":sorted(set(filter(None,overlaps))),"clipped_samples":clipping,"time_adjustments":[],"extended":total_duration>original_duration}
