from __future__ import annotations

import re
import wave
from pathlib import Path

from .logging_utils import redact


def inspect_wav(path: Path) -> dict:
    with wave.open(str(path), "rb") as wav:
        channels,sample_rate,width=wav.getnchannels(),wav.getframerate(),wav.getsampwidth(); frames = wav.readframes(wav.getnframes()); duration = wav.getnframes() / sample_rate
    if channels!=1 or width!=2: return {"duration":duration,"channels":channels,"sample_rate":sample_rate,"valid":False}
    values = [int.from_bytes(frames[i:i+2], "little", signed=True) for i in range(0, len(frames), 2)]
    peak = max((abs(v) for v in values), default=0)
    non_silent = sum(abs(v) > 64 for v in values) / max(len(values), 1)
    clipped = sum(abs(v) >= 32760 for v in values) / max(len(values), 1)
    return {"duration":duration,"channels":channels,"sample_rate":sample_rate,"peak":peak,"non_silent_ratio":non_silent,"clipped_ratio":clipped,"valid":duration>0 and non_silent>0.001 and clipped<0.01}


def validate_outputs(job: dict, translation: dict | None = None, seo: dict | None = None) -> list[str]:
    warnings = []
    if translation:
        translated = {t["segment_id"] for t in translation.get("turns", []) if t.get("translated_text") or t.get("status") == "failed"}
        source = set(job.get("source_segment_ids", translated))
        if translated != source: warnings.append("Not every source segment is translated or explicitly failed")
        text = " ".join(t.get("translated_text", "") for t in translation.get("turns", []))
        if text and len(re.findall(r"\b(the|and|is|are|this|that)\b", text.lower())) > len(text.split()) * 0.3: warnings.append("Target-language output may be English-only")
    if seo and (len(seo.get("title", "")) > 100 or sum(len(str(t)) + 1 for t in seo.get("tags", [])) > 500): warnings.append("SEO platform limits exceeded")
    if redact(job) != job: warnings.append("Potential secret found in output")
    return warnings
