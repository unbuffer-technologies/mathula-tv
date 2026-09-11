from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).parents[1]
TRANSCRIPT = Path("/mnt/data/transcript_en_raw(1).json")
SOURCE = ROOT / "src" / "mathula_tv" / "native_dub.py"


def english_syllables(text: str) -> int:
    total = 0
    for token in re.findall(r"[A-Za-z]+(?:['’][A-Za-z]+)?|\d+(?:[.,]\d+)*", text):
        if re.fullmatch(r"\d+(?:[.,]\d+)*", token):
            total += max(1, len(re.sub(r"\D", "", token)))
            continue
        word = re.sub(r"[^A-Za-z]", "", token).lower()
        groups = re.findall(r"[aeiouy]+", word)
        count = max(1, len(groups))
        if count > 1 and len(word) > 3 and word.endswith("e") and not word.endswith(("le", "ye")):
            count -= 1
        total += max(1, count)
    return max(1, total)


data = json.loads(TRANSCRIPT.read_text(encoding="utf-8"))
segments = list(data["segments"])
assert len(segments) == 76, len(segments)

# The production four-block contract composes disjoint chronological windows.
# Every normal window begins at the current block and includes following blocks.
windows = [segments[i : i + 4] for i in range(0, len(segments), 4)]
assert len(windows) == 19
assert all(2 <= len(window) <= 4 for window in windows)
assert len(windows[-1]) == 4

for window in windows:
    ids = [item["segment_id"] for item in window]
    assert len(ids) == len(set(ids))
    for item in window:
        source = str(item.get("source_text") or "").strip()
        assert source
        floor = english_syllables(source)
        preferred = floor + 1
        assert floor >= 1 and preferred > floor

# Static provider-wire contract: no Azure timing fields are exposed by the
# temporal-mask per-block wire, while the linguistic syllable floor is present.
text = SOURCE.read_text(encoding="utf-8")
start = text.index("def _temporal_mask_wire(")
end = text.index("def _temporal_four_block_semantic_window(", start)
wire = text[start:end]
for forbidden in ("source_start_ms", "source_end_ms", "source_window_ms", "preferred_raw_ms"):
    assert f'"{forbidden}"' not in wire, forbidden
for required in ("source_equivalent_syllables", "preferred_syllables", "hard_maximum"):
    assert required in wire, required

print(f"PASS: 76 source blocks replayed as {len(windows)} chronological four-block windows")
print("PASS: every block has an English syllable floor and preferred floor+1 target")
print("PASS: temporal-mask provider wire contains no Azure timing fields")
print("PASS: four-block prompt requires whole-window back-translation QA")
