"""After the render: were the quantities in the source spoken like quantities?

The source says "being paid 4,500". A listener expects "four thousand five hundred". The dub once sent the voice
"u-4,fayiv zeeroh zeeroh": a digit, then the remaining digits spelt one by one, which is how a docket number is read, not
an amount. The meaning check could not see it (the digits were all there), and neither could the dropped-word check.

For every unit this looks at what the voice was actually sent (the SSML text) and reports a quantity-shaped number of the
source that went out digit by digit. It reports only; it never changes the dub.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Mapping, Sequence

from .pronunciation import _ZU_ENGLISH_DIGIT_WORDS

# 4,500 / R19 500 / 12,000,000: a grouped number, or an amount with the R sign. A bare 3-digit number could be a reference.
_QUANTITY = re.compile(r"(?<![\d,.])(?:R\s?)?[1-9][0-9]{0,2}(?:[,  ][0-9]{3})+(?![\d])|(?<![\d,.])R\s?[0-9]{3,}(?![\d,])")
_SEPARATOR = r"[\s,\-]*"


def quantities(text: str) -> list[str]:
    """Digit strings of the quantity-shaped numbers in ``text`` (4,500 -> '4500')."""
    return [re.sub(r"\D", "", m.group(0)) for m in _QUANTITY.finditer(text)]


def spelt_digits_pattern(digits: str, tail: int = 3) -> re.Pattern[str]:
    """Matches the last ``tail`` digits of a number when they were spelt one by one ("fayiv zeeroh zeeroh")."""
    words = [re.escape(_ZU_ENGLISH_DIGIT_WORDS[d]) for d in digits[-tail:]]
    return re.compile(r"(?<![A-Za-z])" + _SEPARATOR.join(words) + r"(?![A-Za-z])", re.IGNORECASE)


def digit_by_digit(source_text: str, tts_text: str) -> list[dict[str, str]]:
    """Quantities of the source that the voice was sent as spelt-out digits."""
    flagged = []
    for digits in dict.fromkeys(quantities(source_text)):
        if len(digits) >= 3 and spelt_digits_pattern(digits).search(tts_text):
            flagged.append({"number": digits, "problem": "sent to the voice digit by digit, not as an amount"})
    return flagged


def check_numbers(run_dir: Path) -> list[dict[str, Any]]:
    """Check every unit of a rendered run (scorecard + the SSML the voice was sent)."""
    import json

    run_dir = Path(run_dir)
    scorecard = json.loads((run_dir / "scorecard.json").read_text(encoding="utf-8"))
    phrases = run_dir / "render" / "native_dub" / "audio" / "phrases"
    flags: list[dict[str, Any]] = []
    for group in scorecard.get("groups") or []:
        ssml_path = phrases / f"{group.get('group_id')}.azure_tts.ssml"
        if not ssml_path.is_file():
            continue
        tts_text = re.sub(r"<[^>]+>", " ", ssml_path.read_text(encoding="utf-8"))
        for item in digit_by_digit(str(group.get("source_text") or ""), tts_text):
            flags.append({"unit": group.get("group_id"), "speaker": group.get("speaker_id"), **item})
    return flags
