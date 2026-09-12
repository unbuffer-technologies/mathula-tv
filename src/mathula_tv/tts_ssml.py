from __future__ import annotations

import hashlib
import io
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from html import escape
from typing import Iterable, Literal, Sequence


SSML_NAMESPACE = "http://www.w3.org/2001/10/synthesis"
XML_NAMESPACE = "http://www.w3.org/XML/1998/namespace"
_PROVENANCE = object()
_VOICE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}\Z")
_LANGUAGE = re.compile(r"[A-Za-z]{2,3}(?:-[A-Za-z0-9]{2,8}){1,2}\Z")
_URL = re.compile(r"(?:https?|file|ftp|data):", re.IGNORECASE)
_BOOKMARK_MARK = re.compile(r"[A-Za-z0-9_.:-]{1,128}\Z")


class SSMLValidationError(ValueError):
    """Raised when application SSML violates the production allowlist."""


@dataclass(frozen=True)
class SSMLBounds:
    rate_min_percent: int = -12
    rate_max_percent: int = 15
    pitch_min_percent: int = -10
    pitch_max_percent: int = 10
    volume_min_percent: int = -6
    volume_max_percent: int = 6
    pause_max_ms: int = 2_000
    total_pause_max_ms: int = 5_000
    max_text_characters: int = 10_000
    max_parts: int = 500

    def __post_init__(self) -> None:
        pairs = (
            ("rate", self.rate_min_percent, self.rate_max_percent),
            ("pitch", self.pitch_min_percent, self.pitch_max_percent),
            ("volume", self.volume_min_percent, self.volume_max_percent),
        )
        for name, lower, upper in pairs:
            if lower > upper:
                raise ValueError(f"SSML {name} minimum cannot exceed its maximum")
        if self.pause_max_ms < 0 or self.total_pause_max_ms < 0:
            raise ValueError("SSML pause bounds cannot be negative")
        if self.max_text_characters <= 0 or self.max_parts <= 0:
            raise ValueError("SSML size bounds must be positive")


@dataclass(frozen=True)
class TextPart:
    text: str


@dataclass(frozen=True)
class BreakPart:
    duration_ms: int


@dataclass(frozen=True)
class EmphasisPart:
    text: str
    level: Literal["reduced", "moderate", "strong"] = "moderate"


@dataclass(frozen=True)
class SubstitutionPart:
    display_text: str
    tts_text: str


@dataclass(frozen=True)
class CharacterPart:
    text: str


SSMLPart = TextPart | BreakPart | EmphasisPart | SubstitutionPart | CharacterPart


@dataclass(frozen=True)
class SSMLDocument:
    xml: str
    sha256: str
    summary: str
    voice: str
    language: str
    rate_percent: int
    pitch_percent: int
    volume_percent: int
    text_characters: int
    break_count: int
    substitution_count: int
    _provenance: object = field(repr=False, compare=False)

    @property
    def application_owned(self) -> bool:
        return self._provenance is _PROVENANCE


def _tag(local_name: str) -> str:
    return f"{{{SSML_NAMESPACE}}}{local_name}"


def _format_signed(value: int, suffix: str) -> str:
    return f"{value:+d}{suffix}"


def _check_text(value: str, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise SSMLValidationError(f"{field_name} must be text")
    if any(ord(character) < 32 and character not in "\t\n\r" for character in value):
        raise SSMLValidationError(f"{field_name} contains disallowed control characters")
    return value


class SafeSSMLBuilder:
    """Build the only SSML shape accepted by the Azure production adapter.

    Callers provide typed parts rather than XML. This keeps untrusted model output
    in text and attribute values, where it is escaped, and makes the allowlist
    enforceable at the application boundary.
    """

    def __init__(
        self,
        *,
        allowed_voices: Iterable[str],
        bounds: SSMLBounds | None = None,
        verified_emphasis_voices: Iterable[str] = (),
    ):
        self.allowed_voices = frozenset(str(voice).strip() for voice in allowed_voices)
        if not self.allowed_voices:
            raise ValueError("At least one Azure TTS voice must be allowlisted")
        for voice in self.allowed_voices:
            if not _VOICE.fullmatch(voice):
                raise ValueError(f"Invalid Azure TTS voice name: {voice!r}")
        self.verified_emphasis_voices = frozenset(verified_emphasis_voices)
        if not self.verified_emphasis_voices.issubset(self.allowed_voices):
            raise ValueError("Verified emphasis voices must be in the Azure TTS voice allowlist")
        self.bounds = bounds or SSMLBounds()

    def build_plain_text(
        self,
        text: str,
        *,
        voice: str,
        language: str = "zu-ZA",
        rate_percent: int = 0,
        pitch_percent: int = 0,
        volume_percent: int = 0,
    ) -> SSMLDocument:
        return self.build(
            [TextPart(text)],
            voice=voice,
            language=language,
            rate_percent=rate_percent,
            pitch_percent=pitch_percent,
            volume_percent=volume_percent,
        )

    def build(
        self,
        parts: Sequence[SSMLPart],
        *,
        voice: str,
        language: str = "zu-ZA",
        rate_percent: int = 0,
        pitch_percent: int = 0,
        volume_percent: int = 0,
    ) -> SSMLDocument:
        if voice not in self.allowed_voices:
            raise SSMLValidationError(f"Azure TTS voice is not allowlisted: {voice}")
        if not _LANGUAGE.fullmatch(language):
            raise SSMLValidationError("SSML language is invalid")
        self._in_bounds("rate", rate_percent, self.bounds.rate_min_percent, self.bounds.rate_max_percent)
        self._in_bounds("pitch", pitch_percent, self.bounds.pitch_min_percent, self.bounds.pitch_max_percent)
        self._in_bounds(
            "volume", volume_percent, self.bounds.volume_min_percent, self.bounds.volume_max_percent
        )
        if not parts:
            raise SSMLValidationError("SSML must contain at least one text part")
        if len(parts) > self.bounds.max_parts:
            raise SSMLValidationError("SSML contains too many parts")

        rendered: list[str] = []
        text_characters = 0
        break_count = 0
        substitution_count = 0
        total_pause_ms = 0
        has_spoken_text = False
        for part in parts:
            if isinstance(part, TextPart):
                text = _check_text(part.text, field_name="SSML text")
                text_characters += len(text)
                has_spoken_text = has_spoken_text or bool(text.strip())
                rendered.append(escape(text, quote=False))
            elif isinstance(part, BreakPart):
                if isinstance(part.duration_ms, bool) or not isinstance(part.duration_ms, int):
                    raise SSMLValidationError("SSML pause must be an integer number of milliseconds")
                if not 0 <= part.duration_ms <= self.bounds.pause_max_ms:
                    raise SSMLValidationError(
                        f"SSML pause must be between 0 and {self.bounds.pause_max_ms} ms"
                    )
                total_pause_ms += part.duration_ms
                break_count += 1
                rendered.append(f'<break time="{part.duration_ms}ms" />')
            elif isinstance(part, EmphasisPart):
                if voice not in self.verified_emphasis_voices:
                    raise SSMLValidationError(
                        f"SSML emphasis has not been verified for Azure TTS voice: {voice}"
                    )
                if part.level not in {"reduced", "moderate", "strong"}:
                    raise SSMLValidationError("SSML emphasis level is not allowlisted")
                text = _check_text(part.text, field_name="SSML emphasis text")
                if not text.strip():
                    raise SSMLValidationError("SSML emphasis text cannot be empty")
                text_characters += len(text)
                has_spoken_text = True
                rendered.append(f'<emphasis level="{part.level}">{escape(text, quote=False)}</emphasis>')
            elif isinstance(part, SubstitutionPart):
                display = _check_text(part.display_text, field_name="SSML substitution display text")
                alias = _check_text(part.tts_text, field_name="SSML substitution TTS text")
                if not display.strip() or not alias.strip():
                    raise SSMLValidationError("SSML substitution values cannot be empty")
                text_characters += len(display) + len(alias)
                has_spoken_text = True
                substitution_count += 1
                rendered.append(
                    f'<sub alias="{escape(alias, quote=True)}">{escape(display, quote=False)}</sub>'
                )
            elif isinstance(part, CharacterPart):
                text = _check_text(part.text, field_name="SSML character text")
                if not text.strip() or not re.fullmatch(r"[A-Za-z0-9]+", text):
                    raise SSMLValidationError(
                        "SSML character text must be non-empty letters or digits"
                    )
                text_characters += len(text)
                has_spoken_text = True
                rendered.append(
                    f'<say-as interpret-as="characters">{escape(text, quote=False)}</say-as>'
                )
            else:
                raise SSMLValidationError(f"Unknown SSML part type: {type(part).__name__}")
        if not has_spoken_text:
            raise SSMLValidationError("SSML must contain non-empty spoken text")
        if text_characters > self.bounds.max_text_characters:
            raise SSMLValidationError("SSML text exceeds the configured character limit")
        if total_pause_ms > self.bounds.total_pause_max_ms:
            raise SSMLValidationError("SSML pauses exceed the configured total pause limit")

        body = "".join(rendered)
        xml = (
            f'<speak version="1.0" xmlns="{SSML_NAMESPACE}" xml:lang="{language}">'
            f'<voice name="{voice}">'
            f'<prosody rate="{_format_signed(rate_percent, "%")}" '
            f'pitch="{_format_signed(pitch_percent, "%")}" '
            f'volume="{_format_signed(volume_percent, "%")}">{body}</prosody>'
            "</voice></speak>"
        )
        validate_ssml(
            xml,
            bounds=self.bounds,
            allowed_voices=self.allowed_voices,
            verified_emphasis_voices=self.verified_emphasis_voices,
        )
        digest = hashlib.sha256(xml.encode("utf-8")).hexdigest()
        summary = (
            f"voice={voice}; language={language}; rate={rate_percent:+d}%; "
            f"pitch={pitch_percent:+d}%; volume={volume_percent:+d}%; "
            f"text_characters={text_characters}; breaks={break_count}; "
            f"substitutions={substitution_count}; total_pause_ms={total_pause_ms}"
        )
        return SSMLDocument(
            xml=xml,
            sha256=digest,
            summary=summary,
            voice=voice,
            language=language,
            rate_percent=rate_percent,
            pitch_percent=pitch_percent,
            volume_percent=volume_percent,
            text_characters=text_characters,
            break_count=break_count,
            substitution_count=substitution_count,
            _provenance=_PROVENANCE,
        )

    @staticmethod
    def _in_bounds(name: str, value: int, lower: int, upper: int) -> None:
        if isinstance(value, bool) or not isinstance(value, int):
            raise SSMLValidationError(f"SSML {name} must be an integer")
        if not lower <= value <= upper:
            raise SSMLValidationError(f"SSML {name} must be between {lower} and {upper}")


def _render_bookmark_island_body(value: "str | Sequence[SSMLPart]", *, mark: str) -> tuple[str, int]:
    """Render one bookmarked island's spoken content to an XML fragment plus
    its real text-character count (for ``build_group_bookmark_ssml``'s shared
    ``max_text_characters`` bound).

    A plain string is the common case (one implicit ``TextPart``) and renders
    byte-identically to this function's original, string-only contract.  A
    sequence of parts is scoped deliberately narrow -- only ``TextPart`` and
    ``CharacterPart`` -- matching this module's own long-standing reason
    islands never needed the full ``SSMLPart`` union (``build_island_phrase_
    groups`` has never needed ``BreakPart``/``EmphasisPart``/
    ``SubstitutionPart`` inside one island). ``CharacterPart`` support exists
    specifically for a code-switched acronym that must be spelled letter-by-
    letter (e.g. "ANC") within an otherwise plain-text island -- confirmed
    real defect: without it, "ANC" reached Azure as unmarked literal text
    with no letter-spelling markup, left entirely to the voice's own guess.
    """
    if isinstance(value, str):
        checked_text = _check_text(value, field_name="SSML island text")
        if not checked_text.strip():
            raise SSMLValidationError(f"SSML island text cannot be empty: {mark!r}")
        return escape(checked_text, quote=False), len(checked_text)

    fragments: list[str] = []
    text_characters = 0
    has_spoken_text = False
    for part in value:
        if isinstance(part, TextPart):
            checked_text = _check_text(part.text, field_name="SSML island text")
            fragments.append(escape(checked_text, quote=False))
            text_characters += len(checked_text)
            if checked_text.strip():
                has_spoken_text = True
        elif isinstance(part, CharacterPart):
            checked_text = _check_text(part.text, field_name="SSML character text")
            if not checked_text.strip() or not re.fullmatch(r"[A-Za-z0-9]+", checked_text):
                raise SSMLValidationError("SSML character text must be non-empty letters or digits")
            fragments.append(f'<say-as interpret-as="characters">{escape(checked_text, quote=False)}</say-as>')
            text_characters += len(checked_text)
            has_spoken_text = True
        else:
            raise SSMLValidationError(
                f"SSML island parts only support TextPart/CharacterPart, got {type(part).__name__}"
            )
    if not has_spoken_text:
        raise SSMLValidationError(f"SSML island text cannot be empty: {mark!r}")
    return "".join(fragments), text_characters


def build_group_bookmark_ssml(
    islands: Sequence[tuple[str, "str | Sequence[SSMLPart]"]],
    *,
    voice: str,
    allowed_voices: Iterable[str],
    bounds: SSMLBounds | None = None,
    verified_emphasis_voices: Iterable[str] = (),
    language: str = "zu-ZA",
    rate_percent: int = 0,
    pitch_percent: int = 0,
    volume_percent: int = 0,
) -> tuple[str, str]:
    """Build one bookmarked SSML document covering a whole group's speech islands.

    Each island is marked with a self-closing ``<bookmark mark="{island_id}"/>``
    immediately before its text, so a single Speech SDK synthesis call can report
    the real audio offset where each island's speech begins via the
    ``BookmarkReached`` event (see ``speech_islands.slice_wav_by_bookmarks``).
    An island's content is normally a plain string (``build_island_phrase_groups``
    emits a single plain-text part per island); it may also be a sequence of
    ``TextPart``/``CharacterPart`` when one island needs a letter-spelled
    acronym mid-text (see ``_render_bookmark_island_body``) -- still a narrow
    subset of the full ``SSMLPart`` union, not general island formatting.
    Returns ``(xml, sha256)`` rather than an ``SSMLDocument`` since there is no
    other consumer of that dataclass's per-request metadata parity contract
    (``validate_document``) for a bookmarked group document.
    """
    allowed_voices_set = frozenset(str(item).strip() for item in allowed_voices)
    if voice not in allowed_voices_set:
        raise SSMLValidationError(f"Azure TTS voice is not allowlisted: {voice}")
    if not _LANGUAGE.fullmatch(language):
        raise SSMLValidationError("SSML language is invalid")
    resolved_bounds = bounds or SSMLBounds()
    SafeSSMLBuilder._in_bounds(
        "rate", rate_percent, resolved_bounds.rate_min_percent, resolved_bounds.rate_max_percent
    )
    SafeSSMLBuilder._in_bounds(
        "pitch", pitch_percent, resolved_bounds.pitch_min_percent, resolved_bounds.pitch_max_percent
    )
    SafeSSMLBuilder._in_bounds(
        "volume", volume_percent, resolved_bounds.volume_min_percent, resolved_bounds.volume_max_percent
    )
    if not islands:
        raise SSMLValidationError("SSML must contain at least one island")
    if len(islands) > resolved_bounds.max_parts:
        raise SSMLValidationError("SSML contains too many parts")

    seen_marks: set[str] = set()
    rendered: list[str] = []
    text_characters = 0
    for island_id, content in islands:
        mark = str(island_id)
        if not _BOOKMARK_MARK.fullmatch(mark):
            raise SSMLValidationError(f"Invalid SSML bookmark mark: {mark!r}")
        if mark in seen_marks:
            raise SSMLValidationError(f"Duplicate SSML bookmark mark: {mark!r}")
        seen_marks.add(mark)
        island_body, island_char_count = _render_bookmark_island_body(content, mark=mark)
        text_characters += island_char_count
        rendered.append(f'<bookmark mark="{escape(mark, quote=True)}" />')
        rendered.append(island_body)
    if text_characters > resolved_bounds.max_text_characters:
        raise SSMLValidationError("SSML text exceeds the configured character limit")

    body = "".join(rendered)
    xml = (
        f'<speak version="1.0" xmlns="{SSML_NAMESPACE}" xml:lang="{language}">'
        f'<voice name="{voice}">'
        f'<prosody rate="{_format_signed(rate_percent, "%")}" '
        f'pitch="{_format_signed(pitch_percent, "%")}" '
        f'volume="{_format_signed(volume_percent, "%")}">{body}</prosody>'
        "</voice></speak>"
    )
    validate_ssml(
        xml,
        bounds=resolved_bounds,
        allowed_voices=allowed_voices_set,
        verified_emphasis_voices=verified_emphasis_voices,
    )
    digest = hashlib.sha256(xml.encode("utf-8")).hexdigest()
    return xml, digest


def validate_document(
    document: SSMLDocument,
    *,
    bounds: SSMLBounds,
    allowed_voices: Iterable[str],
    verified_emphasis_voices: Iterable[str] = (),
) -> None:
    if not isinstance(document, SSMLDocument) or not document.application_owned:
        raise SSMLValidationError("Azure TTS accepts only application-built SSML documents")
    expected_hash = hashlib.sha256(document.xml.encode("utf-8")).hexdigest()
    if expected_hash != document.sha256:
        raise SSMLValidationError("SSML document hash does not match its content")
    metadata = validate_ssml(
        document.xml,
        bounds=bounds,
        allowed_voices=allowed_voices,
        verified_emphasis_voices=verified_emphasis_voices,
    )
    expected = {
        "voice": document.voice,
        "language": document.language,
        "rate_percent": document.rate_percent,
        "pitch_percent": document.pitch_percent,
        "volume_percent": document.volume_percent,
        "text_characters": document.text_characters,
        "break_count": document.break_count,
        "substitution_count": document.substitution_count,
    }
    if any(metadata[name] != value for name, value in expected.items()):
        raise SSMLValidationError("SSML document metadata does not match its XML")
    expected_summary = (
        f"voice={document.voice}; language={document.language}; rate={document.rate_percent:+d}%; "
        f"pitch={document.pitch_percent:+d}%; volume={document.volume_percent:+d}%; "
        f"text_characters={document.text_characters}; breaks={document.break_count}; "
        f"substitutions={document.substitution_count}; total_pause_ms={metadata['total_pause_ms']}"
    )
    if document.summary != expected_summary:
        raise SSMLValidationError("SSML document summary does not match its XML")


def validate_ssml(
    xml: str,
    *,
    bounds: SSMLBounds,
    allowed_voices: Iterable[str],
    verified_emphasis_voices: Iterable[str] = (),
) -> dict[str, int | str]:
    """Strictly validate serialized SSML, including documents read from disk."""

    if not isinstance(xml, str) or not xml:
        raise SSMLValidationError("SSML document is empty")
    upper = xml.upper()
    if "<!DOCTYPE" in upper or "<!ENTITY" in upper:
        raise SSMLValidationError("SSML declarations and entities are not allowed")
    if "<?" in xml:
        raise SSMLValidationError("SSML processing instructions are not allowed")
    try:
        root = ET.fromstring(xml)
    except ET.ParseError as exc:
        raise SSMLValidationError("SSML is not valid XML") from exc
    try:
        namespaces = list(ET.iterparse(io.StringIO(xml), events=("start-ns",)))
    except ET.ParseError as exc:
        raise SSMLValidationError("SSML namespace declarations are invalid") from exc
    for _, (prefix, namespace) in namespaces:
        if (prefix or "") == "" and namespace == SSML_NAMESPACE:
            continue
        if prefix == "xml" and namespace == XML_NAMESPACE:
            continue
        raise SSMLValidationError("SSML contains an unknown namespace declaration")

    allowed_voices = frozenset(allowed_voices)
    verified_emphasis_voices = frozenset(verified_emphasis_voices)
    allowed_tags = {
        _tag(name)
        for name in (
            "speak",
            "voice",
            "prosody",
            "break",
            "emphasis",
            "sub",
            "say-as",
            "bookmark",
        )
    }
    allowed_attributes = {
        _tag("speak"): {"version", f"{{{XML_NAMESPACE}}}lang"},
        _tag("voice"): {"name"},
        _tag("prosody"): {"rate", "pitch", "volume"},
        _tag("break"): {"time"},
        _tag("emphasis"): {"level"},
        _tag("sub"): {"alias"},
        _tag("say-as"): {"interpret-as"},
        _tag("bookmark"): {"mark"},
    }
    for node in root.iter():
        if node.tag not in allowed_tags:
            raise SSMLValidationError("SSML contains an unknown element or namespace")
        unknown_attributes = set(node.attrib) - allowed_attributes[node.tag]
        if unknown_attributes:
            raise SSMLValidationError("SSML contains an unknown attribute or namespace")
        if any(_URL.search(value) for value in node.attrib.values()):
            raise SSMLValidationError("External URLs are not allowed in SSML")

    if root.tag != _tag("speak") or root.attrib != {
        "version": "1.0",
        f"{{{XML_NAMESPACE}}}lang": root.attrib.get(f"{{{XML_NAMESPACE}}}lang"),
    }:
        raise SSMLValidationError("SSML speak attributes are invalid")
    language = root.attrib.get(f"{{{XML_NAMESPACE}}}lang", "")
    if not _LANGUAGE.fullmatch(language):
        raise SSMLValidationError("SSML language is invalid")
    if root.text and root.text.strip():
        raise SSMLValidationError("Text outside the SSML voice element is not allowed")
    if len(root) != 1 or root[0].tag != _tag("voice"):
        raise SSMLValidationError("SSML must contain exactly one voice")
    voice = root[0]
    if voice.attrib.get("name") not in allowed_voices:
        raise SSMLValidationError("SSML voice is not allowlisted")
    if voice.text and voice.text.strip():
        raise SSMLValidationError("Text outside the SSML prosody element is not allowed")
    if len(voice) != 1 or voice[0].tag != _tag("prosody"):
        raise SSMLValidationError("SSML voice must contain exactly one prosody element")
    if voice.tail and voice.tail.strip():
        raise SSMLValidationError("Text outside the SSML voice element is not allowed")
    prosody = voice[0]
    if prosody.tail and prosody.tail.strip():
        raise SSMLValidationError("Text outside the SSML prosody element is not allowed")
    expected_prosody_attributes = {"rate", "pitch", "volume"}
    if set(prosody.attrib) != expected_prosody_attributes:
        raise SSMLValidationError("SSML prosody attributes are invalid")
    rate = _parse_measure(prosody.attrib["rate"], "%", "rate")
    pitch = _parse_measure(prosody.attrib["pitch"], "%", "pitch")
    volume = _parse_measure(prosody.attrib["volume"], "%", "volume")
    SafeSSMLBuilder._in_bounds("rate", rate, bounds.rate_min_percent, bounds.rate_max_percent)
    SafeSSMLBuilder._in_bounds("pitch", pitch, bounds.pitch_min_percent, bounds.pitch_max_percent)
    SafeSSMLBuilder._in_bounds(
        "volume", volume, bounds.volume_min_percent, bounds.volume_max_percent
    )

    counts = _validate_prosody_children(
        prosody, voice_name=voice.attrib["name"], bounds=bounds, verified_emphasis_voices=verified_emphasis_voices,
    )
    if not counts["has_spoken_text"]:
        raise SSMLValidationError("SSML must contain non-empty spoken text")
    if counts["parts"] > bounds.max_parts or counts["text_characters"] > bounds.max_text_characters:
        raise SSMLValidationError("SSML exceeds configured size bounds")
    if counts["total_pause_ms"] > bounds.total_pause_max_ms:
        raise SSMLValidationError("SSML pauses exceed the configured total pause limit")
    return {
        "voice": voice.attrib["name"],
        "language": language,
        "rate_percent": rate,
        "pitch_percent": pitch,
        "volume_percent": volume,
        "text_characters": counts["text_characters"],
        "break_count": counts["break_count"],
        "substitution_count": counts["substitution_count"],
        "total_pause_ms": counts["total_pause_ms"],
    }


def _validate_prosody_children(
    prosody: ET.Element,
    *,
    voice_name: str,
    bounds: SSMLBounds,
    verified_emphasis_voices: Iterable[str],
) -> dict[str, int | bool]:
    """Validate and tally one <prosody> element's own children (break/emphasis/
    sub/say-as/bookmark plus interleaved text) -- extracted so ``validate_ssml``
    doesn't hand-duplicate this per-element logic inline.
    """
    verified_emphasis_voices = frozenset(verified_emphasis_voices)
    text_characters = len(prosody.text or "")
    total_pause_ms = 0
    break_count = 0
    substitution_count = 0
    parts = 1 if prosody.text else 0
    has_spoken_text = bool((prosody.text or "").strip())
    for child in prosody:
        parts += 1
        if len(child):
            raise SSMLValidationError("Nested SSML elements are not allowed")
        if child.tag == _tag("break"):
            break_count += 1
            duration = _parse_measure(child.attrib.get("time", ""), "ms", "pause")
            if not 0 <= duration <= bounds.pause_max_ms:
                raise SSMLValidationError("SSML pause is outside configured bounds")
            total_pause_ms += duration
            if child.text and child.text.strip():
                raise SSMLValidationError("SSML break elements cannot contain text")
        elif child.tag == _tag("emphasis"):
            if voice_name not in verified_emphasis_voices:
                raise SSMLValidationError("SSML emphasis is not verified for the selected voice")
            if child.attrib.get("level") not in {"reduced", "moderate", "strong"}:
                raise SSMLValidationError("SSML emphasis level is not allowlisted")
            if not (child.text or "").strip():
                raise SSMLValidationError("SSML emphasis text cannot be empty")
            text_characters += len(child.text or "")
            has_spoken_text = True
        elif child.tag == _tag("sub"):
            substitution_count += 1
            if not (child.attrib.get("alias") or "").strip() or not (child.text or "").strip():
                raise SSMLValidationError("SSML substitution values cannot be empty")
            text_characters += len(child.text or "") + len(child.attrib["alias"])
            has_spoken_text = True
        elif child.tag == _tag("say-as"):
            if child.attrib.get("interpret-as") != "characters":
                raise SSMLValidationError(
                    "SSML say-as interpretation is not allowlisted"
                )
            if not re.fullmatch(r"[A-Za-z0-9]+", child.text or ""):
                raise SSMLValidationError(
                    "SSML character text must be non-empty letters or digits"
                )
            text_characters += len(child.text or "")
            has_spoken_text = True
        elif child.tag == _tag("bookmark"):
            if child.text and child.text.strip():
                raise SSMLValidationError("SSML bookmark elements cannot contain text")
            if not _BOOKMARK_MARK.fullmatch(child.attrib.get("mark", "")):
                raise SSMLValidationError("SSML bookmark mark is invalid")
        else:
            raise SSMLValidationError("SSML contains an element in a disallowed position")
        if child.tail:
            text_characters += len(child.tail)
            has_spoken_text = has_spoken_text or bool(child.tail.strip())
            parts += 1
    return {
        "text_characters": text_characters,
        "break_count": break_count,
        "substitution_count": substitution_count,
        "total_pause_ms": total_pause_ms,
        "parts": parts,
        "has_spoken_text": has_spoken_text,
    }


def _parse_measure(value: str, suffix: str, field_name: str) -> int:
    pattern = rf"[+-]?\d+{re.escape(suffix)}\Z"
    if not re.fullmatch(pattern, value):
        raise SSMLValidationError(f"SSML {field_name} has an invalid value")
    return int(value[: -len(suffix)])
