# Pronunciation and target text

Every translated unit preserves three representations:

```json
{
  "faithful_translation": "Complete semantically faithful isiZulu.",
  "spoken_text": "Natural timing-aware isiZulu actually spoken.",
  "tts_text": "Reviewed Azure-friendly pronunciation form."
}
```

- `faithful_translation` is the meaning record and must retain critical facts.
- `spoken_text` is the approved delivered wording. Subtitles and retranscription QC use it.
- `tts_text` may differ only for recorded pronunciation substitutions. It never appears in audience subtitles.

Record every omission, contraction, paraphrase, expansion, and every `spoken_text` → `tts_text` difference. Timing pressure must not silently remove names, numbers, dates, negation, attribution, or other critical detail.

## Versioned dictionary

Global versioned entries and per-job overrides cover personal/place/organisation names, initials, acronyms, numbers, dates, percentages, currency, English code-switching, and Commission/political terminology. An entry counts as reviewed only with real reviewer evidence; a reviewed example is:

```json
{
  "display_text": "Feroz Khan",
  "spoken_text": "Feroz Khan",
  "tts_text": "Feh-rohz Kaan",
  "language": "zu-ZA",
  "source": "human_review",
  "reviewed_by": "named-reviewer",
  "reviewed_at": "2026-07-20T12:00:00+00:00",
  "confidence": 0.9,
  "notes": []
}
```

Precedence is explicit per-job reviewed override, then global reviewed entry, then unchanged text with a review flag. Null/missing reviewer evidence remains provisional and cannot be called reviewed. Never invent reviewer identity or approval. Changes are versioned and linked to affected units.

## Numbers, dates, and initials

Normalisation preserves both the display value and the exact intended spoken value. Compare the generated speech/retranscription against the intended `spoken_text`, including currency, decimal/percentage semantics, dates, acronyms, and negation—not merely character similarity. Ambiguous forms require human confirmation.

## SSML boundary

Pronunciation substitutions flow through the application-controlled safe SSML builder. Claude output is data, never raw XML. The builder escapes text, permits only its verified subset, enforces bounds, rejects unknown tags/namespaces/external resources, validates the final XML, and records its hash. Azure TTS remains the pronunciation authority; even an entry with valid review evidence is a hint that still needs listening validation.

Fluent isiZulu review of `faithful_translation`, `spoken_text`, Azure audio, and OpenVoice audio remains mandatory.
