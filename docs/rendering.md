# Subtitles and production rendering

Rendering creates a review-ready artifact after validated alignment, background, and mix stages. It never uploads to YouTube and never treats rendering success as publication approval.

## Subtitles

Produce:

```text
working/jobs/{job_id}/subtitles/spoken_zu.json
working/jobs/{job_id}/subtitles/subtitles_zu.srt
working/jobs/{job_id}/subtitles/subtitles_zu.vtt
```

Subtitle text is exactly approved `spoken_text`, never pronunciation-respelled `tts_text`. Timings use final aligned unit boundaries and JSON retains speaker attribution, dubbing-unit ID, overlap metadata, and schema version. Source phrase/word provenance remains in the dubbing plan linked by unit ID; it is not duplicated into each subtitle entry. Burned-in subtitles are optional for review; the final master need not burn them in.

## Final audio and MP4

The final mix contains aligned converted dialogue and the selected background candidate, with genuine overlaps preserved and residual-QC status retained for review. It is rendered as compatible AAC at 48 kHz. The renderer:

- preserves the source video stream by stream copy when safe;
- re-encodes video only when required (for example burned subtitles or incompatible source);
- preserves source frame rate, resolution, and duration;
- pads safe trailing audio silence where needed, but never truncates final speech;
- creates neither trailing black frames nor an audio/video duration mismatch;
- validates input/output streams and duration with ffprobe;
- writes atomically, calculates SHA-256, and finalizes the local output only after validation.

`render` writes:

```text
working/jobs/{job_id}/render/final_dubbed.mp4
working/jobs/{job_id}/render/render_manifest.json
```

The subsequent `review` command writes:

```text
working/jobs/{job_id}/review/qc_report.json
working/jobs/{job_id}/review/qc_report.md
```

The render manifest records source video/final-audio/output identities, stream-copy or re-encode decision, codecs, sample rate, source/output duration, resolution/frame rate, subtitle mode, hashes, and `youtube_uploaded: false`.

If final audio exceeds the video beyond tolerance, rendering fails rather than truncating speech. Repair alignment/translation; do not shorten media destructively or append black video.
