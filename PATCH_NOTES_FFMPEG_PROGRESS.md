# Native FFmpeg telemetry restoration

The native dubbing path had replaced real FFmpeg machine progress with generic elapsed-time heartbeats. This overlay restores the established Mathula progress mechanism for FFmpeg stages.

- Reuses `mathula_tv.rendering.run_ffmpeg_with_progress` for the native master.
- Preserves stream-copy video first; H.264 fallback remains available when MP4 remux fails.
- Wires `render_tiktok_hook_edit(... progress_callback=...)` so publication rendering exposes real FFmpeg progress.
- Keeps heartbeat telemetry for Azure TTS and other blocking calls without measurable progress.
