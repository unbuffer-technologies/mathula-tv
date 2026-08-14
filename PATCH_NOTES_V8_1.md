# Mathula TV v8.1 multivariant timing adapter

Fixes automatic multivariant translation request generation for production
`dubbing/dubbing_units.json` artifacts, whose canonical timing fields are
`start_ms`, `preferred_end_ms`, `hard_end_ms`, `preferred_duration_ms`, and
`maximum_duration_ms` rather than transcript-style `end_ms`.

The adapter canonicalizes the collision-safe hard window into `end_ms` and
`available_duration_ms` for the translation request. The direct Azure loader
accepts the same timing aliases defensively. Transcript-style inputs remain
fully compatible.
