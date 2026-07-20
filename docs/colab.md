# OpenVoice Colab GPU worker

Use a fresh GPU runtime and `notebooks/mathula_tv_openvoice_colab_worker.ipynb`. Business rules remain in importable Python modules; the notebook is a thin two-mode operator surface:

- `assets` creates or reuses hash-validated Azure source embeddings, creates job-local target embeddings, converts per-speaker Azure voice candidates, and promotes an asset manifest;
- `conversion` consumes the selected, hash-complete assets and converts queued dubbing turns.

Each mode has a separate server plan/hash, lease task, manifest, and explicit live acknowledgement. The compatibility gate remains pending; completing either mode is evidence, not a production pass.

## Runtime contract

Configure runtime environment/secrets without printing values:

```text
MATHULA_TV_JOB_ID=19ba6d69f1b84132ba4f20599101834a
MATHULA_TV_WORKER_ID=colab-UNIQUE_SAFE_ID
MATHULA_TV_GCP_PROJECT=PROJECT_ID
MATHULA_TV_GCS_BUCKET=PRIVATE_BUCKET
MATHULA_TV_GCS_PREFIX=mathula-tv
MATHULA_TV_REPOSITORY_URL=https://PUBLIC_REPOSITORY_URL
MATHULA_TV_REPOSITORY_COMMIT=EXACT_40_CHARACTER_COMMIT
MATHULA_TV_OPENVOICE_REPOSITORY_URL=https://PUBLIC_OPENVOICE_REPOSITORY_URL
MATHULA_TV_OPENVOICE_REVISION=EXACT_40_CHARACTER_COMMIT
MATHULA_TV_OPENVOICE_WORKER_MODE=assets
MATHULA_TV_OPENVOICE_ASSET_PLAN_SHA256=EXACT_64_CHARACTER_SHA256
# Conversion mode instead uses MATHULA_TV_OPENVOICE_PLAN_SHA256.
MATHULA_TV_OPENVOICE_PYTHON_VERSION=EXACT_VERSION
MATHULA_TV_OPENVOICE_TORCH_VERSION=EXACT_VERSION
MATHULA_TV_OPENVOICE_RUNTIME_FACTORY=module:function
# Optional assets-mode measured QC:
MATHULA_TV_OPENVOICE_CANDIDATE_EVALUATOR_FACTORY=module:function
MATHULA_TV_OPENVOICE_REQUIREMENTS_JSON=["package==exact-version"]
MATHULA_TV_OPENVOICE_CHECKPOINTS_JSON=[{"name":"name","filename":"file","source_path":"private-materialised-path","sha256":"64-hex"}]
MATHULA_TV_OPENVOICE_PRECISION=fp16
MATHULA_TV_LIVE_OPERATION=1
MATHULA_TV_LIVE_OPERATION_ACK=I_ACKNOWLEDGE_LIVE_OPENVOICE_GCS_MUTATION
```

Repository URLs must be public HTTPS URLs without credentials/query strings. Both revisions are exact 40-character commits; dependencies use exact `==` pins; every checkpoint has name/filename/source path/SHA-256. Checkpoint names/SHA-256 values must exactly match the identities configured on the normal server before queueing; only the private worker supplies materialisation paths. The runtime factory is a safe zero-argument `module:function`; the optional evaluator factory returns a callable. Obtain the applicable plan hash from the verified server object. Set the live gate only immediately before the chosen lease claim and do not bake it into the saved notebook.

Colab uses interactive Google authentication. Grant only required private GCS object access. Do not install Pyannote, F5-TTS, or OmniVoice into this worker.

## Assets mode

Before assets mode, the server must have generated reviewed Azure calibration WAVs, same-speaker reels, the identical Azure calibration sentence for every speaker/allowed voice, and `dubbing/openvoice/asset_plan.json`. Configure `WORKER_MODE=assets` and its exact file SHA-256.

The asset lease task is `openvoice_asset_preparation`. Planned inputs include Azure calibration WAV/manifests, speaker reel/manifests, and Azure voice-candidate WAVs. Planned outputs include source/target embeddings and their manifests, converted candidate WAVs, each `speakers/{speaker}/azure_voice_candidates.json`, and `dubbing/openvoice/asset_manifest.json`.

The worker always records measured conversion duration/drift and warnings. The optional evaluator adds speaker similarity, retranscription/word-error approximation, and an artefact count. Without it, candidate QC remains `pending_external_qc` and human review is still required. After verified promotion/completion, return to the server with `python -m mathula_tv.cli reconcile-openvoice-assets JOB_ID --live-operation`, then persist per-speaker voice selection with reviewed `--evaluations FILE` when needed before building/queueing the conversion plan. See the exact schema and constraints in [OpenVoice](openvoice.md#per-speaker-azure-voice-calibration).

## Conversion mode

After the server reconciles assets, persists voice selections, resynthesizes affected Azure units if selection changed, builds the full relative plan, and queues it, configure `WORKER_MODE=conversion` plus `MATHULA_TV_OPENVOICE_PLAN_SHA256`.

The `openvoice_conversion` lease verifies the selected source/target embeddings, Azure turns, same-speaker references, and full plan, loads one model, converts only pending units, promotes canonical WAVs/final manifest, and completes. Return with `python -m mathula_tv.cli reconcile-voice-conversion JOB_ID --live-operation`. Never run assets and conversion modes concurrently for one job.

## Run all 23 stages in order

1. Runtime inspection and selected mode
2. Python and CUDA compatibility
3. Repository checkout at the exact commit
4. Exact dependency installation
5. OpenVoice revision pinning
6. Checkpoint materialisation
7. Checkpoint hash verification
8. Authentication
9. Private GCS connectivity
10. Exact eligible-job discovery
11. Mode-specific generation-precondition lease claim
12. Applicable plan download/hash verification
13. Azure calibration inputs (`assets`) or source audio/embedding download (`conversion`)
14. Speaker reels/candidate inputs (`assets`) or references/target embeddings (`conversion`)
15. All input-hash and WAV validation
16. One model load
17. Embedding/candidate work (`assets`) or pending-turn conversion (`conversion`)
18. Output validation
19. GCS partial upload and verified promotion
20. Mode-specific final manifest creation/promotion/verification
21. Lease completion
22. Exact server-reconciliation instruction
23. Local sensitive-file cleanup

Do not skip a failed pin/hash/runtime/lease stage. Restart only through the resumable manifest and a valid current lease.

## Safe output and return

The notebook may print mode, job/lease identity and expiry, GPU, CUDA/Python/Torch/OpenVoice versions, checkpoint hashes, speaker/unit/asset counts, current asset/unit, selected Azure voice, source/converted duration and ratio, output SHA-256, promotion result, manifest path, and next reconciliation command.

It must not print credentials, keys, tokens, authorization headers, signed URLs, complete private configuration, embeddings/audio, or unnecessary transcripts. Clear unsafe cell output, delete local private assets after verified promotion, and end the runtime.

Follow the exact two-pass server/notebook sequence in [cli.md](cli.md). One explicitly unguarded Phase A alignment may create listening evidence; normal production alignment/rendering requires a calibrated human-reviewed pass.
