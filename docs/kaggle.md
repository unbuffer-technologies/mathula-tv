# OpenVoice Kaggle GPU worker

Kaggle uses `notebooks/mathula_tv_openvoice_kaggle_worker.ipynb`. It supports the same separate `assets` and `conversion` modes, plan hashes, lease tasks, manifests, validation, and promotion rules as the [Colab worker](colab.md). Never run two modes/workers concurrently for one job.

## Setup

Create a private Kaggle notebook, enable a compatible GPU, and enable network only when reviewed checkout/dependency/checkpoint materialisation requires it. Configure the exact job/worker IDs, GCS/project/prefix, public repository URLs, 40-character repository/OpenVoice commits, mode-specific plan SHA-256, Python/Torch versions, zero-argument runtime factory, optional candidate-evaluator factory, `==` dependency JSON, checkpoint JSON/hashes, precision, limits, and live gates listed in [the Colab contract](colab.md#runtime-contract).

Store service-account JSON only in Kaggle Secrets under the exact secret name `MATHULA_TV_GCP_SERVICE_ACCOUNT_JSON`. Never put its value in `.env`, a Dataset, code cell, notebook output, or committed file. The live gates are:

```text
MATHULA_TV_LIVE_OPERATION=1
MATHULA_TV_LIVE_OPERATION_ACK=I_ACKNOWLEDGE_LIVE_OPENVOICE_GCS_MUTATION
```

For the protected proof, `MATHULA_TV_JOB_ID` is exactly `19ba6d69f1b84132ba4f20599101834a`.

## Two passes

Run `assets` mode only after the server promotes the self-verifying asset plan. It claims `openvoice_asset_preparation`, creates or reuses a hash-validated source-embedding cache, creates job-local target embeddings, converts the planned Azure voice candidates, writes embedding/candidate manifests, promotes all verified outputs, and prints the asset-reconciliation command.

After server reconciliation/voice selection (using reviewed `--evaluations FILE` if candidate metrics remain pending) and full plan queueing, restart a clean runtime in `conversion` mode with the conversion-plan SHA-256 and a fresh explicit live acknowledgement. It claims `openvoice_conversion`, validates all selected inputs, converts only pending turns with one model load, promotes verified WAVs/final manifest, completes the lease, and prints conversion reconciliation.

Run all 23 numbered stages; each stage branches safely by selected mode as documented in [Colab](colab.md#run-all-23-stages-in-order). GPU completion is not compatibility approval.

## Privacy and durability

Kaggle working files/output are not authoritative. GCS promoted objects/generations/hashes and the atomic job manifest remain durable truth. Never print secrets, signed URLs, full transcripts, speaker audio/embeddings, or private config. Keep the notebook private, clear unsafe output, delete local private assets, and stop the session.

Back on the server, run `reconcile-openvoice-assets JOB_ID --live-operation` after assets mode or `reconcile-voice-conversion JOB_ID --live-operation` after conversion, then continue [the guarded runbook](cli.md).
