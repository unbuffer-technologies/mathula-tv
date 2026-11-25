# Deploy the speech-to-text + diarization service to Cloud Run with an NVIDIA L4 (same project, region and shape as the OmniVoice service).
# Run from C:\mathula-tv after:  gcloud config configurations activate mathula
# One-time: pyannote's pipeline is gated.  Accept its licence on huggingface.co (pyannote/speaker-diarization-community-1) with the account whose token is used, then store the token:
#   hf auth token | gcloud secrets create hf-token --project mathula-ai --data-file=-          (first time; use "versions add" afterwards)
#   gcloud secrets add-iam-policy-binding hf-token --project mathula-ai --member serviceAccount:<the Cloud Run runtime service account> --role roles/secretmanager.secretAccessor
$Project = "mathula-ai"
$Region = "europe-west1"

gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com secretmanager.googleapis.com --project $Project

gcloud run deploy mathula-stt `
  --project $Project --region $Region `
  --source gcp/stt_service `
  --gpu 1 --gpu-type nvidia-l4 --no-gpu-zonal-redundancy `
  --cpu 4 --memory 16Gi --no-cpu-throttling `
  --min-instances 0 --max-instances 1 --concurrency 2 `
  --timeout 3600 `
  --set-secrets HF_TOKEN=hf-token:latest `
  --no-allow-unauthenticated
