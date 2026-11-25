# One-shot deploy of the STT + diarization service (run from C:\mathula-tv with the "mathula" gcloud configuration active).
#  1. the Hugging Face token (the local `hf` login) -> Secret Manager secret "hf-token", readable by the Cloud Run runtime account; the token is piped, never printed
#  2. gcloud run deploy mathula-stt (L4, scale to zero, private)
#  3. the accounts allowed to call omnivoice are allowed to call mathula-stt too
$ErrorActionPreference = "Continue"
$Project = "mathula-ai"; $Region = "europe-west1"
gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com secretmanager.googleapis.com --project $Project

$exists = (gcloud secrets list --project $Project --filter "name~hf-token" --format "value(name)") -join ""
if (-not $exists) {
    hf auth token | gcloud secrets create hf-token --project $Project --replication-policy automatic --data-file=-
    Write-Output "secret hf-token created"
} else { Write-Output "secret hf-token already exists" }
$number = (gcloud projects describe $Project --format "value(projectNumber)")
$runtime = "$number-compute@developer.gserviceaccount.com"
gcloud secrets add-iam-policy-binding hf-token --project $Project --member "serviceAccount:$runtime" --role roles/secretmanager.secretAccessor | Out-Null
Write-Output "runtime account $runtime may read the secret"

gcloud run deploy mathula-stt --project $Project --region $Region --source gcp/stt_service --gpu 1 --gpu-type nvidia-l4 --no-gpu-zonal-redundancy --cpu 4 --memory 16Gi --no-cpu-throttling --min-instances 0 --max-instances 1 --concurrency 2 --timeout 3600 --set-secrets HF_TOKEN=hf-token:latest --no-allow-unauthenticated
Write-Output "deploy exit $LASTEXITCODE"

$policy = gcloud run services get-iam-policy omnivoice --project $Project --region $Region --format json | ConvertFrom-Json
foreach ($binding in $policy.bindings) {
    if ($binding.role -eq "roles/run.invoker") {
        foreach ($member in $binding.members) {
            gcloud run services add-iam-policy-binding mathula-stt --project $Project --region $Region --member $member --role roles/run.invoker | Out-Null
            Write-Output "invoker: $member"
        }
    }
}
gcloud run services describe mathula-stt --project $Project --region $Region --format "value(status.url)"
