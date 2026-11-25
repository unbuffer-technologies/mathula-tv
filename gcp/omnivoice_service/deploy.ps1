# Deploy the OmniVoice Zulu service to Cloud Run with an NVIDIA L4.
# Run from C:\mathula-tv after:  gcloud config configurations activate mathula   (signed in to the right account, project set)
# Needs a gcloud new enough for --gpu (the 2023 release 421 is not):  gcloud components update
$Project = "mathula-ai"             # the project that received L4 quota (the earlier project-bd95e7fe-76cc-46ed-8f1 was refused for lack of usage history)
$Region = "europe-west1"            # where Google approved L4 quota for project mathula-ai (the effective quota is now 10 L4 GPUs without zonal redundancy). Other L4 regions: us-central1, us-east4, europe-west4, asia-southeast1

gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com --project $Project

# One GPU per instance, up to four instances (the project's L4 quota in europe-west1 is 10), one request per instance so that requests sent together run on separate GPUs (the app serialises generations with a lock, so a higher concurrency only queues them on one GPU), scaled to zero when idle (billed per second while an instance is up). 4 CPU / 16 GiB is the L4 minimum.
gcloud run deploy omnivoice `
  --project $Project --region $Region `
  --source gcp/omnivoice_service `
  --gpu 1 --gpu-type nvidia-l4 --no-gpu-zonal-redundancy `
  --cpu 4 --memory 16Gi --no-cpu-throttling `
  --min-instances 0 --max-instances 4 --concurrency 1 `
  --timeout 3600 `
  --no-allow-unauthenticated
