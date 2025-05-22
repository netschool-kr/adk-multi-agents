gcloud run deploy ecommerce-agent-api ^
    --image us-central1-docker.pkg.dev/adk-multi-agents/agent-repo/agent-app:latest ^
    --platform managed --region us-central1 ^
    --service-account ecommerce-agent-sa@adk-multi-agents.iam.gserviceaccount.com ^
    --set-env-vars GOOGLE_CLOUD_PROJECT=adk-multi-agents,GOOGLE_CLOUD_LOCATION=us-central1 ^
    --allow-unauthenticated
