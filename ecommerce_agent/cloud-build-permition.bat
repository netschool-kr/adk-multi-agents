@echo off
REM ================================
REM Grant Cloud Build service account the necessary IAM roles
REM ================================

REM 1) Specify your GCP project ID here
set PROJECT_ID=adk-multi-agents

REM 2) Fetch the project number automatically
for /f "delims=" %%N in ('gcloud projects describe %PROJECT_ID% --format="value(projectNumber)"') do set PROJECT_NUMBER=%%N

echo Project ID: %PROJECT_ID%
echo Project Number: %PROJECT_NUMBER%

REM 3) Assign the Cloud Run Admin role to the Cloud Build service account
gcloud projects add-iam-policy-binding %PROJECT_ID% ^
  --member="serviceAccount:%PROJECT_NUMBER%@cloudbuild.gserviceaccount.com" ^
  --role="roles/run.admin"

REM 4) Allow the Cloud Build service account to impersonate your runtime service account
gcloud iam service-accounts add-iam-policy-binding ^
  ecommerce-agent-sa@%PROJECT_ID%.iam.gserviceaccount.com ^
  --member="serviceAccount:%PROJECT_NUMBER%@cloudbuild.gserviceaccount.com" ^
  --role="roles/iam.serviceAccountUser" ^
  --project=%PROJECT_ID%

echo.
echo Permissions have been granted successfully.
pause
