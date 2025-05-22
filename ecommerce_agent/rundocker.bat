@echo off
rem --------------------------------------------------------
rem  Batch script to run the ADK e-commerce agent in Docker
rem --------------------------------------------------------

rem Path to your Google ADC JSON
set "ADC_PATH=C:\Users\user\AppData\Roaming\gcloud\legacy_credentials\yourmail@gmail.com\adc.json"

rem Image and container names
set "IMAGE_NAME=ecommerse-agents"
set "CONTAINER_NAME=ecommerse-agents"

rem Run the container
docker run --rm ^
  -p 8080:8080 ^
  --name %CONTAINER_NAME% ^
  -v "%ADC_PATH%":/secrets/adc.json ^
  -e GOOGLE_APPLICATION_CREDENTIALS=/secrets/adc.json ^
  %IMAGE_NAME%
