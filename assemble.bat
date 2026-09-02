@echo off
rem ==================================================
rem  TIHUANSHENGYIN - one-key preflight for big assets
rem  Guarantee flow: (A) copy original project folder
rem  with big assets (fastest), or (B) clone this repo
rem  then run this script; details: see DEPLOY.md top.
rem ==================================================
setlocal
cd /d "%~dp0"
set "MISSING=0"
echo Checking required big assets...
if exist "rvc" (echo   OK   rvc) else (echo   MISS rvc ^& set MISSING=1)
if exist "sovits_service\so-vits-svc-4.1-Stable" (echo   OK   sovits_service\so-vits-svc-4.1-Stable) else (echo   MISS sovits_service\so-vits-svc-4.1-Stable ^& set MISSING=1)
if exist "gptsovits_service\GPT-SoVITS" (echo   OK   gptsovits_service\GPT-SoVITS) else (echo   MISS gptsovits_service\GPT-SoVITS ^& set MISSING=1)
if exist "openvoice_service\checkpoints_v2" (echo   OK   openvoice_service\checkpoints_v2) else (echo   MISS openvoice_service\checkpoints_v2 ^& set MISSING=1)
if exist "runtime" (echo   OK   runtime) else (echo   MISS runtime ^& set MISSING=1)
echo.
if %MISSING%==0 (
  echo ALL big assets present. Run start.bat now.
) else (
  echo Some big assets missing. See DEPLOY.md (top section
  "Deployment guarantee") for download instructions.
)
pause
