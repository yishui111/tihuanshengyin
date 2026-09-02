@echo off
setlocal enabledelayedexpansion
REM ============================================================
REM  Voice Swap Workbench - one launcher for everything
REM  Usage:
REM    start.bat           -> start Hub + engine A only (RVC,
REM                           the engine trained models go to)
REM    start.bat A C       -> start Hub + engines A and C
REM    start.bat A B C D   -> start Hub + ALL engines
REM    start.bat stop      -> stop ALL services
REM    start.bat stop B    -> stop only engine B
REM    start.bat help      -> show this usage
REM  Web  : http://127.0.0.1:8000/   (Hub / batch voice swap)
REM  Ports: Hub=8000  A=8010  B=8020  C=8030  D=8040
REM  Env overrides: PY312_PYTHON / PY310_PYTHON (python.exe paths)
REM                 HUB_PORT / API_PORT (per service, default above)
REM ============================================================
set "ROOT=%~dp0"
set "PATH=%ROOT%runtime\ffmpeg\bin;%PATH%"

if /i "%~1"=="stop" goto :stop
if /i "%~1"=="help" goto :help

REM ---- resolve python interpreters ----
set "PY312=%PY312_PYTHON%"
if not defined PY312 if exist "%ROOT%runtime\py312\python.exe" set "PY312=%ROOT%runtime\py312\python.exe"
if not defined PY312 if exist "%ROOT%runtime\py312\Scripts\python.exe" set "PY312=%ROOT%runtime\py312\Scripts\python.exe"
set "PY310=%PY310_PYTHON%"
if not defined PY310 if exist "%ROOT%runtime\py310\python.exe" set "PY310=%ROOT%runtime\py310\python.exe"
if not defined PY310 if exist "%ROOT%runtime\py310\Scripts\python.exe" set "PY310=%ROOT%runtime\py310\Scripts\python.exe"
if not defined PY312 (
  echo [ERROR] Python 3.12 not found. Set PY312_PYTHON to your python.exe
  echo         or follow DEPLOY.md "step 3" to create runtime\py312.
  exit /b 1
)
if not defined PY310 (
  echo [ERROR] Python 3.10 not found. Set PY310_PYTHON to your python.exe
  echo         or follow DEPLOY.md "step 3" to create runtime\py310.
  exit /b 1
)

REM ================= START =================
REM Default: Hub + engine A only (RVC, where trained models go).
REM Pass engine letters to add more: start.bat A C
set "WANT=A"
if not "%~1"=="" set "WANT="
for %%S in (%*) do set "WANT=!WANT!%%S"

echo ============================================
echo   Voice Swap Workbench launcher
echo   Hub:    http://127.0.0.1:8000/
echo   Engines: A=8010 B=8020 C=8030 D=8040
echo   Starting: %WANT%   (skip if already running)
echo   Default: Hub + A only. Use e.g. "start.bat A C" to add engines.
echo ============================================

call :start Hub 8000 "%PY312%" "%ROOT%hub\server.py"
if not "%WANT:A=%"=="%WANT%" call :start A 8010 "%PY312%" "%ROOT%rvc_service\rvc_character_api.py"
if not "%WANT:B=%"=="%WANT%" call :start B 8020 "%PY310%" "%ROOT%openvoice_service\openvoice_clone_api.py"
if not "%WANT:C=%"=="%WANT%" call :start C 8030 "%PY312%" "%ROOT%sovits_service\sovits_cn_api.py"
if not "%WANT:D=%"=="%WANT%" call :start D 8040 "%PY312%" "%ROOT%gptsovits_service\gptsovits_cn_api.py"

echo.
echo Done. Opening Hub page (first model load takes 10-30s)...
timeout /t 3 >nul 2>nul || ping -n 4 127.0.0.1 >nul
start "" "http://127.0.0.1:8000/"
timeout /t 8 >nul 2>nul || ping -n 9 127.0.0.1 >nul
exit /b 0

REM ---- start one service if its port is free ----
:start
set "SVC=%~1"
set "PORT=%~2"
set "PY=%~3"
set "SCRIPT=%~4"
if "%SVC%"=="A" set "RVC_ROOT=%ROOT%rvc"
if "%SVC%"=="B" set "OV_WATERMARK=0"
if "%SVC%"=="B" set "TORCH_HOME=%ROOT%runtime\cache\torch"
set "API_PORT=%PORT%"
powershell -NoProfile -Command "if (Get-NetTCPConnection -LocalPort %PORT% -State Listen -ErrorAction SilentlyContinue) { exit 1 } else { exit 0 }"
if errorlevel 1 goto :already
start "Svc-%SVC%-%PORT%" /min "%PY%" "%SCRIPT%"
echo [start] %SVC% -> http://127.0.0.1:%PORT%/
exit /b 0
:already
echo [skip] %SVC% already running on port %PORT%
exit /b 0

REM ================= STOP =================
:stop
if "%~2"=="" (
  echo Stopping ALL services: 8000/8010/8020/8030/8040...
  powershell -NoProfile -Command "$ports=8000,8010,8020,8030,8040; foreach($p in $ports){ $c=Get-NetTCPConnection -LocalPort $p -State Listen -ErrorAction SilentlyContinue; if($c){ $c|ForEach-Object{Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue}; Write-Output ('port '+$p+' -> stopped') } else { Write-Output ('port '+$p+' -> not running') } }"
) else (
  for %%S in (%*) do if /i not "%%S"=="stop" call :stop_one %%S
)
echo.
exit /b 0

REM ---- stop one service ----
:stop_one
set "SPORT="
if /i "%~1"=="Hub" set "SPORT=8000"
if /i "%~1"=="A" set "SPORT=8010"
if /i "%~1"=="B" set "SPORT=8020"
if /i "%~1"=="C" set "SPORT=8030"
if /i "%~1"=="D" set "SPORT=8040"
if not defined SPORT (
  echo Unknown service: %~1   ^(use A B C D or Hub^)
  exit /b 0
)
powershell -NoProfile -Command "$c=Get-NetTCPConnection -LocalPort %SPORT% -State Listen -ErrorAction SilentlyContinue; if($c){ $c|ForEach-Object{Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue}; Write-Output ('%~1 -> stopped') } else { Write-Output ('%~1 -> not running') }"
exit /b 0

REM ================= HELP =================
:help
echo.
echo   start.bat           start Hub + engine A (RVC, default)
echo   start.bat A C       start Hub + engines A and C
echo   start.bat A B C D   start Hub + ALL engines
echo   start.bat stop      stop ALL services
echo   start.bat stop B    stop only engine B
echo   start.bat help      show this help
echo.
echo   Engines: A=RVC(8010, trained models)  B=OpenVoice(8020, clone)
echo            C=SoVITS(8030, built-in CN chars)  D=GPT-SoVITS(8040, re-say)
echo   Hub:    8000   http://127.0.0.1:8000/
echo.
exit /b 0
