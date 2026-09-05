@echo on
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
REM  If a default port is taken by ANOTHER program, the service
REM  falls back to the next port (8001, 8002, ...). Chosen ports
REM  are saved to last_run_ports.txt; stop.bat frees exactly ours.
REM  Env overrides: PY312_PYTHON / PY310_PYTHON (python.exe paths)
REM                 HUB_PORT / A_PORT / B_PORT / C_PORT / D_PORT
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
set "WANT=A"
if not "%~1"=="" set "WANT="
for %%S in (%*) do set "WANT=!WANT!%%S"

echo ============================================
echo   Voice Swap Workbench launcher
echo   Default ports: Hub=8000  A=8010  B=8020  C=8030  D=8040
echo   Ports taken by other programs are skipped automatically.
echo ============================================

call :start HUB_PORT 8000 "%PY312%" "%ROOT%hub\server.py" hub
if not "%WANT:A=%"=="%WANT%" call :start A_PORT 8010 "%PY312%" "%ROOT%rvc_service\rvc_character_api.py" rvc-character
if not "%WANT:B=%"=="%WANT%" call :start B_PORT 8020 "%PY310%" "%ROOT%openvoice_service\openvoice_clone_api.py" openvoice-clone
if not "%WANT:C=%"=="%WANT%" call :start C_PORT 8030 "%PY312%" "%ROOT%sovits_service\sovits_cn_api.py" sovits-cn
if not "%WANT:D=%"=="%WANT%" call :start D_PORT 8040 "%PY312%" "%ROOT%gptsovits_service\gptsovits_cn_api.py" gptsovits-cn

echo.
REM ---- remember chosen ports so stop.bat frees exactly ours ----
> "%ROOT%last_run_ports.txt" echo %HUB_PORT%
>> "%ROOT%last_run_ports.txt" echo %A_PORT%
>> "%ROOT%last_run_ports.txt" echo %B_PORT%
>> "%ROOT%last_run_ports.txt" echo %C_PORT%
>> "%ROOT%last_run_ports.txt" echo %D_PORT%
echo Done. Opening Hub page (first model load takes 10-30s)...
timeout /t 3 >nul 2>nul || ping -n 4 127.0.0.1 >nul
start "" "http://127.0.0.1:%HUB_PORT%/"
timeout /t 8 >nul 2>nul || ping -n 9 127.0.0.1 >nul
exit /b 0

REM ---- start one service.
REM  %1 = port env var (HUB_PORT/A_PORT/...), %2 = default port,
REM  %3 = python, %4 = script, %5 = /health signature of OUR service.
REM  A port is usable only if we can BIND 0.0.0.0:port (same bind as
REM  the servers do). Connect-probes lie on TUN-proxy machines, and
REM  Windows lets a loopback-specific bind succeed even when
REM  0.0.0.0:port belongs to another program.
REM  If the port answers /health as OUR service, it is already
REM  running -> skip. Retry up to 3 rounds against flapping ports.
:start
set "PORTVAR=%~1"
set "DEFPORT=%~2"
set "PY=%~3"
set "SCRIPT=%~4"
set "SVCNAME=%~5"
set "ROUNDS=0"
:start_try
set /a ROUNDS+=1
if %ROUNDS% gtr 3 (
  echo [ERROR] no usable port found for %SVCNAME%. Not started.
  exit /b 1
)
set "PORT="
set "PICKMODE="
set "SVCSUB=%SVCNAME%"
for /f "tokens=1,2" %%A in ('powershell -NoProfile -Command "$want=%DEFPORT%; $svc='%SVCSUB%'; for($i=0; $i -lt 6; $i++){ $p=$want+$i; $l=$null; try { $l = [System.Net.Sockets.TcpListener]::new([System.Net.IPAddress]::Any, $p); $l.Start() } catch { $l=$null }; if ($l) { $l.Stop(); Write-Output (''+$p+' free'); exit }; $body=$null; try { $req=[System.Net.HttpWebRequest]::Create(('http://127.0.0.1:'+${p}+'/health')); $req.Proxy = New-Object System.Net.WebProxy($null); $req.Timeout = 2000; $res = $req.GetResponse(); $body = (New-Object IO.StreamReader($res.GetResponseStream())).ReadToEnd(); $res.Close() } catch {}; if ($body -and $body -like ('*'+$svc+'*')){ Write-Output (''+$p+' ours'); exit } }"') do (
  set "PORT=%%A"
  set "PICKMODE=%%B"
)
if not defined PORT (
  echo [ERROR] no usable port found for %SVCNAME%. Not started.
  exit /b 1
)
set "API_PORT=%PORT%"
if "%SVCNAME%"=="hub" set "HUB_PORT=%PORT%"
if "%SVCNAME%"=="rvc-character" set "RVC_ROOT=%ROOT%rvc"
if "%SVCNAME%"=="openvoice-clone" set "OV_WATERMARK=0"
if "%SVCNAME%"=="openvoice-clone" set "TORCH_HOME=%ROOT%runtime\cache\torch"
if "%PICKMODE%"=="ours" (
  echo [skip] %SVCNAME% already running on port %PORT%
  exit /b 0
)
REM flap safety: re-verify the port is still free right before start
powershell -NoProfile -Command "$l = $null; try { $l = [System.Net.Sockets.TcpListener]::new([System.Net.IPAddress]::Any, %PORT%); $l.Start() } catch { exit 1 }; if ($l) { $l.Stop() }; exit 0"
if errorlevel 1 (
  echo [INFO] port %PORT% just got taken. Retrying...
  goto :start_try
)
start "Svc-%SVCNAME%-%PORT%" /min "%PY%" "%SCRIPT%"
echo [start] %SVCNAME% -> http://127.0.0.1:%PORT%/
exit /b 0

REM ================= STOP =================
:stop
if "%~2"=="" (
  echo Stopping voice services. Other programs are not touched.
  call :stop_all
) else (
  for %%S in (%*) do if /i not "%%S"=="stop" call :stop_one %%S
)
echo.
exit /b 0

:stop_all
powershell -NoProfile -Command "$ports = @(); if (Test-Path ('%ROOT%last_run_ports.txt')) { $ports += Get-Content ('%ROOT%last_run_ports.txt') | ForEach-Object { [int]$_ } }; $ports += 8000,8010,8020,8030,8040; $ports = $ports | Select-Object -Unique; $names = '*hub*','*rvc-character*','*openvoice-clone*','*sovits-cn*','*gptsovits-cn*'; foreach ($p in $ports) { $c = Get-NetTCPConnection -LocalPort $p -State Listen -ErrorAction SilentlyContinue; if (-not $c) { continue }; $ours = $false; $body=$null; try { $req = [System.Net.HttpWebRequest]::Create('http://127.0.0.1:'+${p}+'/health'); $req.Proxy = New-Object System.Net.WebProxy($null); $req.Timeout = 2000; $res = $req.GetResponse(); $body = (New-Object IO.StreamReader($res.GetResponseStream())).ReadToEnd(); $res.Close(); foreach ($n in $names) { if ($body -like $n) { $ours = $true } } } catch {}; if ($ours) { $c | ForEach-Object { Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue }; Write-Output ('port ' + $p + ' -> stopped') } else { Write-Output ('port ' + $p + ' -> other program, not touched') } }"
exit /b 0

REM ---- stop one service (by letter): find OUR listener on default
REM      or fallback ports by health signature, then kill it ----
:stop_one
set "SVC=%~1"
set "SVCNAME=%SVC%"
set "SPORT=0"
if /i "%SVC%"=="Hub" set "SVCNAME=hub"
if /i "%SVC%"=="A" set "SVCNAME=rvc-character"
if /i "%SVC%"=="B" set "SVCNAME=openvoice-clone"
if /i "%SVC%"=="C" set "SVCNAME=sovits-cn"
if /i "%SVC%"=="D" set "SVCNAME=gptsovits-cn"
powershell -NoProfile -Command "$ports=@(%SPORT%); if (Test-Path ('%ROOT%last_run_ports.txt')) { $ports += Get-Content ('%ROOT%last_run_ports.txt') | ForEach-Object { [int]$_ } }; $ports = $ports | Select-Object -Unique; $done=$false; foreach ($p in $ports) { $c = Get-NetTCPConnection -LocalPort $p -State Listen -ErrorAction SilentlyContinue; if (-not $c) { continue }; $body=$null; try { $req = [System.Net.HttpWebRequest]::Create('http://127.0.0.1:'+${p}+'/health'); $req.Proxy = New-Object System.Net.WebProxy($null); $req.Timeout = 2000; $res = $req.GetResponse(); $body = (New-Object IO.StreamReader($res.GetResponseStream())).ReadToEnd(); $res.Close(); if ($body -like ('*%SVCNAME%*')) { $c | ForEach-Object { Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue }; Write-Output ('%SVC% -> stopped (port ' + $p + ')'); $done=$true; break } } catch {} }; if (-not $done) { Write-Output ('%SVC% -> not running') }"
exit /b 0

REM ================= HELP =================
:help
echo.
echo   start.bat           start Hub + engine A (RVC, default)
echo   start.bat A C       start Hub + engines A and C
echo   start.bat A B C D   start Hub + ALL engines
echo   start.bat stop      stop ALL services (ours only)
echo   start.bat stop B    stop only engine B
echo   start.bat help      show this help
echo.
echo   Ports auto-fall-back when another program takes the default
echo   (e.g. Hub 8000 -> 8001). stop.bat only stops OUR services.
echo   Engines: A=RVC(8010, trained models)  B=OpenVoice(8020, clone)
echo            C=SoVITS(8030, built-in CN chars)  D=GPT-SoVITS(8040, re-say)
echo.
exit /b 0
