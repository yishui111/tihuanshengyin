@echo off
setlocal
REM ============================================================
REM  Stop ALL voice services - kills ONLY listeners whose owning
REM  process command line belongs to THIS project's scripts;
REM  ports used by other programs are left alone. Honors fallback
REM  ports recorded by start.bat in last_run_ports.txt.
REM ============================================================
echo Stopping voice services ...
powershell -NoProfile -Command "$root='%~dp0'; $ports = @(); if (Test-Path ($root + 'last_run_ports.txt')) { $ports += Get-Content ($root + 'last_run_ports.txt') | Where-Object { $_ -match '^[0-9]+$' } | ForEach-Object { [int]$_ } }; $ports += 8000,8010,8011,8020,8030,8040; $ports = $ports | Select-Object -Unique; $markers = '*hub*server.py*','*rvc_character_api.py*','*openvoice_clone_api.py*','*sovits_cn_api.py*','*gptsovits_cn_api.py*'; foreach ($p in $ports) { $conns = Get-NetTCPConnection -LocalPort $p -State Listen -ErrorAction SilentlyContinue; if (-not $conns) { continue }; $killed = $false; foreach ($c in $conns) { $proc = Get-CimInstance Win32_Process -Filter ('ProcessId=' + $c.OwningProcess) -ErrorAction SilentlyContinue; if (-not $proc) { Start-Sleep -Milliseconds 300; $proc = Get-CimInstance Win32_Process -Filter ('ProcessId=' + $c.OwningProcess) -ErrorAction SilentlyContinue }; if ($proc) { $m = $false; foreach ($mk in $markers) { if ($proc.CommandLine -like $mk) { $m = $true } }; if ($m) { Stop-Process -Id $c.OwningProcess -Force -ErrorAction SilentlyContinue; $killed = $true } } }; if ($killed) { Write-Output ('port ' + $p + ' -> stopped') } else { Write-Output ('port ' + $p + ' -> other program, not touched') } }"
echo.
pause
