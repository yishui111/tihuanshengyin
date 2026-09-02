@echo off
setlocal
REM ============================================================
REM  Stop ALL voice services: Hub(8000) + A(8010) + B(8020) + C(8030) + D(8040)
REM  Kills only processes listening on these ports (safe for other apps).
REM ============================================================
echo Stopping all voice services (8000/8010/8020/8030/8040) ...
powershell -NoProfile -Command "$ports = 8000,8010,8020,8030,8040; foreach ($p in $ports) { $c = Get-NetTCPConnection -LocalPort $p -State Listen -ErrorAction SilentlyContinue; if ($c) { $c | ForEach-Object { Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue }; Write-Output ('port ' + $p + ' -> stopped') } else { Write-Output ('port ' + $p + ' -> not running') } }"
echo.
pause
