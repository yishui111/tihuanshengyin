@echo off
setlocal
REM ============================================================
REM  Stop ALL voice services - kills ONLY ports whose /health
REM  answers as OUR services; ports used by other programs are
REM  left alone. Honors fallback ports recorded by start.bat in
REM  last_run_ports.txt.
REM ============================================================
echo Stopping voice services ...
powershell -NoProfile -Command "$ports = @(); if (Test-Path ('%~dp0last_run_ports.txt')) { $ports += Get-Content ('%~dp0last_run_ports.txt') | ForEach-Object { [int]$_ } }; $ports += 8000,8010,8020,8030,8040; $ports = $ports | Select-Object -Unique; $names = '*hub*','*rvc-character*','*openvoice-clone*','*sovits-cn*','*gptsovits-cn*'; foreach ($p in $ports) { $c = Get-NetTCPConnection -LocalPort $p -State Listen -ErrorAction SilentlyContinue; if (-not $c) { continue }; $ours = $false; $body=$null; try { $req = [System.Net.HttpWebRequest]::Create('http://127.0.0.1:'+${p}+'/health'); $req.Proxy = New-Object System.Net.WebProxy($null); $req.Timeout = 2000; $res = $req.GetResponse(); $body = (New-Object IO.StreamReader($res.GetResponseStream())).ReadToEnd(); $res.Close(); foreach ($n in $names) { if ($body -like $n) { $ours = $true } } } catch {}; if ($ours) { $c | ForEach-Object { Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue }; Write-Output ('port ' + $p + ' -> stopped') } else { Write-Output ('port ' + $p + ' -> other program, not touched') } }"
echo.
pause
