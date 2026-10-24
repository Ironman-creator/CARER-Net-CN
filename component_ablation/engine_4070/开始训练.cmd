@echo off
setlocal
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0run_fives_component_ablation_4070.ps1" -Stage All
set "COMPONENT_EXIT=%ERRORLEVEL%"
echo.
if not "%COMPONENT_EXIT%"=="0" echo Pipeline stopped. Check outputs\logs and README_4070.md.
pause
exit /b %COMPONENT_EXIT%
