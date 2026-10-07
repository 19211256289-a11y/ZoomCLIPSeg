@echo off
if "%~1"=="" (
  "%~dp0.venv\Scripts\python.exe" "%~dp0tools\launch_zoom_export.py" -Action Run
  pause
) else (
  "%~dp0.venv\Scripts\python.exe" "%~dp0tools\launch_zoom_export.py" %*
)
