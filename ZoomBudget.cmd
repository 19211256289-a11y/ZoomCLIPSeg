@echo off
"%~dp0.venv\Scripts\python.exe" "%~dp0tools\launch_zoom_budget.py" %*
exit /b %errorlevel%
