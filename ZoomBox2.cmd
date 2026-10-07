@echo off
"%~dp0.venv\Scripts\python.exe" "%~dp0tools\launch_zoom_box2.py" %*
exit /b %errorlevel%
