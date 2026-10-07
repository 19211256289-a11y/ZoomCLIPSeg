@echo off
"%~dp0.venv\Scripts\python.exe" "%~dp0tools\launch_polyp_pvt.py" %*
exit /b %errorlevel%
