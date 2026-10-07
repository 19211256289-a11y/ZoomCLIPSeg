@echo off
"%~dp0.venv\Scripts\python.exe" "%~dp0tools\launch_pvt_ext.py" %*
exit /b %errorlevel%
