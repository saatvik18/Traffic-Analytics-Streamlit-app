@echo off
REM Launch the dashboard without worrying about the working directory.
REM The venv lives one level up, in S:\TrafficAnalytics\venv
cd /d "%~dp0"
"%~dp0..\venv\Scripts\python.exe" -m streamlit run app.py
pause
