@echo off
cd /d "%~dp0"
call .venv312\Scripts\activate.bat
python -m streamlit run frontend_app_live.py
pause
