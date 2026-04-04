@echo off
cd /d "%~dp0"
start "NTB Backend" cmd /k "call .venv312\Scripts\activate.bat && python -m uvicorn backend_main:app --reload"
start "NTB Frontend" cmd /k "call .venv312\Scripts\activate.bat && python -m streamlit run frontend_app_live.py"
