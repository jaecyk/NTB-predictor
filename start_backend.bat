@echo off
cd /d "%~dp0"
call .venv312\Scripts\activate.bat
python -m uvicorn backend_main:app --reload
pause
