@echo off
echo ============================================
echo  JJK Domain Expansion - Start Backend
echo ============================================

cd /d "%~dp0backend"

echo Checking Python...
python --version

echo.
echo Installing dependencies (first time only)...
python -m pip install -r requirements.txt

echo.
echo Starting FastAPI server on http://localhost:8000
echo Press Ctrl+C to stop.
echo.
python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
pause
