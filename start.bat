@echo off
cd /d "%~dp0"

echo ==> Installing backend dependencies...
uv sync --extra api

echo ==> Installing frontend dependencies...
cd frontend && call npm install && cd ..

echo ==> Starting backend (port 8000)...
start "OpenNoMark Backend" uv run uvicorn opennomark.api:app --port 8000

echo ==> Starting frontend (port 5173)...
cd frontend
start "OpenNoMark Frontend" npm run dev
cd ..

echo.
echo Backend:  http://localhost:8000
echo Frontend: http://localhost:5173
echo Close the terminal windows to stop.
echo.
pause
