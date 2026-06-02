@echo off
cd /d "%~dp0"

set "BACKEND_PORT=48291"
set "FRONTEND_PORT=48292"

echo ==> Installing backend dependencies...
uv sync --extra api

echo ==> Installing frontend dependencies...
cd frontend && call npm install && cd ..

echo ==> Starting backend (port %BACKEND_PORT%)...
start "OpenNoMark Backend" uv run uvicorn opennomark.api:app --port %BACKEND_PORT%

echo ==> Starting frontend (port %FRONTEND_PORT%)...
cd frontend
start "OpenNoMark Frontend" npm run dev
cd ..

echo.
echo Backend:  http://localhost:%BACKEND_PORT%
echo Frontend: http://localhost:%FRONTEND_PORT%
echo Close the terminal windows to stop.
echo.
pause
