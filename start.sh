#!/usr/bin/env bash
set -e

cd "$(dirname "$0")"

echo "==> Installing backend dependencies..."
uv sync --extra api

echo "==> Installing frontend dependencies..."
cd frontend && npm install && cd ..

echo "==> Starting backend (port 8000)..."
uv run uvicorn opennomark.api:app --port 8000 &
BACKEND_PID=$!

echo "==> Starting frontend (port 5173)..."
cd frontend && npm run dev &
FRONTEND_PID=$!

trap "kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit" INT TERM

echo ""
echo "Backend:  http://localhost:8000"
echo "Frontend: http://localhost:5173"
echo "Press Ctrl+C to stop."
echo ""

wait
