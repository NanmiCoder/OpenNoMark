#!/usr/bin/env bash
set -e

cd "$(dirname "$0")"

BACKEND_PORT=48291
FRONTEND_PORT=48292

echo "==> Installing backend dependencies..."
uv sync --extra api

echo "==> Installing frontend dependencies..."
cd frontend && npm install && cd ..

echo "==> Starting backend (port $BACKEND_PORT)..."
uv run uvicorn opennomark.api:app --port "$BACKEND_PORT" &
BACKEND_PID=$!

echo "==> Starting frontend (port $FRONTEND_PORT)..."
cd frontend && npm run dev &
FRONTEND_PID=$!

trap "kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit" INT TERM

echo ""
echo "Backend:  http://localhost:$BACKEND_PORT"
echo "Frontend: http://localhost:$FRONTEND_PORT"
echo "Press Ctrl+C to stop."
echo ""

wait
