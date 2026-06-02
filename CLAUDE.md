# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common commands

Dependency management uses **uv** (not pip). Frontend uses **npm**.

```bash
# Install deps
uv sync                         # core (torch, transformers, opencv, PIL)
uv sync --extra api             # + fastapi/uvicorn for Web UI
uv sync --extra dev             # + pytest

# CLI (entry point declared in pyproject.toml → opennomark.cli:main)
uv run opennomark <files|dirs|globs> -o output/ [--debug] [--device cpu|cuda|mps]

# API server (singleton pipeline loaded on first request)
uv run uvicorn opennomark.api:app --port 48291

# Frontend (React 19 + Vite 8 + Tailwind 4)
cd frontend && npm install && npm run dev      # dev server on :48292
cd frontend && npm run build                   # builds to frontend/dist (auto-served by api.py if present)
cd frontend && npm run lint

# Tests (42 cases)
uv run pytest tests/ -v
uv run pytest tests/test_pipeline.py -v                     # single file
uv run pytest tests/test_pipeline.py::test_name -v          # single test

# One-shot start (installs deps, runs backend + frontend)
./start.sh        # macOS/Linux
start.bat         # Windows
```

Tests referencing real sample images in `gemini_images/` and `豆包/` auto-skip via `pytest.skip` when those directories are absent — see `tests/conftest.py:45-61`.

## Architecture

### Two-stage "smart routing" pipeline (`opennomark/pipeline.py`)

Both stages are attempted on every image — this is intentional, not a fallthrough:

```
image ──► Stage 1: Gemini reverse-alpha (strict gate)
            │
            ├─ cleaned: working = alpha-cleaned image
            └─ rejected / no-watermark: working = original
          │
          ▼
      Stage 2: OWLv2 detect → filter_watermarks → LaMa inpaint
          │   (boxes overlapping Stage 1's cleaned region are dropped)
          ▼
      final image
```

Stage 2 **always runs** so text overlays (豆包 "AI 生成") are still caught even after a Gemini sparkle was alpha-removed.

### Stage 1 — Gemini reverse alpha (`opennomark/gemini_alpha.py`)

A lossless path for Gemini's white sparkle watermark, based on the known alpha-compositing formula. Key decisions:

- **Wide-net detection, strict acceptance**: `detect_gemini_watermark` casts a wide template-matching net, but `remove_gemini_watermark` only accepts the result when `confidence ≥ 0.95` AND `border_mismatch < 3` gray levels (see `_quality_accept`, lines 224-236). This two-layer gate exists because alpha-map misalignment produces a visible dark/light diamond artifact that looks *worse* than letting LaMa handle it.
- **Linear-light math**: reverse alpha runs in sRGB-decoded linear-light space (`_srgb_to_linear` / `_linear_to_srgb`) — doing it on sRGB bytes directly produces a darkening "dent". Any future changes here must preserve this color space step.
- **Pre-computed alpha maps** ship in `opennomark/assets/gemini_alpha_{48,96}.npy` (48 for ≤1024px images, 96 for larger).
- On rejection the function returns the **original** image unchanged and `status="rejected"` — the pipeline then falls through to Stage 2.

### Stage 2 — OWLv2 + LaMa

**Detector (`opennomark/detector.py`)**: OWLv2 open-vocabulary detection with text queries `["watermark", "logo", "icon", "symbol", "badge", "stamp"]` at a low score threshold (0.05). Raw detections are then filtered by `filter_watermarks` to keep only boxes that are (a) smaller than 30% of the image's short side and (b) centered inside a corner region (15% from any edge). Most false positives (UI controls, centered text) are rejected here.

**Inpainter (`opennomark/inpainter.py`)**: wraps a TorchScript LaMa model.
- **Tight mask defaults (`padding=3, feather=4`) are load-bearing** — see the docstring at line 28. Larger values cause LaMa to bleed across high-contrast structural edges (e.g. paint white fabric over a black panel adjacent to a sparkle). Do not relax these without re-validating on the `examples/` set.
- The model is downloaded on first run from `github.com/enesmsahin/simple-lama-inpainting/releases` to `~/.cache/torch/hub/checkpoints/big-lama.pt`.
- Loaded with `map_location="cpu"` because the serialized checkpoint is CUDA — this is what lets it run on Mac.
- After LaMa produces a result, `inpaint()` alpha-blends it back against the original using the feathered mask, so only the masked pixels are modified.

### Device selection

Both models accept a `device` kwarg (passed through from the `--device` CLI flag) and auto-select when not given, but they select **differently**:

- `WatermarkDetector` (OWLv2): MPS → CUDA → CPU.
- `LamaInpainter`: CUDA → CPU. **MPS requests are silently rewritten to CPU** because LaMa's TorchScript graph uses ops (e.g. FFT variants) that Apple MPS does not support — trying to run on MPS produces garbage, not an error. Do not "fix" this redirect without first confirming the full LaMa op set is MPS-supported.

The checkpoint is CUDA-serialized, so `LamaInpainter` always loads with `map_location="cpu"` then `.to(self.device)` — loading directly to CUDA on a CPU-only machine would fail at unpickle time.

Pipeline does **not** cache models across invocations when used via CLI, but `api.py` holds a module-level `_pipeline` singleton that is lazy-initialized on first `/api/remove` request.

### FastAPI backend (`opennomark/api.py`)

- Uploads and outputs are written to `tempfile.gettempdir()/opennomark_{uploads,outputs}` with an 8-char job id.
- If `frontend/dist` exists, it is mounted at `/` so the single-port deployment works (`uvicorn` only, no vite dev server needed).
- CORS is wide-open (`allow_origins=["*"]`) for the split-port dev setup.
- Split-port development uses backend `48291` and frontend `48292`; keep the Vite proxy and launch scripts aligned when changing either port.

## Directories worth knowing

- `opennomark/assets/` — pre-computed Gemini alpha maps; do not regenerate without re-validating the quality gate thresholds.
- `examples/` — small canonical before/after samples used in the README.
- `experiments/` — ablation scripts (`exp_decision.py`, `exp_gain_sweep.py`, `exp_linear_light.py`, `exp_mask_shape.py`, `exp_posterior.py`) that justify the current threshold constants. Consult these before tuning Stage 1 thresholds or mask parameters.
- `gemini_images/`, `豆包/`, `verify/` — larger real-image test sets used by fixtures; not required for unit tests.
- `skill/opennomark.md` — Claude Code skill file; `tests/test_skill.py` validates its frontmatter format.
