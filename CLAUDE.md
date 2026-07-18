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

# Tests
uv run pytest tests/ -v
uv run pytest tests/test_pipeline.py -v                     # single file
uv run pytest tests/test_pipeline.py::test_name -v          # single test

# One-shot start (installs deps, runs backend + frontend)
./start.sh        # macOS/Linux
start.bat         # Windows
```

Tests referencing real sample images in `gemini_images/` and `豆包/` auto-skip via `pytest.skip` when those directories are absent — see `tests/conftest.py:45-61`.

## Architecture

### Fused visual-expert pipeline (`opennomark/localizer.py`)

```
image ──┬─► Gemini catalog detector (48/32, 96/64, 96/192)
        ├─► OWLv2 corner prompts ─┐
        ├─► OWLv2 generic prompts├─► conservative cross-expert fusion
        └─► lazy PP-OCRv5 polygons┘              │
                                                 ▼
                                    precise mask(s) → local LaMa
                                                 │
                                                 ▼
                                          residual check
```

The precise Gemini shape mask remains the default when both experts fire. A
non-overlapping text signature may replace a spatial-template match only when
it is in the same corner, scores at least as strongly, and lies closer to the
image edges. This resolves catalog-shaped background false positives without
provider names or filenames participating in production inference.

### Gemini branch (`opennomark/gemini_alpha.py`)

A deterministic catalog detector followed by shape-aware local inpainting:

- **Resolution tiers first**: large official outputs try 96/192 (May 2026) and 96/64; preview/1K outputs try 48/32 and legacy 96/64. Do not let all layouts compete across tiers.
- **Two-signal scoring**: spatial NCC and Sobel-edge NCC must both clear thresholds in `opennomark/assets/gemini_detector.json`.
- **Data-driven calibration**: regenerate the threshold model with `scripts/train_gemini_detector.py`; it trains on real positives plus same-image hard negatives and deduplicates inputs by SHA-256.
- **Tight mask**: `create_gemini_mask` thresholds the known alpha silhouette, dilates it by 5px, and feathers by 2px. Do not replace it with a full 96x96 rectangle without re-validating text-overlap samples.
- **Local LaMa**: `inpaint_local` uses roughly a 384x384 crop for a 96px mark. This is both faster and less destructive than full-image inference.

The older reverse-alpha helpers remain for experiments, but the production pipeline no longer uses them because complex backgrounds can produce visible positive/negative diamond residuals.

### Open-vocabulary, OCR, and LaMa

**Detector (`opennomark/detector.py`)**: OWLv2 runs the established corner prompt set and the generic watermark prompt set in separate model passes. Keep that query competition separate: adding generic phrases to the calibrated platform pass regresses the real corpus. `filter_watermarks` has a `corner_signature` lane plus a stricter `generic_anywhere` lane, clusters evidence per visual region, permits at most four regions, and fails the entire generic lane closed when its total coarse area exceeds 12%.

**Text detector (`opennomark/text_detector.py`)**: PP-OCRv5 mobile detection and recognition are imported and loaded lazily. Strong watermark vocabulary can be accepted anywhere; URLs, handles, and dates require edge placement. OCR polygons refine generic OWLv2 rectangles. Do not turn arbitrary recognized scene text into removal masks. Dense/tiled OCR candidates must keep returning an overflow report rather than a convenient prefix.

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

- `opennomark/assets/` — Gemini alpha maps plus trained detector thresholds; retrain and run the `many_images` regression when changing them.
- `scripts/train_gemini_detector.py` — calibration entry point for real Gemini samples and hard negatives.
- `skills/opennomark/` — portable Agent Skill discovered by `npx skills add NanmiCoder/OpenNoMark`; keep it free of machine-specific paths.
- `examples/` — the seven-provider real-image acceptance corpus plus canonical README outputs.
- `experiments/` — ablation scripts (`exp_decision.py`, `exp_gain_sweep.py`, `exp_linear_light.py`, `exp_mask_shape.py`, `exp_posterior.py`) that justify the current threshold constants. Consult these before tuning Stage 1 thresholds or mask parameters.
- `gemini_images/`, `豆包/`, `verify/` — larger real-image test sets used by fixtures; not required for unit tests.
