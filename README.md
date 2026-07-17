# OpenNoMark

**English** | [中文](README.zh-CN.md)

Watermark detection and seamless removal for AI-generated images.

Uses a dedicated catalog-trained detector + local **LaMa** repair for Gemini, with **OWLv2** + LaMa as the generic fallback for Doubao, DALL-E, and other visible watermarks.

## Showcase

### Google Gemini — bottom-right sparkle watermark

Top row: originals (with watermark). Bottom row: cleaned by OpenNoMark.

|  |  |  |
| :---: | :---: | :---: |
| ![](examples/gemini/gemini_sample_1.png) | ![](examples/gemini/gemini_sample_2.png) | ![](examples/gemini/gemini_sample_3.png) |
| ![](examples/gemini/clean_gemini_sample_1.png) | ![](examples/gemini/clean_gemini_sample_2.png) | ![](examples/gemini/clean_gemini_sample_3.png) |

### Doubao (豆包) — top-left "AI 生成" text watermark

Top row: originals (with watermark). Bottom row: cleaned by OpenNoMark.

|  |  |  |
| :---: | :---: | :---: |
| ![](examples/doubao/doubao_sample_1.jpg) | ![](examples/doubao/doubao_sample_2.jpg) | ![](examples/doubao/doubao_sample_3.jpg) |
| ![](examples/doubao/clean_doubao_sample_1.jpg) | ![](examples/doubao/clean_doubao_sample_2.jpg) | ![](examples/doubao/clean_doubao_sample_3.jpg) |

The cleaned region is reconstructed by LaMa from surrounding textures — no blur or smudge artifacts.

| Platform | Watermark | Position | Detection Rate |
|------|---------|------|--------|
| Google Gemini | Diamond sparkle icon | Bottom-right | 100% |
| Doubao (豆包) | "AI 生成" text label | Top-left | 85% |

## Quick Start

### Requirements

- Python >= 3.10
- macOS (Apple Silicon MPS) / Linux / Windows (NVIDIA CUDA) / CPU
- 16GB+ RAM recommended
- NVIDIA GPU users: install the CUDA build of PyTorch (see [pytorch.org](https://pytorch.org/get-started/locally/))

### Install

```bash
git clone https://github.com/NanmiCoder/OpenNoMark.git
cd OpenNoMark

# Using uv (recommended)
uv sync                     # core dependencies
uv sync --extra api         # + Web UI / API
cd frontend && npm install && cd ..

# Or using pip
pip install -e .
pip install -e ".[api]"     # + Web UI / API
```

Install only the CLI from GitHub without cloning the repository:

```bash
uv tool install git+https://github.com/NanmiCoder/OpenNoMark.git
opennomark --version
```

Models are auto-downloaded on first run:
- OWLv2 (~500MB, HuggingFace)
- LaMa (~196MB, GitHub Release)

### CLI

```bash
# Single image
uv run opennomark image.png -o output/

# Multiple images
uv run opennomark img1.png img2.jpg img3.png -o output/

# Entire directory
uv run opennomark ./my_images/ -o output/

# Mixed directories
uv run opennomark gemini_images/ doubao_images/ -o output/

# With debug output (saves detection boxes and masks)
uv run opennomark ./images/ -o output/ --debug

# Machine-readable output for agents and scripts
uv run opennomark ./images/ -o output/ --json
```

### Agent Skill

OpenNoMark ships as a standard cross-agent Skill discovered by the latest
[`skills`](https://github.com/vercel-labs/skills) CLI. Install it through NPM for Codex,
Claude Code, Cursor, OpenCode, and other supported agents:

```bash
# Let the installer detect your agents
npx skills add NanmiCoder/OpenNoMark --skill opennomark

# Non-interactive global install for selected agents
npx skills add NanmiCoder/OpenNoMark --skill opennomark -g \
  -a codex -a claude-code -a cursor -y
```

The Skill is intentionally a thin, portable instruction layer. It invokes the
versioned Python CLI directly or through `uvx`, so the image pipeline has one
implementation across every agent.

### Web UI

**One-shot launcher (recommended):**

```bash
# macOS / Linux
./start.sh

# Windows
start.bat
```

Installs dependencies and starts both backend and frontend together.

**Manual start:**

```bash
# Terminal 1: backend
uv run uvicorn opennomark.api:app --port 48291

# Terminal 2: frontend
cd frontend && npm run dev
```

Open `http://localhost:48292` and drop in images. Supports batch upload, before/after preview, and single-file download.

### Python API

```python
from opennomark.pipeline import WatermarkRemovalPipeline

pipeline = WatermarkRemovalPipeline()

# Single image
result_img, meta = pipeline.process("image.png", "clean_image.png")
print(meta)  # {'status': 'cleaned', 'watermarks_found': 1, ...}

# Batch
results = pipeline.process_batch(
    ["img1.png", "img2.jpg"],
    output_dir="output/",
    callback=lambda i, total, m: print(f"[{i}/{total}] {m['status']}")
)
```

## Project Structure

```
OpenNoMark/
├── opennomark/              # Core Python package
│   ├── gemini_alpha.py      # Gemini catalog detector + sparkle mask
│   ├── detector.py          # OWLv2 watermark detection
│   ├── inpainter.py         # LaMa seamless inpainting (feather + alpha blend)
│   ├── pipeline.py          # Smart-routing pipeline
│   ├── cli.py               # CLI entry point
│   ├── api.py               # FastAPI backend
│   └── assets/              # Alpha templates + trained detector thresholds
├── frontend/                # React + Vite + Tailwind CSS frontend
│   └── src/App.tsx          # Main UI (drag-and-drop + before/after preview)
├── scripts/
│   └── train_gemini_detector.py # Data-driven Gemini detector calibration
├── tests/                   # Test suite (51 cases)
│   ├── test_gemini_alpha.py # Gemini layouts + many_images regression
│   ├── test_detector.py     # Detector unit tests
│   ├── test_inpainter.py    # Inpainter unit tests
│   ├── test_pipeline.py     # Pipeline + E2E tests
│   ├── test_cli.py          # CLI integration tests
│   ├── test_api.py          # FastAPI tests
│   └── test_skill.py        # Skill format validation
├── skills/                  # Cross-agent Skills (vercel-labs/skills layout)
│   └── opennomark/
│       ├── SKILL.md
│       └── agents/openai.yaml
├── examples/                # Sample images (with-watermark originals)
│   ├── gemini/              # Google Gemini samples
│   └── doubao/              # Doubao samples
├── start.sh                 # One-shot launcher (macOS / Linux)
├── start.bat                # One-shot launcher (Windows)
├── pyproject.toml           # Project config
├── uv.lock                  # Dependency lockfile
└── LICENSE
```

## How It Works

### Smart Routing Pipeline

```
input ──→ [catalog anchors: 48/32, 96/64, 96/192]
            │
            ├─ Gemini spatial+edge match → [tight sparkle mask] → [local LaMa] → clean image
            │
            └─ no Gemini match → [lazy OWLv2] → corner filter → [LaMa] → clean image
```

**Path A — dedicated Gemini catalog + local LaMa**: Gemini output tiers use known watermark sizes and anchors. The detector scores both luminance shape and Sobel-edge shape, including the May 2026 `96×96 / 192px margin` layout documented by [GargantuaX/gemini-watermark-remover](https://github.com/GargantuaX/gemini-watermark-remover). Thresholds are calibrated by `scripts/train_gemini_detector.py` from real positives and same-image hard negatives. A tight sparkle-shaped mask is repaired inside a small local crop, avoiding alpha-inversion dents and full-image inference.

**Path B — OWLv2 + LaMa (generic fallback)**: when no Gemini match is present, OWLv2 (0.6B params, open-vocabulary detection) locates generic watermark candidates and LaMa reconstructs them. OWLv2 is loaded lazily and is not run after a confirmed Gemini match, so unrelated UI icons are not erased.

| Platform | Default Method | Detector |
|------|---------|---------------|
| Google Gemini | Local shape-mask + LaMa | Catalog spatial + edge model |
| Doubao (豆包) | OWLv2 + LaMa | Open vocabulary |
| DALL-E / other | OWLv2 + LaMa | Open vocabulary |

### Hardware Acceleration

Device is auto-selected by default; override with `--device cuda|mps|cpu`.

| Platform | OWLv2 (detect) | LaMa (inpaint) |
|------|---------------|--------------|
| Linux / Windows + NVIDIA CUDA | CUDA | CUDA |
| macOS (Apple Silicon) | MPS | CPU (MPS lacks some LaMa ops — auto-fallback) |
| CPU-only | CPU | CPU |

LaMa's TorchScript checkpoint is CUDA-serialized. It's deserialized with `map_location="cpu"` for compatibility with non-NVIDIA machines, then moved to the target device.

## Testing

```bash
# Install dev dependencies
uv sync --extra dev

# Run all 51 tests
uv run pytest tests/ -v

# Unit tests only
uv run pytest tests/test_detector.py tests/test_inpainter.py -v

# Integration / E2E tests only
uv run pytest tests/test_pipeline.py tests/test_cli.py tests/test_api.py -v
```

## Known Limitations

- Gemini support targets the known visible sparkle layouts; future Gemini layout changes require catalog recalibration
- The generic OWLv2 fallback may miss very low-contrast non-Gemini watermarks
- Only small corner watermarks are currently handled; full-image tiled watermarks are out of scope
- UI controls in app screenshots (back arrows, setting icons) can be misclassified as watermarks

## Disclaimer

This project is intended for academic research, technical study, and copyright-compliance use cases only. Users must follow each AI platform's content usage policies and their local laws and regulations. Do not use this tool to infringe intellectual property or to generate content that violates regulations.

## License

[Apache-2.0](LICENSE)
