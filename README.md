# OpenNoMark

**English** | [中文](README.zh-CN.md)

Watermark detection and seamless removal for AI-generated images.

Built on **OWLv2** (open-vocabulary object detection) for watermark localization + **LaMa** (large-mask inpainting) for seamless reconstruction. Removes visible watermarks from Gemini, Doubao, DALL-E and other major AI image platforms.

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
```

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
│   ├── gemini_alpha.py      # Gemini reverse alpha blending (linear-light, strict gates)
│   ├── detector.py          # OWLv2 watermark detection
│   ├── inpainter.py         # LaMa seamless inpainting (feather + alpha blend)
│   ├── pipeline.py          # Smart-routing pipeline
│   ├── cli.py               # CLI entry point
│   ├── api.py               # FastAPI backend
│   └── assets/              # Pre-computed watermark template data
├── frontend/                # React + Vite + Tailwind CSS frontend
│   └── src/App.tsx          # Main UI (drag-and-drop + before/after preview)
├── tests/                   # Test suite (42 cases)
│   ├── test_detector.py     # Detector unit tests
│   ├── test_inpainter.py    # Inpainter unit tests
│   ├── test_pipeline.py     # Pipeline + E2E tests
│   ├── test_cli.py          # CLI integration tests
│   ├── test_api.py          # FastAPI tests
│   └── test_skill.py        # Skill format validation
├── skill/                   # Claude Code Skill
│   └── opennomark.md
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
input ┬─[Gemini sparkle near-perfect match?]─yes→ [linear-light reverse alpha] ┐
      │                         └─no───────────────────────────────────────────┤
      └─→ [OWLv2 detect] → corner filter → [LaMa inpaint] ─────────────────────┴→ clean image
```

**Path A — Gemini reverse alpha blending (strict trigger, theoretically lossless)**: when the NCC template match confidence ≥ 0.95 AND the reconstructed region matches its surrounding background within 3 gray levels, we invert the alpha compositing formula in **linear-light space**: `original_linear = (watermarked_linear - α) / (1 - α)`, then convert back to sRGB. The strict thresholds exist to avoid the "dent artifact" that appears when the alpha map is misaligned (the sparkle position becomes a visible gray/dark diamond). This path triggers rarely — mostly on simple-background, natively generated Gemini images.

**Path B — OWLv2 + LaMa (main path)**: handles the vast majority of images. OWLv2 (0.6B params, open-vocabulary detection) locates watermark candidates; after position/size filtering, LaMa reconstructs the region from surrounding texture. Works stably across Gemini, Doubao, DALL-E and others.

| Platform | Default Method | Alpha Fast Path |
|------|---------|---------------|
| Google Gemini | OWLv2 + LaMa | If match ≥ 0.95 AND borders blend |
| Doubao (豆包) | OWLv2 + LaMa | — |
| DALL-E / other | OWLv2 + LaMa | — |

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

# Run all 42 tests
uv run pytest tests/ -v

# Unit tests only
uv run pytest tests/test_detector.py tests/test_inpainter.py -v

# Integration / E2E tests only
uv run pytest tests/test_pipeline.py tests/test_cli.py tests/test_api.py -v
```

## Known Limitations

- OWLv2 is a general-purpose detector — it may miss very low-contrast watermarks (e.g. light text on a white background)
- Only small corner watermarks are currently handled; full-image tiled watermarks are out of scope
- UI controls in app screenshots (back arrows, setting icons) can be misclassified as watermarks

## Disclaimer

This project is intended for academic research, technical study, and copyright-compliance use cases only. Users must follow each AI platform's content usage policies and their local laws and regulations. Do not use this tool to infringe intellectual property or to generate content that violates regulations.

## License

[Apache-2.0](LICENSE)
