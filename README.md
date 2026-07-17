<div align="center">

# OpenNoMark

**Local, batch-first watermark removal for AI-generated images.**

Inspect every file. Compare the result. Download one image or the whole batch.

**English** · [简体中文](README.zh-CN.md)

</div>

OpenNoMark combines watermark localization with content-aware LaMa inpainting. It is designed for visible corner marks found in images from Gemini, Doubao, Qwen, and similar generators, while keeping the complete workflow on your own machine.

> The project removes visible overlays; it does not alter or remove invisible provenance metadata or content credentials.

[Web workbench](#web-workbench) · [Results](#real-image-results) · [Install](#choose-your-workflow) · [Architecture](#how-it-works) · [Verification](#dataset-and-verification)

## Why OpenNoMark

| Batch without losing context | Review before saving | Local by default |
| :--- | :--- | :--- |
| Upload one image or a whole set. Each file has its own queued, uploading, processing, completed, or failed state. | Compare the original and processed image, retry individual failures, and download one result or every completed result as a ZIP. | Detection and reconstruction run locally. The application does not send images to a hosted inference API. |

The same processing core powers the Web UI, command-line interface, Python API, and cross-agent Skill. You can start visually, automate later, and get the same output semantics in every workflow.

## Web workbench

The responsive workbench is built for both single-image inspection and long-running batches. It keeps finished results available while the remaining images continue, shows per-file progress, and makes the next action clear on desktop and mobile.

<!-- Keep this file as a real capture of the running frontend, never a mockup. -->
![OpenNoMark Web workbench showing a batch and before/after comparison](docs/assets/opennomark-workbench.png)

### A simple batch workflow

1. Drop or select PNG, JPEG, or WebP images.
2. Start the pending files and follow each image independently.
3. Open any completed item for a before/after comparison.
4. Download one result immediately, retry a failure, or export all completed results as a ZIP.
5. Add more images without reprocessing results that are already complete.

The interface includes an English and Simplified Chinese language switch.

## Real-image results

These examples are repository artifacts, not design mockups. The first row contains source images; the second row contains the corresponding OpenNoMark outputs.

### Gemini · sparkle mark

| Original | Original | Original |
| :---: | :---: | :---: |
| ![Gemini source sample 1](examples/gemini/gemini_sample_1.png) | ![Gemini source sample 2](examples/gemini/gemini_sample_2.png) | ![Gemini source sample 3](examples/gemini/gemini_sample_3.png) |
| **Processed** | **Processed** | **Processed** |
| ![Processed Gemini sample 1](examples/gemini/clean_gemini_sample_1.png) | ![Processed Gemini sample 2](examples/gemini/clean_gemini_sample_2.png) | ![Processed Gemini sample 3](examples/gemini/clean_gemini_sample_3.png) |

### Doubao · “AI generated” label

| Original | Original | Original |
| :---: | :---: | :---: |
| ![Doubao source sample 1](examples/doubao/doubao_sample_1.jpg) | ![Doubao source sample 2](examples/doubao/doubao_sample_2.jpg) | ![Doubao source sample 3](examples/doubao/doubao_sample_3.jpg) |
| **Processed** | **Processed** | **Processed** |
| ![Processed Doubao sample 1](examples/doubao/clean_doubao_sample_1.jpg) | ![Processed Doubao sample 2](examples/doubao/clean_doubao_sample_2.jpg) | ![Processed Doubao sample 3](examples/doubao/clean_doubao_sample_3.jpg) |

### Qwen · icon and text signature

| Original | Processed |
| :---: | :---: |
| ![Qwen source with a visible bottom-right signature](examples/qwen/image_049781270343728.png) | ![The same Qwen image after OpenNoMark processing](examples/qwen/clean_image_049781270343728.png) |

LaMa reconstructs the masked area from its surroundings instead of blurring the watermark. Results still depend on the mark, contrast, and underlying texture; review important images in the comparison view.

## Choose your workflow

### 1. Web UI · best for visual review and batches

Requirements: Python 3.10+, [uv](https://docs.astral.sh/uv/), and Node.js/npm. A machine with at least 16 GB of memory is recommended.

```bash
git clone https://github.com/NanmiCoder/OpenNoMark.git
cd OpenNoMark

# macOS / Linux
./start.sh

# Windows
start.bat
```

Open [http://localhost:48292](http://localhost:48292). The launcher installs the API and frontend dependencies, then starts both development servers.

For manual startup:

```bash
# Terminal 1 · API
uv sync --extra api
uv run uvicorn opennomark.api:app --port 48291

# Terminal 2 · Web UI
cd frontend
npm install
npm run dev
```

### 2. CLI · best for folders and automation

Install the command without cloning the repository:

```bash
uv tool install git+https://github.com/NanmiCoder/OpenNoMark.git
opennomark --version
```

Process a file, several paths, a directory, or a glob:

```bash
opennomark image.png -o output/
opennomark img1.png img2.jpg -o output/
opennomark ./my-images/ -o output/
opennomark "./incoming/*.webp" -o output/

# Detection boxes and masks for debugging
opennomark ./my-images/ -o output/ --debug

# One machine-readable response for agents and scripts
opennomark ./my-images/ -o output/ --json

# Explicit device selection when needed
opennomark image.png -o output/ --device cpu
```

The first processing run downloads the required model weights: OWLv2 is approximately 500 MB and LaMa is approximately 196 MB.

### 3. Agent Skill · best inside coding agents

OpenNoMark follows the cross-agent [`skills`](https://github.com/vercel-labs/skills) layout. Install the portable instruction layer for Codex, Claude Code, Cursor, OpenCode, and other supported agents:

```bash
# Detect installed agents automatically
npx skills add NanmiCoder/OpenNoMark --skill opennomark

# Non-interactive global install for selected agents
npx skills add NanmiCoder/OpenNoMark --skill opennomark -g \
  -a codex -a claude-code -a cursor -y
```

The Skill invokes the same versioned CLI, directly when installed or through `uvx`; it does not maintain a second image-processing implementation.

### 4. Python API · best for application integration

```python
from opennomark.pipeline import WatermarkRemovalPipeline

pipeline = WatermarkRemovalPipeline()

image, metadata = pipeline.process("image.png", "clean_image.png")
print(metadata["status"], metadata["watermarks_found"])

results = pipeline.process_batch(
    ["img1.png", "img2.jpg"],
    output_dir="output/",
    callback=lambda index, total, item: print(index, total, item["status"]),
)
```

## How it works

OpenNoMark separates **where the watermark is** from **how the missing content is reconstructed**. That boundary lets detection evolve across generators without duplicating the inpainting stack.

```text
Input image
    │
    ▼
Unified WatermarkLocalizer
    ├── Trained spatial + edge template evidence
    └── Lazy OWLv2 semantic proposals
              │
              ▼
    Calibrated edge/shape scoring
    + coverage-aware deduplication
              │
              ▼
      Tight, feathered mask
              │
              ▼
       Local LaMa inpaint
              │
              ▼
     Same-expert residual check
       ├── passed → cleaned
       └── residual → one controlled retry
              │
              ▼
    Image result + unified regions metadata
```

- **One production contract:** every detector returns `box`, `score`, `source`, and `method` through the same region interface. The pipeline never routes by filename or provider directory.
- **Calibrated localization:** Gemini's trained spatial detector supplies a shape mask. The open-vocabulary expert combines `watermark` and `brand watermark` proposals with edge distance, text geometry, confidence ranking, and overlap-aware deduplication. The calibration profile is packaged with the application.
- **Reconstruction and validation:** LaMa runs against a compact local mask, then the same visual expert checks the repaired area. A residual triggers one mask-only retry; unresolved evidence is reported as `partial` instead of silently claiming success.
- **Lazy loading:** OWLv2 is loaded only when high-precision spatial evidence does not already identify the mark.
- **Safety boundary:** the pipeline targets small visible corner marks. It is not a general object-removal tool and intentionally avoids broad full-image edits.

### Device behavior

| Environment | Detection | Inpainting |
| :--- | :--- | :--- |
| NVIDIA CUDA | CUDA | CUDA |
| Apple Silicon | MPS when available | CPU fallback |
| CPU-only | CPU | CPU |

LaMa is loaded through CPU-compatible deserialization before being moved to a supported target. Requests for LaMa on MPS fall back to CPU because its TorchScript graph includes operations not supported reliably by MPS.

## Dataset and verification

The repository keeps real source images grouped by generator. Treat these directories as a regression corpus when changing candidate generation, scoring, masks, or inpainting:

| Corpus | Coverage focus |
| :--- | :--- |
| [`examples/gemini/`](examples/gemini/) | Sparkle sizes, output tiers, anchors, contrast, and textured backgrounds |
| [`examples/doubao/`](examples/doubao/) | Text labels and newer visible logo variants across portraits and complex scenes |
| [`examples/qwen/`](examples/qwen/) | Qwen logo variants, scale changes, and diverse corner backgrounds |

Do not infer quality from one showcase image or a fixed headline percentage. Run the automated suite, inspect every corpus result, and compare the modified area at full resolution.

The checked [full-corpus report](docs/verification/corpus-full.json) covers all 56 current original images: Gemini 19/19, Doubao 16/16, and Qwen 21/21. This is a regression result for the committed corpus, not a promise that every future watermark style will be recognized.

```bash
# Backend and integration tests
uv sync --extra api --extra dev
uv run pytest tests/ -v

# Frontend quality gates
cd frontend
npm install
npm run lint
npm run build

# Fast, filename-independent localization gate across every original image
cd ..
uv run python -m tests.dataset_evaluation --mode localize \
  --output verify/localization.json

# Full release gate: localized pixel change, containment, and residual checks
uv run python -m tests.dataset_evaluation --mode full \
  --output verify/full.json --results-dir verify/corpus
```

## Development map

```text
OpenNoMark/
├── opennomark/
│   ├── pipeline.py          # routing and processing metadata
│   ├── localizer.py         # unified visual-evidence region contract
│   ├── detector.py          # calibrated open-vocabulary proposals
│   ├── gemini_alpha.py      # Gemini detection and mask utilities
│   ├── inpainter.py         # LaMa masking, local repair, and blending
│   ├── cli.py               # command-line entry point
│   ├── api.py               # FastAPI upload and download endpoints
│   └── assets/              # packaged detector data and alpha maps
├── frontend/                # React 19 + Vite Web workbench
├── skills/opennomark/       # cross-agent Skill
├── examples/                # Gemini, Doubao, and Qwen regression images
├── scripts/                 # data-driven detector calibration
└── tests/                   # unit, integration, API, CLI, and Skill tests
```

Install a source checkout for development with:

```bash
uv sync --extra api --extra dev
cd frontend && npm install
```

When changing detection, evaluate false positives as carefully as successful removals. A missed mark leaves the source image intact; an incorrect mask can erase real content.

## Privacy, scope, and limitations

- Processing runs on your machine, but model weights are fetched from Hugging Face and GitHub Releases on first use.
- The local Web API writes uploads and results under your operating system's temporary directory. Do not expose the development server to an untrusted network; it has no authentication layer.
- Low-contrast marks, new layouts, or marks far from the corners can still be missed.
- Full-frame tiled watermarks, invisible provenance signals, and arbitrary object removal are outside the current scope.
- Inpainting is generative reconstruction. Fine text, faces, repeated patterns, and hard edges deserve manual review.

Use OpenNoMark only on images you own or are authorized to modify, and follow the source platform's terms and applicable law.

## License

OpenNoMark is provided under the [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International license](LICENSE). In particular, the repository is licensed for non-commercial use; review the full license before redistributing or integrating it.
