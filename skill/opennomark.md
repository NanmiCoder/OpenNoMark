---
name: opennomark
description: AI watermark detection and removal. Use when user wants to remove watermarks from AI-generated images (Gemini, Doubao, DALL-E, etc). Triggers on keywords like "remove watermark", "watermark", "去水印", "去除水印", "水印去除".
---

# OpenNoMark - AI Watermark Removal

Remove watermarks from AI-generated images using OWLv2 detection + LaMa inpainting.

## Usage

When the user asks to remove watermarks from images, run the CLI tool:

```bash
cd /Users/nanmi/workspace/github/OpenNoMark
uv run opennomark <image_paths_or_dirs> -o <output_dir>
```

## Examples

Single file:
```bash
uv run opennomark /path/to/image.png -o output
```

Multiple files:
```bash
uv run opennomark image1.png image2.jpg image3.png -o output
```

Entire directory:
```bash
uv run opennomark /path/to/images/ -o output
```

Multiple directories:
```bash
uv run opennomark gemini_images/ doubao_images/ -o output
```

With debug output (saves detection boxes and masks):
```bash
uv run opennomark /path/to/images/ -o output --debug
```

## API Server

Start the FastAPI backend:
```bash
cd /Users/nanmi/workspace/github/OpenNoMark
uv run uvicorn opennomark.api:app --reload --port 48291
```

## Supported Platforms

- Google Gemini (diamond icon, bottom-right)
- Doubao/豆包 ("AI 生成" text, top-left)
- More platforms can be added by expanding detection queries

## Limitations

- Detection uses OWLv2 general-purpose model, may miss watermarks with very low contrast
- Best for small corner watermarks (icons, short text)
- Large or center-positioned watermarks may need manual mask input
