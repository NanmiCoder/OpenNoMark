---
name: opennomark
description: Remove visible watermarks from user-provided or authorized images, including Gemini sparkle marks, AI-generator corner signatures, and compact high-confidence text watermarks. Use when a user asks to remove, clean, erase, or batch-process visible image watermarks, including requests containing "remove watermark", "去水印", "去除水印", or "水印去除".
---

# OpenNoMark

Use the OpenNoMark CLI as the deterministic execution layer. Preserve every source image and write cleaned files to a separate output directory.

## Run

1. Confirm that the inputs are user-provided or that the user is authorized to edit them.
2. Resolve every requested image path. Accept PNG, JPEG, and WebP files, directories, and glob patterns.
3. Prefer an installed `opennomark` executable. Otherwise run it from the official repository with `uvx`:

```bash
opennomark <inputs...> --output <output-directory> --json
```

```bash
uvx --from git+https://github.com/NanmiCoder/OpenNoMark.git opennomark <inputs...> --output <output-directory> --json
```

4. Set `--device cuda` only when CUDA is available. Allow automatic device selection otherwise. Do not force `mps`; LaMa safely falls back to CPU on Apple Silicon.
5. Parse the JSON response. Treat exit code `0` with `status: "ok"` as success, `1` as invalid or missing input, and `2` as a model-loading or processing failure.
6. Report the absolute paths in `results[].output`, grouped by `cleaned` and `no_watermark`. Treat `partial` as a failure that requires review: it can mean unresolved residual evidence or that dense/tiled candidates exceeded the automatic-removal safety budget. Do not claim an image was cleaned when its status says otherwise.
7. Inspect at least one output when visual inspection is available. For batch requests, inspect representative outputs and any low-confidence or unexpected result.

The first run can take longer because model weights are downloaded and cached. Keep the command running while downloads or inference are active.

## Useful options

```bash
# Batch a directory without overwriting the originals
opennomark /path/to/images --output /path/to/cleaned --json

# Save detector boxes and masks when diagnosing a miss
opennomark /path/to/image.png --output /path/to/debug --debug --json

# Inspect the installed CLI version
opennomark --version
```

Use `--debug` only for diagnosis because it creates extra artifacts beside the cleaned output.
