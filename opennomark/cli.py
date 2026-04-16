"""CLI interface for OpenNoMark watermark removal."""

import argparse
import glob
import os
import sys


def resolve_paths(inputs):
    """Resolve input paths: support files, directories, and globs."""
    paths = []
    valid_exts = {".png", ".jpg", ".jpeg", ".webp"}

    for inp in inputs:
        if os.path.isdir(inp):
            for ext in valid_exts:
                paths.extend(glob.glob(os.path.join(inp, f"*{ext}")))
        elif os.path.isfile(inp):
            paths.append(inp)
        else:
            expanded = glob.glob(inp)
            paths.extend(f for f in expanded if os.path.isfile(f))

    # Filter valid image extensions and deduplicate
    paths = list(dict.fromkeys(
        p for p in paths if os.path.splitext(p)[1].lower() in valid_exts
    ))
    return sorted(paths)


def main():
    parser = argparse.ArgumentParser(
        prog="opennomark",
        description="AI watermark detection and removal",
    )
    parser.add_argument(
        "inputs", nargs="+",
        help="Image files, directories, or glob patterns",
    )
    parser.add_argument(
        "-o", "--output", default="output",
        help="Output directory (default: output)",
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Save detection debug images and masks",
    )
    parser.add_argument(
        "--device", default=None,
        help="Device: cpu, cuda, mps (default: auto)",
    )

    args = parser.parse_args()
    paths = resolve_paths(args.inputs)

    if not paths:
        print("No valid images found.", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(paths)} image(s) to process.")

    from .pipeline import WatermarkRemovalPipeline

    device = args.device
    pipeline = WatermarkRemovalPipeline(device=device)

    def on_progress(i, total, meta):
        status = meta["status"]
        found = meta["watermarks_found"]
        name = os.path.basename(meta["input"])
        if status == "cleaned":
            print(f"  [{i}/{total}] {name} -> {found} watermark(s) removed")
        else:
            print(f"  [{i}/{total}] {name} -> no watermark found")

    results = pipeline.process_batch(paths, args.output, save_debug=args.debug, callback=on_progress)

    cleaned = sum(1 for r in results if r["status"] == "cleaned")
    skipped = sum(1 for r in results if r["status"] == "no_watermark")
    print(f"\nDone! {cleaned} cleaned, {skipped} skipped. Output: {args.output}/")


if __name__ == "__main__":
    main()
