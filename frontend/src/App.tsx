import { useState, useCallback, useRef } from "react";

/* ─── Types ─── */
interface ProcessResult {
  filename: string;
  job_id?: string;
  status: "cleaned" | "no_watermark" | "error";
  watermarks_found: number;
  download_url: string | null;
  error?: string;
}

interface ImageEntry {
  file: File;
  preview: string;
  result?: ProcessResult;
  cleanPreview?: string;
}

/* ─── Before/After Compare ─── */
function CompareSlider({ before, after }: { before: string; after: string }) {
  const ref = useRef<HTMLDivElement>(null);
  const [split, setSplit] = useState(50);

  const handleMove = useCallback(
    (clientX: number) => {
      if (!ref.current) return;
      const rect = ref.current.getBoundingClientRect();
      const pct = ((clientX - rect.left) / rect.width) * 100;
      setSplit(Math.max(2, Math.min(98, pct)));
    },
    []
  );

  const onPointerDown = useCallback(
    (e: React.PointerEvent) => {
      e.preventDefault();
      (e.target as HTMLElement).setPointerCapture(e.pointerId);
      handleMove(e.clientX);
    },
    [handleMove]
  );

  const onPointerMove = useCallback(
    (e: React.PointerEvent) => {
      if (e.buttons === 0) return;
      handleMove(e.clientX);
    },
    [handleMove]
  );

  return (
    <div
      ref={ref}
      className="compare-container rounded-lg overflow-hidden select-none"
      style={{ "--split": `${split}%` } as React.CSSProperties}
      onPointerDown={onPointerDown}
      onPointerMove={onPointerMove}
    >
      {/* Before (full, underneath) */}
      <img src={before} alt="Before" className="block w-full" draggable={false} />

      {/* After (clipped) */}
      <div className="after-clip">
        <img src={after} alt="After" className="block w-full" draggable={false} />
      </div>

      {/* Divider line + handle */}
      <div className="divider">
        <div className="divider-handle">
          <svg width="14" height="14" viewBox="0 0 14 14" fill="none">
            <path d="M4 2L1 7L4 12" stroke="#0a0a0c" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
            <path d="M10 2L13 7L10 12" stroke="#0a0a0c" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
        </div>
      </div>

      {/* Labels */}
      <span className="absolute top-3 left-3 px-2 py-0.5 text-[10px] font-mono uppercase tracking-wider rounded bg-black/60 text-[var(--color-text-muted)]">
        Before
      </span>
      <span className="absolute top-3 right-3 px-2 py-0.5 text-[10px] font-mono uppercase tracking-wider rounded bg-black/60 text-[var(--color-accent)]">
        After
      </span>
    </div>
  );
}

/* ─── Status Dot ─── */
function StatusDot({ status }: { status: string }) {
  const color =
    status === "cleaned" ? "bg-[var(--color-success)]" :
    status === "no_watermark" ? "bg-[var(--color-warn)]" :
    "bg-[var(--color-error)]";
  return <div className={`w-1.5 h-1.5 rounded-full ${color}`} />;
}

/* ─── App ─── */
export default function App() {
  const [images, setImages] = useState<ImageEntry[]>([]);
  const [processing, setProcessing] = useState(false);
  const [progress, setProgress] = useState({ current: 0, total: 0 });
  const [dragActive, setDragActive] = useState(false);
  const [selectedIdx, setSelectedIdx] = useState<number | null>(null);

  const hasResults = images.some((img) => img.result);

  const addFiles = useCallback((fileList: FileList | File[]) => {
    const newEntries: ImageEntry[] = Array.from(fileList)
      .filter((f) => f.type.startsWith("image/"))
      .map((f) => ({ file: f, preview: URL.createObjectURL(f) }));
    setImages((prev) => [...prev, ...newEntries]);
  }, []);

  const removeImage = (idx: number) => {
    setImages((prev) => {
      URL.revokeObjectURL(prev[idx].preview);
      if (prev[idx].cleanPreview) URL.revokeObjectURL(prev[idx].cleanPreview!);
      return prev.filter((_, i) => i !== idx);
    });
    if (selectedIdx === idx) setSelectedIdx(null);
    else if (selectedIdx !== null && selectedIdx > idx) setSelectedIdx(selectedIdx - 1);
  };

  const processAll = async () => {
    if (images.length === 0) return;
    setProcessing(true);
    setProgress({ current: 0, total: images.length });

    const formData = new FormData();
    images.forEach((img) => formData.append("files", img.file));

    try {
      const res = await fetch("/api/remove", { method: "POST", body: formData });
      const data = await res.json();

      setImages((prev) =>
        prev.map((img, i) => {
          const r = data.results[i] as ProcessResult;
          return {
            ...img,
            result: r,
            cleanPreview: r.download_url || undefined,
          };
        })
      );
      setProgress({ current: images.length, total: images.length });

      // Auto-select first cleaned result
      const firstCleaned = (data.results as ProcessResult[]).findIndex(
        (r) => r.status === "cleaned"
      );
      if (firstCleaned >= 0) setSelectedIdx(firstCleaned);
    } catch {
      setImages((prev) =>
        prev.map((img) => ({
          ...img,
          result: {
            filename: img.file.name,
            status: "error" as const,
            watermarks_found: 0,
            download_url: null,
            error: "Connection failed",
          },
        }))
      );
    } finally {
      setProcessing(false);
    }
  };

  const clearAll = () => {
    images.forEach((img) => {
      URL.revokeObjectURL(img.preview);
      if (img.cleanPreview) URL.revokeObjectURL(img.cleanPreview);
    });
    setImages([]);
    setSelectedIdx(null);
  };

  const stats = {
    cleaned: images.filter((i) => i.result?.status === "cleaned").length,
    skipped: images.filter((i) => i.result?.status === "no_watermark").length,
    errors: images.filter((i) => i.result?.status === "error").length,
  };

  const selected = selectedIdx !== null ? images[selectedIdx] : null;

  return (
    <div className="min-h-screen flex flex-col" style={{ background: "var(--color-bg)" }}>
      {/* ── Header ── */}
      <header
        className="shrink-0 flex items-center justify-between px-5 h-12"
        style={{ borderBottom: "1px solid var(--color-border-subtle)" }}
      >
        <div className="flex items-center gap-2.5">
          {/* Logo mark: rotated square = diamond, nods to "remove mark" */}
          <div
            className="w-5 h-5 rotate-45 rounded-[3px]"
            style={{ background: "var(--color-accent)" }}
          />
          <span className="text-sm font-semibold tracking-tight" style={{ color: "var(--color-text)" }}>
            OpenNoMark
          </span>
          <span
            className="text-[10px] font-mono px-1.5 py-0.5 rounded ml-1"
            style={{ background: "var(--color-surface-raised)", color: "var(--color-text-dim)" }}
          >
            v0.1.0
          </span>
        </div>

        <div className="flex items-center gap-3">
          {images.length > 0 && (
            <button
              onClick={clearAll}
              className="text-xs px-3 py-1.5 rounded-md transition-colors"
              style={{ color: "var(--color-text-muted)" }}
              onMouseEnter={(e) => (e.currentTarget.style.background = "var(--color-surface-hover)")}
              onMouseLeave={(e) => (e.currentTarget.style.background = "transparent")}
            >
              Clear all
            </button>
          )}
          {images.length > 0 && !processing && (
            <button
              onClick={processAll}
              className="text-xs font-medium px-4 py-1.5 rounded-md transition-all"
              style={{
                background: "var(--color-accent)",
                color: "var(--color-bg)",
              }}
              onMouseEnter={(e) => (e.currentTarget.style.opacity = "0.85")}
              onMouseLeave={(e) => (e.currentTarget.style.opacity = "1")}
            >
              Remove watermarks ({images.length})
            </button>
          )}
        </div>
      </header>

      {/* ── Main ── */}
      <div className="flex-1 flex overflow-hidden">
        {/* ── Left panel: image list ── */}
        <aside
          className="w-64 shrink-0 flex flex-col overflow-y-auto"
          style={{ borderRight: "1px solid var(--color-border-subtle)", background: "var(--color-surface)" }}
        >
          {/* Upload trigger */}
          <div className="p-3">
            <label
              className={`flex flex-col items-center justify-center gap-1.5 p-4 rounded-lg cursor-pointer transition-all ${
                dragActive ? "upload-zone-active" : ""
              }`}
              style={{
                border: `1.5px dashed ${dragActive ? "var(--color-accent)" : "var(--color-border)"}`,
                background: dragActive ? "var(--color-accent-muted)" : "transparent",
              }}
              onDragOver={(e) => { e.preventDefault(); setDragActive(true); }}
              onDragLeave={() => setDragActive(false)}
              onDrop={(e) => { e.preventDefault(); setDragActive(false); addFiles(e.dataTransfer.files); }}
            >
              <input
                type="file"
                multiple
                accept="image/*"
                className="hidden"
                onChange={(e) => e.target.files && addFiles(e.target.files)}
              />
              <svg width="20" height="20" viewBox="0 0 20 20" fill="none" style={{ color: "var(--color-text-dim)" }}>
                <path d="M10 4V16M4 10H16" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
              </svg>
              <span className="text-[11px]" style={{ color: "var(--color-text-dim)" }}>
                Drop or click
              </span>
            </label>
          </div>

          {/* Image list */}
          <div className="flex-1 px-3 pb-3 space-y-1">
            {images.map((img, i) => (
              <div
                key={i}
                onClick={() => setSelectedIdx(i)}
                className="stagger-item group flex items-center gap-2.5 p-1.5 rounded-lg cursor-pointer transition-colors"
                style={{
                  animationDelay: `${i * 40}ms`,
                  background: selectedIdx === i ? "var(--color-surface-hover)" : "transparent",
                  border: selectedIdx === i ? "1px solid var(--color-border)" : "1px solid transparent",
                }}
                onMouseEnter={(e) => {
                  if (selectedIdx !== i) e.currentTarget.style.background = "var(--color-surface-raised)";
                }}
                onMouseLeave={(e) => {
                  if (selectedIdx !== i) e.currentTarget.style.background = "transparent";
                }}
              >
                <img
                  src={img.preview}
                  alt=""
                  className="w-9 h-9 rounded object-cover shrink-0"
                />
                <div className="flex-1 min-w-0">
                  <p className="text-[11px] truncate" style={{ color: "var(--color-text)" }}>
                    {img.file.name}
                  </p>
                  {img.result && (
                    <div className="flex items-center gap-1.5 mt-0.5">
                      <StatusDot status={img.result.status} />
                      <span className="text-[10px]" style={{ color: "var(--color-text-dim)" }}>
                        {img.result.status === "cleaned"
                          ? `${img.result.watermarks_found} removed`
                          : img.result.status === "no_watermark"
                          ? "None found"
                          : "Error"}
                      </span>
                    </div>
                  )}
                </div>
                <button
                  onClick={(e) => { e.stopPropagation(); removeImage(i); }}
                  className="opacity-0 group-hover:opacity-100 w-5 h-5 rounded flex items-center justify-center transition-opacity"
                  style={{ color: "var(--color-text-dim)" }}
                  aria-label="Remove image"
                >
                  <svg width="10" height="10" viewBox="0 0 10 10">
                    <path d="M1 1L9 9M9 1L1 9" stroke="currentColor" strokeWidth="1.2" strokeLinecap="round" />
                  </svg>
                </button>
              </div>
            ))}
          </div>

          {/* Stats bar */}
          {hasResults && (
            <div
              className="shrink-0 px-4 py-2.5 flex gap-4 text-[10px] font-mono"
              style={{ borderTop: "1px solid var(--color-border-subtle)" }}
            >
              <span style={{ color: "var(--color-success)" }}>{stats.cleaned} cleaned</span>
              <span style={{ color: "var(--color-warn)" }}>{stats.skipped} skipped</span>
              {stats.errors > 0 && (
                <span style={{ color: "var(--color-error)" }}>{stats.errors} failed</span>
              )}
            </div>
          )}
        </aside>

        {/* ── Center: preview ── */}
        <main className="flex-1 flex items-center justify-center p-6 overflow-auto">
          {processing && (
            <div className="flex flex-col items-center gap-4">
              <div className="w-48 h-1 rounded-full overflow-hidden" style={{ background: "var(--color-surface-raised)" }}>
                <div
                  className="h-full rounded-full progress-bar-active transition-all duration-500"
                  style={{
                    background: "var(--color-accent)",
                    width: `${progress.total > 0 ? (progress.current / progress.total) * 100 : 0}%`,
                  }}
                />
              </div>
              <span className="text-xs font-mono" style={{ color: "var(--color-text-muted)" }}>
                Processing {images.length} images...
              </span>
            </div>
          )}

          {!processing && !selected && images.length === 0 && (
            <div className="text-center max-w-xs">
              <div
                className="w-16 h-16 mx-auto mb-5 rounded-2xl rotate-45 flex items-center justify-center"
                style={{ background: "var(--color-surface-raised)", border: "1px solid var(--color-border)" }}
              >
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" className="-rotate-45">
                  <path d="M21 15V19C21 20.1 20.1 21 19 21H5C3.9 21 3 20.1 3 19V15" stroke="var(--color-text-dim)" strokeWidth="1.5" strokeLinecap="round" />
                  <path d="M12 3V15M12 3L8 7M12 3L16 7" stroke="var(--color-text-dim)" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
                </svg>
              </div>
              <p className="text-sm mb-1" style={{ color: "var(--color-text-muted)" }}>
                Drop images to get started
              </p>
              <p className="text-xs" style={{ color: "var(--color-text-dim)" }}>
                Supports Gemini, Doubao, DALL-E watermarks
              </p>
            </div>
          )}

          {!processing && !selected && images.length > 0 && (
            <div className="text-center">
              <p className="text-sm mb-3" style={{ color: "var(--color-text-muted)" }}>
                {images.length} image{images.length > 1 ? "s" : ""} ready
              </p>
              <button
                onClick={processAll}
                className="text-sm font-medium px-6 py-2.5 rounded-lg transition-all"
                style={{ background: "var(--color-accent)", color: "var(--color-bg)" }}
                onMouseEnter={(e) => (e.currentTarget.style.opacity = "0.85")}
                onMouseLeave={(e) => (e.currentTarget.style.opacity = "1")}
              >
                Remove watermarks
              </button>
            </div>
          )}

          {!processing && selected && (
            <div className="w-full max-w-2xl">
              {selected.result?.status === "cleaned" && selected.cleanPreview ? (
                <CompareSlider before={selected.preview} after={selected.cleanPreview} />
              ) : (
                <img
                  src={selected.preview}
                  alt={selected.file.name}
                  className="w-full rounded-lg"
                />
              )}

              {/* Image info bar */}
              <div
                className="flex items-center justify-between mt-3 px-1 text-[11px] font-mono"
                style={{ color: "var(--color-text-dim)" }}
              >
                <span>{selected.file.name}</span>
                <div className="flex items-center gap-3">
                  <span>{(selected.file.size / 1024).toFixed(0)} KB</span>
                  {selected.result?.status === "cleaned" && selected.result.download_url && (
                    <a
                      href={selected.result.download_url}
                      className="px-2.5 py-1 rounded transition-colors"
                      style={{
                        background: "var(--color-accent-muted)",
                        color: "var(--color-accent)",
                        border: "1px solid var(--color-accent-border)",
                      }}
                    >
                      Download
                    </a>
                  )}
                </div>
              </div>
            </div>
          )}
        </main>
      </div>
    </div>
  );
}
