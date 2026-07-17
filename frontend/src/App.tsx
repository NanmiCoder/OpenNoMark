import { useCallback, useEffect, useRef, useState } from "react";
import type { CSSProperties } from "react";
import {
  ArrowClockwiseIcon,
  ArrowRightIcon,
  CheckCircleIcon,
  DownloadSimpleIcon,
  ImageSquareIcon,
  ImagesIcon,
  PlusIcon,
  ShieldCheckIcon,
  SparkleIcon,
  TrashIcon,
  UploadSimpleIcon,
  WarningCircleIcon,
  XCircleIcon,
} from "@phosphor-icons/react";

import { CompareSlider } from "./components/CompareSlider";
import { MagneticButton } from "./components/MagneticButton";

type ResultStatus = "cleaned" | "no_watermark" | "error";

interface ProcessResult {
  filename: string;
  job_id?: string;
  status: ResultStatus;
  watermarks_found: number;
  download_url: string | null;
  error?: string;
}

interface ImageEntry {
  id: string;
  file: File;
  preview: string;
  result?: ProcessResult;
}

const acceptedExtensions = /\.(png|jpe?g|webp)$/i;

function formatBytes(bytes: number) {
  if (bytes < 1024 * 1024) return `${Math.max(1, Math.round(bytes / 1024))} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function normalizeResult(raw: Partial<ProcessResult> | undefined, file: File): ProcessResult {
  if (!raw || raw.error || !raw.status) {
    return {
      filename: file.name,
      status: "error",
      watermarks_found: 0,
      download_url: null,
      error: raw?.error || "The server returned an incomplete result.",
    };
  }
  return {
    filename: raw.filename || file.name,
    job_id: raw.job_id,
    status: raw.status,
    watermarks_found: raw.watermarks_found || 0,
    download_url: raw.download_url || null,
    error: raw.error,
  };
}

function StatusMark({ status }: { status: ResultStatus }) {
  if (status === "cleaned") {
    return <CheckCircleIcon size={15} weight="fill" className="text-[var(--accent)]" />;
  }
  if (status === "no_watermark") {
    return <WarningCircleIcon size={15} weight="fill" className="text-[var(--warning)]" />;
  }
  return <XCircleIcon size={15} weight="fill" className="text-[var(--danger)]" />;
}

function QueueStatus({ result }: { result?: ProcessResult }) {
  if (!result) {
    return <span className="font-mono text-[10px] uppercase tracking-[0.12em] text-[var(--ink-faint)]">Queued</span>;
  }
  return (
    <span className="flex items-center gap-1.5 text-[11px] text-[var(--ink-muted)]">
      <StatusMark status={result.status} />
      {result.status === "cleaned"
        ? `${result.watermarks_found} removed`
        : result.status === "no_watermark"
          ? "No mark found"
          : "Needs attention"}
    </span>
  );
}

function EmptyWorkbench() {
  return (
    <div className="flex min-h-[420px] flex-1 flex-col items-center justify-center px-6 py-16 text-center">
      <div className="mark-study mb-8" aria-hidden="true">
        <span className="mark-study__frame" />
        <span className="mark-study__core" />
        <SparkleIcon className="mark-study__spark" size={24} weight="regular" />
      </div>
      <p className="mb-2 text-sm font-semibold text-[var(--ink)]">Your comparison appears here</p>
      <p className="max-w-[36ch] text-sm leading-6 text-[var(--ink-muted)]">
        Add an image, run the cleaner, then drag across the result to inspect every repaired edge.
      </p>
    </div>
  );
}

function ProcessingWorkbench({ count }: { count: number }) {
  return (
    <div className="flex min-h-[420px] flex-1 flex-col p-5 sm:p-8" role="status" aria-live="polite">
      <div className="skeleton-preview flex-1 rounded-[1.75rem]" />
      <div className="mt-6 grid gap-3 sm:grid-cols-[1fr_auto] sm:items-end">
        <div>
          <div className="skeleton-line h-3 w-28 rounded-full" />
          <div className="skeleton-line mt-3 h-2.5 w-52 max-w-full rounded-full" />
        </div>
        <span className="font-mono text-[10px] uppercase tracking-[0.14em] text-[var(--ink-faint)]">
          Processing {count} {count === 1 ? "image" : "images"}
        </span>
      </div>
    </div>
  );
}

export default function App() {
  const [images, setImages] = useState<ImageEntry[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [processing, setProcessing] = useState(false);
  const [dragActive, setDragActive] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const imagesRef = useRef<ImageEntry[]>([]);

  useEffect(() => {
    imagesRef.current = images;
  }, [images]);

  useEffect(() => {
    return () => {
      imagesRef.current.forEach((entry) => URL.revokeObjectURL(entry.preview));
    };
  }, []);

  const addFiles = useCallback((fileList: FileList | File[]) => {
    const incoming = Array.from(fileList);
    const valid = incoming.filter(
      (file) => file.type.startsWith("image/") || acceptedExtensions.test(file.name),
    );

    if (valid.length !== incoming.length) {
      setMessage("Some files were skipped. Use PNG, JPEG, or WebP images.");
    } else {
      setMessage(null);
    }

    setImages((current) => {
      const known = new Set(
        current.map((entry) => `${entry.file.name}:${entry.file.size}:${entry.file.lastModified}`),
      );
      const next = valid
        .filter((file) => !known.has(`${file.name}:${file.size}:${file.lastModified}`))
        .map((file) => ({
          id: crypto.randomUUID(),
          file,
          preview: URL.createObjectURL(file),
        }));

      if (!selectedId && next[0]) setSelectedId(next[0].id);
      return [...current, ...next];
    });
  }, [selectedId]);

  const removeImage = (id: string) => {
    const index = images.findIndex((entry) => entry.id === id);
    const target = images[index];
    if (!target) return;
    URL.revokeObjectURL(target.preview);
    const next = images.filter((entry) => entry.id !== id);
    setImages(next);
    if (selectedId === id) {
      setSelectedId(next[Math.min(index, Math.max(0, next.length - 1))]?.id || null);
    }
  };

  const clearAll = () => {
    images.forEach((entry) => URL.revokeObjectURL(entry.preview));
    setImages([]);
    setSelectedId(null);
    setMessage(null);
  };

  const processAll = async () => {
    if (!images.length || processing) return;
    setProcessing(true);
    setMessage(null);

    const snapshot = images;
    const formData = new FormData();
    snapshot.forEach((entry) => formData.append("files", entry.file));

    try {
      const response = await fetch("/api/remove", { method: "POST", body: formData });
      if (!response.ok) throw new Error(`Server returned ${response.status}.`);
      const payload = (await response.json()) as { results?: Partial<ProcessResult>[] };
      if (!Array.isArray(payload.results)) throw new Error("The server response did not include results.");

      const byId = new Map(
        snapshot.map((entry, index) => [entry.id, normalizeResult(payload.results?.[index], entry.file)]),
      );
      setImages((current) =>
        current.map((entry) => ({ ...entry, result: byId.get(entry.id) || entry.result })),
      );

      const firstCleaned = snapshot.find((entry) => byId.get(entry.id)?.status === "cleaned");
      if (firstCleaned) setSelectedId(firstCleaned.id);

      const failed = [...byId.values()].filter((result) => result.status === "error").length;
      if (failed) setMessage(`${failed} ${failed === 1 ? "image needs" : "images need"} attention.`);
    } catch (error) {
      const detail = error instanceof Error ? error.message : "Connection failed.";
      setMessage(`Could not finish the batch. ${detail}`);
      const failedIds = new Set(snapshot.map((entry) => entry.id));
      setImages((current) =>
        current.map((entry) =>
          failedIds.has(entry.id)
            ? {
                ...entry,
                result: {
                  filename: entry.file.name,
                  status: "error",
                  watermarks_found: 0,
                  download_url: null,
                  error: detail,
                },
              }
            : entry,
        ),
      );
    } finally {
      setProcessing(false);
    }
  };

  const selected = images.find((entry) => entry.id === selectedId) || images[0] || null;
  const stats = {
    cleaned: images.filter((entry) => entry.result?.status === "cleaned").length,
    untouched: images.filter((entry) => entry.result?.status === "no_watermark").length,
    failed: images.filter((entry) => entry.result?.status === "error").length,
  };
  const hasResults = images.some((entry) => entry.result);

  return (
    <div className="app-shell min-h-[100dvh] bg-[var(--canvas)] text-[var(--ink)]">
      <header className="border-b border-[var(--line)]">
        <div className="mx-auto flex h-[72px] max-w-[1400px] items-center justify-between px-4 sm:px-6 lg:px-10">
          <a href="/" className="group flex items-center gap-3" aria-label="OpenNoMark home">
            <span className="brand-mark" aria-hidden="true"><span /></span>
            <span>
              <span className="block text-sm font-semibold tracking-[-0.02em]">OpenNoMark</span>
              <span className="block font-mono text-[9px] uppercase tracking-[0.16em] text-[var(--ink-faint)]">
                Local image repair
              </span>
            </span>
          </a>

          <div className="flex items-center gap-3 sm:gap-6">
            <span className="hidden items-center gap-2 text-xs text-[var(--ink-muted)] sm:flex">
              <ShieldCheckIcon size={16} weight="regular" className="text-[var(--accent)]" />
              Files stay on this server
            </span>
            <a
              href="https://github.com/NanmiCoder/OpenNoMark"
              target="_blank"
              rel="noreferrer"
              className="rounded-full border border-[var(--line)] px-3 py-2 font-mono text-[10px] uppercase tracking-[0.12em] text-[var(--ink-muted)] transition-colors hover:border-[var(--line-strong)] hover:text-[var(--ink)]"
            >
              Source
            </a>
          </div>
        </div>
      </header>

      <main className="mx-auto grid w-full max-w-[1400px] gap-10 px-4 py-8 sm:px-6 sm:py-12 lg:grid-cols-[minmax(300px,0.72fr)_minmax(0,1.45fr)] lg:gap-14 lg:px-10 lg:py-16">
        <section className="min-w-0">
          <div className="max-w-[560px]">
            <p className="mb-5 flex items-center gap-2 font-mono text-[10px] uppercase tracking-[0.18em] text-[var(--accent)]">
              <span className="h-px w-8 bg-[var(--accent)]" />
              Precision, not a blur brush
            </p>
            <h1 className="max-w-[11ch] text-4xl font-semibold leading-[0.98] tracking-[-0.055em] sm:text-5xl lg:text-6xl">
              Remove the mark. Keep the image.
            </h1>
            <p className="mt-6 max-w-[52ch] text-base leading-7 text-[var(--ink-muted)]">
              Purpose-built Gemini detection finds the sparkle by layout and edge shape. Local LaMa repair rebuilds only the marked pixels.
            </p>
          </div>

          <label
            className={`upload-surface mt-9 block cursor-pointer rounded-[1.75rem] border p-5 transition-colors duration-300 sm:p-6 ${
              dragActive ? "border-[var(--accent)] bg-[var(--accent-soft)]" : "border-[var(--line-strong)] bg-[var(--paper)]"
            } ${processing ? "pointer-events-none opacity-55" : ""}`}
            onDragOver={(event) => { event.preventDefault(); setDragActive(true); }}
            onDragLeave={() => setDragActive(false)}
            onDrop={(event) => {
              event.preventDefault();
              setDragActive(false);
              addFiles(event.dataTransfer.files);
            }}
          >
            <input
              type="file"
              multiple
              accept="image/png,image/jpeg,image/webp"
              className="sr-only"
              disabled={processing}
              onChange={(event) => {
                if (event.target.files) addFiles(event.target.files);
                event.currentTarget.value = "";
              }}
            />
            <span className="flex items-start justify-between gap-6">
              <span>
                <span className="flex h-11 w-11 items-center justify-center rounded-full border border-[var(--line)] bg-[var(--canvas)] text-[var(--accent)]">
                  <UploadSimpleIcon size={20} weight="regular" />
                </span>
                <span className="mt-8 block text-base font-semibold tracking-[-0.02em]">
                  Drop images here
                </span>
                <span className="mt-1 block text-sm leading-6 text-[var(--ink-muted)]">
                  or click to choose PNG, JPEG, and WebP files
                </span>
              </span>
              <span className="font-mono text-[10px] uppercase tracking-[0.14em] text-[var(--ink-faint)]">
                Batch ready
              </span>
            </span>
          </label>

          {message && (
            <div className="mt-4 flex items-start gap-2.5 border-l-2 border-[var(--danger)] py-1 pl-3 text-sm leading-6 text-[var(--ink-muted)]" role="alert">
              <WarningCircleIcon size={17} weight="fill" className="mt-1 shrink-0 text-[var(--danger)]" />
              <span>{message}</span>
            </div>
          )}

          <div className="mt-5 flex flex-wrap items-center gap-3">
            <MagneticButton onClick={processAll} disabled={!images.length || processing}>
              {processing ? "Repairing images" : hasResults ? "Run batch again" : "Remove watermarks"}
              {processing ? <SparkleIcon size={16} weight="regular" /> : <ArrowRightIcon size={16} weight="bold" />}
            </MagneticButton>
            {images.length > 0 && (
              <MagneticButton onClick={clearAll} disabled={processing} variant="quiet">
                <TrashIcon size={16} weight="regular" />
                Clear batch
              </MagneticButton>
            )}
          </div>

          <div className="mt-12 border-t border-[var(--line)] pt-5">
            <div className="flex items-end justify-between gap-4">
              <div>
                <p className="font-mono text-[10px] uppercase tracking-[0.16em] text-[var(--ink-faint)]">Queue</p>
                <p className="mt-1 text-sm text-[var(--ink-muted)]">
                  {images.length ? `${images.length} ${images.length === 1 ? "image" : "images"}` : "No images added"}
                </p>
              </div>
              {hasResults && (
                <div className="flex gap-3 font-mono text-[9px] uppercase tracking-[0.1em] text-[var(--ink-faint)]">
                  <span>{stats.cleaned} clean</span>
                  <span>{stats.untouched} unchanged</span>
                  {stats.failed > 0 && <span className="text-[var(--danger)]">{stats.failed} failed</span>}
                </div>
              )}
            </div>

            {images.length === 0 ? (
              <button
                type="button"
                className="mt-5 flex w-full items-center gap-3 border-y border-dashed border-[var(--line)] py-5 text-left text-sm text-[var(--ink-faint)]"
                onClick={() => document.querySelector<HTMLInputElement>('input[type="file"]')?.click()}
              >
                <PlusIcon size={17} weight="regular" />
                Add your first image to start a batch
              </button>
            ) : (
              <div className="mt-3 divide-y divide-[var(--line)] border-b border-[var(--line)]">
                {images.map((entry, index) => (
                  <div
                    key={entry.id}
                    className="queue-item grid grid-cols-[1fr_auto] items-center gap-2 py-2.5"
                    style={{ "--delay": `${index * 55}ms` } as CSSProperties}
                  >
                    <button
                      type="button"
                      onClick={() => setSelectedId(entry.id)}
                      className={`grid min-w-0 grid-cols-[44px_1fr] items-center gap-3 rounded-xl p-1.5 text-left transition-colors ${
                        selected?.id === entry.id ? "bg-[var(--paper)]" : "hover:bg-[var(--paper)]"
                      }`}
                      aria-pressed={selected?.id === entry.id}
                    >
                      <img src={entry.preview} alt="" className="h-11 w-11 rounded-lg object-cover" />
                      <span className="min-w-0">
                        <span className="block truncate text-sm font-medium tracking-[-0.01em]">{entry.file.name}</span>
                        <span className="mt-1 flex items-center gap-2">
                          <QueueStatus result={entry.result} />
                          <span className="text-[10px] text-[var(--ink-faint)]">{formatBytes(entry.file.size)}</span>
                        </span>
                      </span>
                    </button>
                    <button
                      type="button"
                      onClick={() => removeImage(entry.id)}
                      disabled={processing}
                      className="flex h-10 w-10 items-center justify-center rounded-full text-[var(--ink-faint)] transition-colors hover:bg-[var(--danger-soft)] hover:text-[var(--danger)] disabled:opacity-35"
                      aria-label={`Remove ${entry.file.name}`}
                    >
                      <TrashIcon size={16} weight="regular" />
                    </button>
                  </div>
                ))}
              </div>
            )}
          </div>
        </section>

        <section className="workbench-frame min-w-0 self-start overflow-hidden rounded-[2rem] border border-[var(--line)] bg-[var(--paper)] shadow-[0_32px_80px_-48px_rgba(44,51,47,0.48)] lg:sticky lg:top-8">
          <div className="flex min-h-[64px] items-center justify-between gap-4 border-b border-[var(--line)] px-5 py-3 sm:px-7">
            <div className="min-w-0">
              <p className="font-mono text-[9px] uppercase tracking-[0.17em] text-[var(--ink-faint)]">Inspection desk</p>
              <p className="mt-1 truncate text-sm font-medium">{selected?.file.name || "No image selected"}</p>
            </div>
            <span className="flex shrink-0 items-center gap-2 rounded-full border border-[var(--line)] px-3 py-1.5 font-mono text-[9px] uppercase tracking-[0.12em] text-[var(--ink-faint)]">
              <span className={`h-1.5 w-1.5 rounded-full ${processing ? "status-breathe bg-[var(--warning)]" : "bg-[var(--accent)]"}`} />
              {processing ? "Working" : "Ready"}
            </span>
          </div>

          {processing ? (
            <ProcessingWorkbench count={images.length} />
          ) : !selected ? (
            <EmptyWorkbench />
          ) : (
            <div className="p-4 sm:p-6 lg:p-7">
              {selected.result?.status === "cleaned" && selected.result.download_url ? (
                <CompareSlider
                  before={selected.preview}
                  after={selected.result.download_url}
                  filename={selected.file.name}
                />
              ) : (
                <div className="relative flex min-h-[360px] items-center justify-center overflow-hidden rounded-[1.75rem] bg-[var(--paper-deep)] p-3 sm:min-h-[460px]">
                  <img
                    src={selected.preview}
                    alt={selected.file.name}
                    className="max-h-[62vh] w-full rounded-2xl object-contain"
                  />
                  {!selected.result && (
                    <span className="absolute left-4 top-4 rounded-full border border-white/15 bg-[rgba(34,34,31,0.72)] px-3 py-1 font-mono text-[9px] uppercase tracking-[0.14em] text-white backdrop-blur-md">
                      Awaiting repair
                    </span>
                  )}
                </div>
              )}

              <div className="mt-6 grid gap-5 border-t border-[var(--line)] pt-5 sm:grid-cols-[1fr_auto] sm:items-center">
                <div className="min-w-0">
                  <div className="flex items-center gap-2">
                    {selected.result ? (
                      <StatusMark status={selected.result.status} />
                    ) : (
                      <ImageSquareIcon size={16} weight="regular" className="text-[var(--ink-faint)]" />
                    )}
                    <p className="text-sm font-semibold">
                      {!selected.result
                        ? "Ready for processing"
                        : selected.result.status === "cleaned"
                          ? "Watermark removed"
                          : selected.result.status === "no_watermark"
                            ? "No watermark detected"
                            : "Processing failed"}
                    </p>
                  </div>
                  <p className="mt-1 truncate text-xs leading-5 text-[var(--ink-muted)]">
                    {selected.result?.error || `${formatBytes(selected.file.size)} · Original preserved`}
                  </p>
                </div>

                {selected.result?.status === "cleaned" && selected.result.download_url ? (
                  <MagneticButton
                    variant="secondary"
                    onClick={() => window.location.assign(selected.result?.download_url || "")}
                  >
                    <DownloadSimpleIcon size={16} weight="regular" />
                    Download
                  </MagneticButton>
                ) : selected.result?.status === "error" ? (
                  <MagneticButton variant="secondary" onClick={processAll}>
                    <ArrowClockwiseIcon size={16} weight="regular" />
                    Retry batch
                  </MagneticButton>
                ) : (
                  <span className="flex items-center gap-2 font-mono text-[9px] uppercase tracking-[0.13em] text-[var(--ink-faint)]">
                    <ImagesIcon size={15} weight="regular" />
                    Batch-safe output
                  </span>
                )}
              </div>
            </div>
          )}
        </section>
      </main>

      <footer className="mx-auto flex w-full max-w-[1400px] flex-col gap-3 border-t border-[var(--line)] px-4 py-6 text-xs text-[var(--ink-faint)] sm:flex-row sm:items-center sm:justify-between sm:px-6 lg:px-10">
        <span>Gemini catalog detection · OWLv2 fallback · Local LaMa repair</span>
        <span className="font-mono text-[9px] uppercase tracking-[0.14em]">Open source · v0.2.0</span>
      </footer>
    </div>
  );
}
