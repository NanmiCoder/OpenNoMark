import { useCallback, useEffect, useRef, useState } from "react";
import type { CSSProperties } from "react";
import { AnimatePresence, motion } from "framer-motion";
import {
  ArrowClockwiseIcon,
  ArrowRightIcon,
  CheckCircleIcon,
  CircleNotchIcon,
  ClockCountdownIcon,
  DownloadSimpleIcon,
  FileZipIcon,
  GithubLogoIcon,
  ImageSquareIcon,
  ImagesIcon,
  PlusIcon,
  ShieldCheckIcon,
  SparkleIcon,
  TrashIcon,
  TranslateIcon,
  UploadSimpleIcon,
  WarningCircleIcon,
  XCircleIcon,
} from "@phosphor-icons/react";

import { CompareSlider } from "./components/CompareSlider";
import { MagneticButton } from "./components/MagneticButton";
import { useLocale } from "./i18n";
import type { Copy } from "./i18n";

type ResultStatus = "cleaned" | "no_watermark" | "error";
type TaskPhase = "ready" | "queued" | "uploading" | "processing" | "done" | "error";
type UiErrorCode = "connection" | "serverStatus" | "missingResult" | "incompleteResult" | "processing" | "download" | "unknown";

type UiMessage =
  | { kind: "skippedFiles" }
  | { kind: "batchFailure"; count: number }
  | { kind: "zipFailure"; error: UiError };

interface UiError {
  code: UiErrorCode;
  status?: number;
}

interface ProcessResult {
  filename: string;
  job_id?: string;
  status: ResultStatus;
  watermarks_found: number;
  download_url: string | null;
  error?: string;
  error_code?: UiErrorCode;
  error_status?: number;
}

interface ImageEntry {
  id: string;
  file: File;
  preview: string;
  phase: TaskPhase;
  uploadProgress: number;
  result?: ProcessResult;
}

interface BatchProgress {
  completed: number;
  total: number;
}

const acceptedExtensions = /\.(png|jpe?g|webp)$/i;
const activePhases = new Set<TaskPhase>(["queued", "uploading", "processing"]);

class RequestFailure extends Error {
  code: UiErrorCode;
  status?: number;

  constructor(code: UiErrorCode, status?: number) {
    super(code);
    this.name = "RequestFailure";
    this.code = code;
    this.status = status;
  }
}

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
      error: raw?.error,
      error_code: raw?.error ? "processing" : "incompleteResult",
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

function processImage(
  entry: ImageEntry,
  onProgress: (phase: "uploading" | "processing", progress: number) => void,
) {
  return new Promise<ProcessResult>((resolve, reject) => {
    const request = new XMLHttpRequest();
    const formData = new FormData();
    formData.append("files", entry.file);

    request.open("POST", "/api/remove");
    request.responseType = "json";
    request.upload.onloadstart = () => onProgress("uploading", 0);
    request.upload.onprogress = (event) => {
      const progress = event.lengthComputable ? Math.round((event.loaded / event.total) * 100) : 12;
      onProgress("uploading", progress);
    };
    request.upload.onload = () => onProgress("processing", 100);
    request.onerror = () => reject(new RequestFailure("connection"));
    request.onload = () => {
      if (request.status < 200 || request.status >= 300) {
        reject(new RequestFailure("serverStatus", request.status));
        return;
      }
      const payload = request.response as { results?: Partial<ProcessResult>[] } | null;
      if (!payload || !Array.isArray(payload.results)) {
        reject(new RequestFailure("missingResult"));
        return;
      }
      resolve(normalizeResult(payload.results[0], entry.file));
    };
    request.send(formData);
  });
}

function triggerDownload(url: string) {
  const link = document.createElement("a");
  link.href = url;
  link.download = "";
  document.body.appendChild(link);
  link.click();
  link.remove();
}

function errorCopy(t: Copy, error: UiError): string {
  if (error.code === "serverStatus") return t.errors.serverStatus(error.status);
  return t.errors[error.code];
}

function resultErrorCopy(t: Copy, result: ProcessResult): string {
  const code = result.error_code || "processing";
  return errorCopy(t, { code, status: result.error_status });
}

function messageCopy(t: Copy, message: UiMessage): string {
  if (message.kind === "skippedFiles") return t.skippedFiles;
  if (message.kind === "batchFailure") return t.batchFailure(message.count);
  return t.zipFailure(errorCopy(t, message.error));
}

function StatusMark({ status }: { status: ResultStatus }) {
  if (status === "cleaned") {
    return <CheckCircleIcon size={15} weight="fill" className="text-[var(--accent)]" />;
  }
  if (status === "no_watermark") {
    return <CheckCircleIcon size={15} weight="fill" className="text-[var(--ink-faint)]" />;
  }
  return <XCircleIcon size={15} weight="fill" className="text-[var(--danger)]" />;
}

function QueueStatus({ entry, t }: { entry: ImageEntry; t: Copy }) {
  if (entry.phase === "queued") {
    return (
      <span className="flex items-center gap-1.5 text-[11px] text-[var(--ink-muted)]">
        <ClockCountdownIcon size={14} weight="regular" /> {t.waitingInQueue}
      </span>
    );
  }
  if (entry.phase === "uploading") {
    return (
      <span className="flex items-center gap-1.5 text-[11px] text-[var(--ink-muted)]">
        <UploadSimpleIcon size={14} weight="regular" /> {t.uploading(entry.uploadProgress)}
      </span>
    );
  }
  if (entry.phase === "processing") {
    return (
      <span className="flex items-center gap-1.5 text-[11px] font-medium text-[var(--accent)]">
        <CircleNotchIcon size={14} weight="bold" className="task-spinner" /> {t.removingWatermark}
      </span>
    );
  }
  if (entry.result) {
    return (
      <span className="flex items-center gap-1.5 text-[11px] text-[var(--ink-muted)]">
        <StatusMark status={entry.result.status} />
        {entry.result.status === "cleaned"
          ? t.removedCount(entry.result.watermarks_found)
          : entry.result.status === "no_watermark"
            ? t.noMarkReady
            : t.needsAttention}
      </span>
    );
  }
  return (
    <span className="font-mono text-[10px] uppercase tracking-[0.12em] text-[var(--ink-faint)]">
      {t.ready}
    </span>
  );
}

function TaskProgress({ entry }: { entry: ImageEntry }) {
  const visible = activePhases.has(entry.phase) || entry.phase === "done" || entry.phase === "error";
  if (!visible) return null;

  const scale =
    entry.phase === "done" ? 1 : entry.phase === "error" ? 1 : entry.phase === "uploading" ? entry.uploadProgress / 100 : 0;

  return (
    <span className="task-progress" aria-hidden="true">
      {entry.phase === "processing" ? (
        <span className="task-progress__indeterminate" />
      ) : (
        <span
          className={`task-progress__fill ${entry.phase === "error" ? "task-progress__fill--error" : ""}`}
          style={{ "--task-scale": scale } as CSSProperties}
        />
      )}
    </span>
  );
}

function EmptyWorkbench({ t }: { t: Copy }) {
  return (
    <div className="flex min-h-[420px] flex-1 flex-col items-center justify-center px-6 py-16 text-center">
      <div className="mark-study mb-8" aria-hidden="true">
        <span className="mark-study__frame" />
        <span className="mark-study__core" />
        <SparkleIcon className="mark-study__spark" size={24} weight="regular" />
      </div>
      <p className="mb-2 text-sm font-semibold text-[var(--ink)]">{t.comparisonEmptyTitle}</p>
      <p className="max-w-[36ch] text-sm leading-6 text-[var(--ink-muted)]">
        {t.comparisonEmptyDescription}
      </p>
    </div>
  );
}

function SelectedTaskOverlay({ entry, t }: { entry: ImageEntry; t: Copy }) {
  if (!activePhases.has(entry.phase)) return null;

  const copy =
    entry.phase === "queued"
      ? t.waitingPrevious
      : entry.phase === "uploading"
        ? t.uploadingOverlay(entry.uploadProgress)
        : t.rebuildingPixels;

  return (
    <div className="pointer-events-none absolute inset-x-4 top-4 flex items-center justify-between gap-3 rounded-2xl border border-white/15 bg-[rgba(34,34,31,0.78)] px-4 py-3 text-white shadow-[inset_0_1px_0_rgba(255,255,255,0.12)] backdrop-blur-md">
      <span className="flex min-w-0 items-center gap-2 text-xs font-medium">
        {entry.phase === "processing" ? (
          <CircleNotchIcon size={15} weight="bold" className="task-spinner shrink-0" />
        ) : (
          <ClockCountdownIcon size={15} weight="regular" className="shrink-0" />
        )}
        <span className="truncate">{copy}</span>
      </span>
      <span className="shrink-0 font-mono text-[9px] uppercase tracking-[0.13em] text-white/65">
        {entry.phase === "processing" ? t.working : t.queued}
      </span>
    </div>
  );
}

export default function App() {
  const { locale, setLocale, t } = useLocale();
  const [images, setImages] = useState<ImageEntry[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [processing, setProcessing] = useState(false);
  const [activeId, setActiveId] = useState<string | null>(null);
  const [batchProgress, setBatchProgress] = useState<BatchProgress>({ completed: 0, total: 0 });
  const [downloadingBatch, setDownloadingBatch] = useState(false);
  const [dragActive, setDragActive] = useState(false);
  const [message, setMessage] = useState<UiMessage | null>(null);
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
      setMessage({ kind: "skippedFiles" });
    } else {
      setMessage(null);
    }
    setBatchProgress({ completed: 0, total: 0 });

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
          phase: "ready" as const,
          uploadProgress: 0,
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
    setBatchProgress({ completed: 0, total: 0 });
    if (selectedId === id) {
      setSelectedId(next[Math.min(index, Math.max(0, next.length - 1))]?.id || null);
    }
  };

  const clearAll = () => {
    images.forEach((entry) => URL.revokeObjectURL(entry.preview));
    setImages([]);
    setSelectedId(null);
    setMessage(null);
    setBatchProgress({ completed: 0, total: 0 });
  };

  const processEntries = async (ids?: string[]) => {
    if (!images.length || processing) return;
    const requested = ids ? new Set(ids) : null;
    const targets = images.filter((entry) => !requested || requested.has(entry.id));
    if (!targets.length) return;

    const targetIds = new Set(targets.map((entry) => entry.id));
    setProcessing(true);
    setMessage(null);
    setBatchProgress({ completed: 0, total: targets.length });
    setSelectedId((current) => current || targets[0].id);
    setImages((current) =>
      current.map((entry) =>
        targetIds.has(entry.id)
          ? { ...entry, phase: "queued", uploadProgress: 0, result: undefined }
          : entry,
      ),
    );

    let failures = 0;
    for (let index = 0; index < targets.length; index += 1) {
      const target = targets[index];
      setActiveId(target.id);
      try {
        const result = await processImage(target, (phase, uploadProgress) => {
          setImages((current) =>
            current.map((entry) =>
              entry.id === target.id ? { ...entry, phase, uploadProgress } : entry,
            ),
          );
        });
        const phase = result.status === "error" ? "error" : "done";
        if (phase === "error") failures += 1;
        setImages((current) =>
          current.map((entry) =>
            entry.id === target.id ? { ...entry, phase, uploadProgress: 100, result } : entry,
          ),
        );
      } catch (error) {
        failures += 1;
        const failure = error instanceof RequestFailure
          ? error
          : new RequestFailure("unknown");
        setImages((current) =>
          current.map((entry) =>
            entry.id === target.id
              ? {
                  ...entry,
                  phase: "error",
                  result: {
                    filename: entry.file.name,
                    status: "error",
                    watermarks_found: 0,
                    download_url: null,
                    error_code: failure.code,
                    error_status: failure.status,
                  },
                }
              : entry,
          ),
        );
      }
      setBatchProgress({ completed: index + 1, total: targets.length });
    }

    setActiveId(null);
    setProcessing(false);
    if (failures) {
      setMessage({ kind: "batchFailure", count: failures });
    }
  };

  const selected = images.find((entry) => entry.id === selectedId) || images[0] || null;
  const activeEntry = images.find((entry) => entry.id === activeId) || null;
  const downloadable = images.filter(
    (entry) => entry.result?.download_url && entry.result.job_id && entry.result.status !== "error",
  );
  const pendingIds = images.filter((entry) => entry.phase === "ready").map((entry) => entry.id);
  const failedIds = images.filter((entry) => entry.phase === "error").map((entry) => entry.id);
  const stats = {
    cleaned: images.filter((entry) => entry.result?.status === "cleaned").length,
    untouched: images.filter((entry) => entry.result?.status === "no_watermark").length,
    failed: failedIds.length,
  };
  const hasResults = images.some((entry) => entry.result);
  const completedPercent = batchProgress.total
    ? Math.round((batchProgress.completed / batchProgress.total) * 100)
    : 0;
  const processPrimary = processing || pendingIds.length > 0 || downloadable.length === 0;
  const processLabel = !images.length
    ? t.removeWatermarks
    : pendingIds.length
      ? pendingIds.length < images.length
        ? t.processNew(pendingIds.length)
        : t.removeFromCount(images.length)
      : t.tryAllAgain;

  const downloadBatch = async () => {
    if (!downloadable.length || downloadingBatch) return;
    if (downloadable.length === 1) {
      triggerDownload(downloadable[0].result?.download_url || "");
      return;
    }

    setDownloadingBatch(true);
    setMessage(null);
    try {
      const response = await fetch("/api/download-batch", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          items: downloadable.map((entry) => ({
            job_id: entry.result?.job_id,
            filename: entry.file.name,
          })),
        }),
      });
      if (!response.ok) throw new RequestFailure("serverStatus", response.status);
      const blob = await response.blob();
      const url = URL.createObjectURL(blob);
      triggerDownload(url);
      window.setTimeout(() => URL.revokeObjectURL(url), 1000);
    } catch (error) {
      const failure = error instanceof RequestFailure
        ? error
        : new RequestFailure("download");
      setMessage({
        kind: "zipFailure",
        error: { code: failure.code, status: failure.status },
      });
    } finally {
      setDownloadingBatch(false);
    }
  };

  const selectedBusy = selected ? activePhases.has(selected.phase) : false;
  const inspectionLabel = selectedBusy
    ? selected.phase === "processing"
      ? t.processing
      : t.waiting
    : selected?.phase === "error"
      ? t.attention
      : t.ready;

  return (
    <div className="app-shell min-h-[100dvh] bg-[var(--canvas)] text-[var(--ink)]">
      <header className="border-b border-[var(--line)]">
        <div className="mx-auto flex h-[72px] max-w-[1400px] items-center justify-between px-4 sm:px-6 lg:px-10">
          <a href="/" className="group flex items-center gap-3" aria-label={t.brandHome}>
            <span className="brand-mark" aria-hidden="true"><span /></span>
            <span>
              <span className="block text-sm font-semibold tracking-[-0.02em]">OpenNoMark</span>
              <span className="block font-mono text-[9px] uppercase tracking-[0.16em] text-[var(--ink-faint)]">
                {t.brandTagline}
              </span>
            </span>
          </a>

          <div className="flex items-center gap-2 sm:gap-3">
            <button
              type="button"
              className="inline-flex min-h-10 items-center gap-2 rounded-full border border-[var(--line)] bg-[var(--paper)] px-3 font-mono text-[10px] font-medium uppercase tracking-[0.1em] text-[var(--ink-muted)] transition-colors hover:border-[var(--line-strong)] hover:text-[var(--ink)] active:scale-[0.98]"
              onClick={() => setLocale(locale === "en" ? "zh-CN" : "en")}
              aria-label={t.switchLanguage}
              title={t.switchLanguage}
            >
              <TranslateIcon size={15} weight="regular" aria-hidden="true" />
              <span aria-hidden="true">{locale === "en" ? "中文" : "EN"}</span>
              <span className="sr-only">{t.language}</span>
            </button>
            <a
              href="https://github.com/NanmiCoder/OpenNoMark"
              target="_blank"
              rel="noreferrer"
              aria-label={t.source}
              title={t.source}
              className="inline-flex min-h-10 items-center gap-2 rounded-full border border-[var(--line)] bg-[var(--paper)] px-3 font-mono text-[10px] font-medium uppercase tracking-[0.1em] text-[var(--ink-muted)] transition-colors hover:border-[var(--line-strong)] hover:text-[var(--ink)] active:scale-[0.98]"
            >
              <GithubLogoIcon size={16} weight="fill" aria-hidden="true" />
              <span className="hidden sm:inline">{t.source}</span>
            </a>
          </div>
        </div>
      </header>

      <main className="mx-auto grid w-full max-w-[1400px] gap-10 px-4 py-8 sm:px-6 sm:py-12 lg:grid-cols-[minmax(320px,0.78fr)_minmax(0,1.42fr)] lg:gap-14 lg:px-10 lg:py-16">
        <section className="min-w-0">
          <div className="max-w-[560px]">
            <p className="mb-5 flex items-center gap-2 font-mono text-[10px] uppercase tracking-[0.18em] text-[var(--accent)]">
              <span className="h-px w-8 bg-[var(--accent)]" />
              {t.eyebrow}
            </p>
            <h1 className="max-w-[11ch] text-4xl font-semibold leading-[0.98] tracking-[-0.055em] sm:text-5xl lg:text-6xl">
              {t.heroTitle}
            </h1>
            <p className="mt-6 max-w-[52ch] text-base leading-7 text-[var(--ink-muted)]">
              {t.heroDescription}
            </p>
            <p className="mt-4 flex items-center gap-2 text-xs font-medium text-[var(--ink-muted)]">
              <ShieldCheckIcon size={16} weight="regular" className="shrink-0 text-[var(--accent)]" aria-hidden="true" />
              {t.privacy}
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
                  {t.dropTitle}
                </span>
                <span className="mt-1 block text-sm leading-6 text-[var(--ink-muted)]">
                  {t.dropHint}
                </span>
              </span>
              <span className="font-mono text-[10px] uppercase tracking-[0.14em] text-[var(--ink-faint)]">
                {t.batchReady}
              </span>
            </span>
          </label>

          {message && (
            <div className="mt-4 flex items-start gap-2.5 border-l-2 border-[var(--danger)] py-1 pl-3 text-sm leading-6 text-[var(--ink-muted)]" role="alert">
              <WarningCircleIcon size={17} weight="fill" className="mt-1 shrink-0 text-[var(--danger)]" />
              <span>{messageCopy(t, message)}</span>
            </div>
          )}

          <div className="mt-5 flex flex-wrap items-center gap-3">
            {processPrimary ? (
              <MagneticButton
                onClick={() => processEntries(pendingIds.length ? pendingIds : undefined)}
                disabled={!images.length || processing}
              >
                {processing
                  ? t.processingCount(Math.min(batchProgress.completed + 1, batchProgress.total), batchProgress.total)
                  : processLabel}
                {processing ? <CircleNotchIcon size={16} weight="bold" className="task-spinner" /> : <ArrowRightIcon size={16} weight="bold" />}
              </MagneticButton>
            ) : (
              <MagneticButton onClick={downloadBatch} disabled={downloadingBatch}>
                {downloadable.length > 1 ? <FileZipIcon size={17} weight="regular" /> : <DownloadSimpleIcon size={17} weight="regular" />}
                {downloadingBatch
                  ? t.preparingZip
                  : downloadable.length > 1
                    ? t.downloadAsZip(downloadable.length)
                    : t.downloadResult}
              </MagneticButton>
            )}

            {pendingIds.length > 0 && downloadable.length > 0 && !processing && (
              <MagneticButton onClick={downloadBatch} disabled={downloadingBatch} variant="secondary">
                {downloadable.length > 1 ? <FileZipIcon size={17} weight="regular" /> : <DownloadSimpleIcon size={17} weight="regular" />}
                {downloadable.length > 1 ? t.downloadReady(downloadable.length) : t.downloadReadyResult}
              </MagneticButton>
            )}
            {downloadable.length > 0 && pendingIds.length === 0 && !processing && (
              <MagneticButton onClick={() => processEntries()} variant="secondary">
                <ArrowClockwiseIcon size={16} weight="regular" />
                {t.processAllAgain}
              </MagneticButton>
            )}
            {failedIds.length > 0 && !processing && (
              <MagneticButton onClick={() => processEntries(failedIds)} variant="secondary">
                <ArrowClockwiseIcon size={16} weight="regular" />
                {t.retryFailed(failedIds.length)}
              </MagneticButton>
            )}
            {images.length > 0 && (
              <MagneticButton onClick={clearAll} disabled={processing} variant="quiet">
                <TrashIcon size={16} weight="regular" />
                {t.clearBatch}
              </MagneticButton>
            )}
          </div>

          {(processing || (batchProgress.total > 0 && hasResults)) && (
            <div className="batch-progress mt-6 rounded-2xl border border-[var(--line)] bg-[var(--paper)] px-4 py-3.5" role="status" aria-live="polite">
              <div className="flex items-center justify-between gap-4">
                <span className="min-w-0">
                  <span className="block font-mono text-[9px] uppercase tracking-[0.15em] text-[var(--ink-faint)]">
                    {processing ? t.batchInProgress : t.batchComplete}
                  </span>
                  <span className="mt-1 block truncate text-xs font-medium text-[var(--ink-muted)]">
                    {processing && activeEntry
                      ? t.nowProcessing(activeEntry.file.name)
                      : t.resultsReady(downloadable.length)}
                  </span>
                </span>
                <span className="shrink-0 font-mono text-xs font-semibold text-[var(--ink)]">
                  {batchProgress.completed}/{batchProgress.total}
                </span>
              </div>
              <div className="batch-progress__track mt-3" aria-hidden="true">
                <span
                  className="batch-progress__fill"
                  style={{ "--batch-scale": completedPercent / 100 } as CSSProperties}
                />
              </div>
            </div>
          )}

          <div className="mt-12 border-t border-[var(--line)] pt-5">
            <div className="flex items-end justify-between gap-4">
              <div>
                <p className="font-mono text-[10px] uppercase tracking-[0.16em] text-[var(--ink-faint)]">{t.queue}</p>
                <p className="mt-1 text-sm text-[var(--ink-muted)]">
                  {images.length ? t.imageCount(images.length) : t.noImages}
                </p>
              </div>
              {hasResults && (
                <div className="flex flex-wrap justify-end gap-3 font-mono text-[9px] uppercase tracking-[0.1em] text-[var(--ink-faint)]">
                  <span>{t.cleanCount(stats.cleaned)}</span>
                  <span>{t.unchangedCount(stats.untouched)}</span>
                  {stats.failed > 0 && <span className="text-[var(--danger)]">{t.failedCount(stats.failed)}</span>}
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
                {t.addFirstImage}
              </button>
            ) : (
              <div className="mt-3 divide-y divide-[var(--line)] border-b border-[var(--line)]">
                <AnimatePresence initial={false}>
                  {images.map((entry) => (
                    <motion.div
                      layout
                      key={entry.id}
                      initial={{ opacity: 0, y: 8 }}
                      animate={{ opacity: 1, y: 0 }}
                      exit={{ opacity: 0, x: -12 }}
                      transition={{ type: "spring", stiffness: 180, damping: 22 }}
                      className="queue-item grid grid-cols-[1fr_auto] items-center gap-2 py-2.5"
                    >
                      <button
                        type="button"
                        onClick={() => setSelectedId(entry.id)}
                        className={`grid min-w-0 grid-cols-[44px_1fr] items-center gap-3 rounded-xl p-1.5 text-left transition-colors ${
                          selected?.id === entry.id ? "bg-[var(--paper)]" : "hover:bg-[var(--paper)]"
                        }`}
                        aria-pressed={selected?.id === entry.id}
                      >
                        <span className="relative">
                          <img src={entry.preview} alt="" className="h-11 w-11 rounded-lg object-cover" />
                          {entry.phase === "done" && (
                            <span className="absolute -bottom-1 -right-1 grid h-4 w-4 place-items-center rounded-full bg-[var(--paper)] text-[var(--accent)]">
                              <CheckCircleIcon size={14} weight="fill" />
                            </span>
                          )}
                        </span>
                        <span className="min-w-0">
                          <span className="block truncate text-sm font-medium tracking-[-0.01em]">{entry.file.name}</span>
                          <span className="mt-1 flex min-w-0 items-center gap-2">
                            <QueueStatus entry={entry} t={t} />
                            <span className="shrink-0 text-[10px] text-[var(--ink-faint)]">{formatBytes(entry.file.size)}</span>
                          </span>
                          <TaskProgress entry={entry} />
                        </span>
                      </button>

                      <div className="flex items-center">
                        {entry.result?.download_url && entry.phase === "done" && (
                          <button
                            type="button"
                            onClick={() => triggerDownload(entry.result?.download_url || "")}
                            className="queue-action text-[var(--ink-faint)] hover:bg-[var(--accent-soft)] hover:text-[var(--accent)]"
                            aria-label={t.downloadFile(entry.file.name)}
                            title={t.downloadThisResult}
                          >
                            <DownloadSimpleIcon size={16} weight="regular" />
                          </button>
                        )}
                        {entry.phase === "error" && (
                          <button
                            type="button"
                            onClick={() => processEntries([entry.id])}
                            disabled={processing}
                            className="queue-action text-[var(--danger)] hover:bg-[var(--danger-soft)] disabled:opacity-35"
                            aria-label={t.retryFile(entry.file.name)}
                            title={t.retryThisImage}
                          >
                            <ArrowClockwiseIcon size={16} weight="regular" />
                          </button>
                        )}
                        <button
                          type="button"
                          onClick={() => removeImage(entry.id)}
                          disabled={processing}
                          className="queue-action text-[var(--ink-faint)] hover:bg-[var(--danger-soft)] hover:text-[var(--danger)] disabled:opacity-35"
                          aria-label={t.removeFile(entry.file.name)}
                          title={t.removeFromBatch}
                        >
                          <TrashIcon size={16} weight="regular" />
                        </button>
                      </div>
                    </motion.div>
                  ))}
                </AnimatePresence>
              </div>
            )}
          </div>
        </section>

        <section className="workbench-frame min-w-0 self-start overflow-hidden rounded-[2rem] border border-[var(--line)] bg-[var(--paper)] shadow-[0_32px_80px_-48px_rgba(44,51,47,0.48)] lg:sticky lg:top-8">
          <div className="flex min-h-[64px] items-center justify-between gap-4 border-b border-[var(--line)] px-5 py-3 sm:px-7">
            <div className="min-w-0">
              <p className="font-mono text-[9px] uppercase tracking-[0.17em] text-[var(--ink-faint)]">{t.inspectionDesk}</p>
              <p className="mt-1 truncate text-sm font-medium">{selected?.file.name || t.noImageSelected}</p>
            </div>
            <span className="flex shrink-0 items-center gap-2 rounded-full border border-[var(--line)] px-3 py-1.5 font-mono text-[9px] uppercase tracking-[0.12em] text-[var(--ink-faint)]">
              <span className={`h-1.5 w-1.5 rounded-full ${selectedBusy ? "status-breathe bg-[var(--warning)]" : selected?.phase === "error" ? "bg-[var(--danger)]" : "bg-[var(--accent)]"}`} />
              {inspectionLabel}
            </span>
          </div>

          {!selected ? (
            <EmptyWorkbench t={t} />
          ) : (
            <div className="p-4 sm:p-6 lg:p-7">
              {selected.result?.status === "cleaned" && selected.result.download_url && selected.phase === "done" ? (
                <CompareSlider
                  before={selected.preview}
                  after={selected.result.download_url}
                  filename={selected.file.name}
                  copy={t}
                />
              ) : (
                <div className={`relative flex min-h-[360px] items-center justify-center overflow-hidden rounded-[1.75rem] bg-[var(--paper-deep)] p-3 sm:min-h-[460px] ${selected.phase === "processing" ? "image-processing" : ""}`}>
                  <img
                    src={selected.preview}
                    alt={selected.file.name}
                    className="max-h-[62vh] w-full rounded-2xl object-contain"
                  />
                  <SelectedTaskOverlay entry={selected} t={t} />
                  {selected.phase === "ready" && (
                    <span className="absolute left-4 top-4 rounded-full border border-white/15 bg-[rgba(34,34,31,0.72)] px-3 py-1 font-mono text-[9px] uppercase tracking-[0.14em] text-white backdrop-blur-md">
                      {t.awaitingRepair}
                    </span>
                  )}
                </div>
              )}

              <div className="mt-6 grid gap-5 border-t border-[var(--line)] pt-5 sm:grid-cols-[1fr_auto] sm:items-center">
                <div className="min-w-0">
                  <div className="flex items-center gap-2">
                    {selected.result ? (
                      <StatusMark status={selected.result.status} />
                    ) : selectedBusy ? (
                      <CircleNotchIcon size={16} weight="bold" className="task-spinner text-[var(--accent)]" />
                    ) : (
                      <ImageSquareIcon size={16} weight="regular" className="text-[var(--ink-faint)]" />
                    )}
                    <p className="text-sm font-semibold">
                      {selectedBusy
                        ? selected.phase === "processing"
                          ? t.removingWatermark
                          : t.waitingInBatch
                        : !selected.result
                          ? t.readyForProcessing
                          : selected.result.status === "cleaned"
                            ? t.watermarkRemoved
                            : selected.result.status === "no_watermark"
                              ? t.noWatermarkDetected
                              : t.processingFailed}
                    </p>
                  </div>
                  <p className="mt-1 truncate text-xs leading-5 text-[var(--ink-muted)]">
                    {selected.result?.status === "error"
                      ? resultErrorCopy(t, selected.result)
                      : `${formatBytes(selected.file.size)} · ${t.originalPreserved}`}
                  </p>
                </div>

                {selected.result?.download_url && selected.phase === "done" ? (
                  <MagneticButton
                    variant="secondary"
                    onClick={() => triggerDownload(selected.result?.download_url || "")}
                  >
                    <DownloadSimpleIcon size={16} weight="regular" />
                    {t.downloadThisImage}
                  </MagneticButton>
                ) : selected.phase === "error" ? (
                  <MagneticButton variant="secondary" onClick={() => processEntries([selected.id])} disabled={processing}>
                    <ArrowClockwiseIcon size={16} weight="regular" />
                    {t.retryThisImage}
                  </MagneticButton>
                ) : (
                  <span className="flex items-center gap-2 font-mono text-[9px] uppercase tracking-[0.13em] text-[var(--ink-faint)]">
                    <ImagesIcon size={15} weight="regular" />
                    {processing ? t.progressComplete(batchProgress.completed, batchProgress.total) : t.batchSafeOutput}
                  </span>
                )}
              </div>
            </div>
          )}
        </section>
      </main>

      <footer className="mx-auto flex w-full max-w-[1400px] flex-col gap-3 border-t border-[var(--line)] px-4 py-6 text-xs text-[var(--ink-faint)] sm:flex-row sm:items-center sm:justify-between sm:px-6 lg:px-10">
        <span>{t.footerPipeline}</span>
        <span className="font-mono text-[9px] uppercase tracking-[0.14em]">{t.openSource}</span>
      </footer>
    </div>
  );
}
