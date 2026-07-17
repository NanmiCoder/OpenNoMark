import { useCallback, useEffect, useState } from "react";

export type Locale = "en" | "zh-CN";

const localeStorageKey = "opennomark.locale";

export interface Copy {
  documentTitle: string;
  brandHome: string;
  brandTagline: string;
  privacy: string;
  source: string;
  language: string;
  switchLanguage: string;
  eyebrow: string;
  heroTitle: string;
  heroDescription: string;
  dropTitle: string;
  dropHint: string;
  batchReady: string;
  skippedFiles: string;
  removeWatermarks: string;
  processNew: (count: number) => string;
  removeFromCount: (count: number) => string;
  tryAllAgain: string;
  processingCount: (current: number, total: number) => string;
  preparingZip: string;
  downloadAsZip: (count: number) => string;
  downloadResult: string;
  downloadReady: (count: number) => string;
  downloadReadyResult: string;
  processAllAgain: string;
  retryFailed: (count: number) => string;
  clearBatch: string;
  batchInProgress: string;
  batchComplete: string;
  nowProcessing: (filename: string) => string;
  resultsReady: (count: number) => string;
  queue: string;
  imageCount: (count: number) => string;
  noImages: string;
  cleanCount: (count: number) => string;
  unchangedCount: (count: number) => string;
  failedCount: (count: number) => string;
  addFirstImage: string;
  waitingInQueue: string;
  uploading: (progress: number) => string;
  removingWatermark: string;
  removedCount: (count: number) => string;
  noMarkReady: string;
  needsAttention: string;
  ready: string;
  comparisonEmptyTitle: string;
  comparisonEmptyDescription: string;
  waitingPrevious: string;
  uploadingOverlay: (progress: number) => string;
  rebuildingPixels: string;
  working: string;
  queued: string;
  processing: string;
  waiting: string;
  attention: string;
  inspectionDesk: string;
  noImageSelected: string;
  awaitingRepair: string;
  waitingInBatch: string;
  readyForProcessing: string;
  watermarkRemoved: string;
  noWatermarkDetected: string;
  processingFailed: string;
  originalPreserved: string;
  downloadThisImage: string;
  retryThisImage: string;
  progressComplete: (completed: number, total: number) => string;
  batchSafeOutput: string;
  downloadFile: (filename: string) => string;
  downloadThisResult: string;
  retryFile: (filename: string) => string;
  removeFile: (filename: string) => string;
  removeFromBatch: string;
  compareImages: (filename: string) => string;
  originalImage: (filename: string) => string;
  original: string;
  cleaned: string;
  footerPipeline: string;
  openSource: string;
  batchFailure: (count: number) => string;
  zipFailure: (detail: string) => string;
  errors: {
    connection: string;
    serverStatus: (status?: number) => string;
    missingResult: string;
    incompleteResult: string;
    processing: string;
    download: string;
    unknown: string;
  };
}

const en: Copy = {
  documentTitle: "OpenNoMark · Local watermark remover",
  brandHome: "OpenNoMark home",
  brandTagline: "Local image repair",
  privacy: "Files stay on this server",
  source: "Source",
  language: "Language",
  switchLanguage: "切换至中文",
  eyebrow: "Precision, not a blur brush",
  heroTitle: "Remove the mark. Keep the image.",
  heroDescription: "Upload one image or a whole set. Every file keeps its own status, result, retry, and download path.",
  dropTitle: "Drop images here",
  dropHint: "One image or many · PNG, JPEG, and WebP",
  batchReady: "Batch ready",
  skippedFiles: "Some files were skipped. Use PNG, JPEG, or WebP images.",
  removeWatermarks: "Remove watermarks",
  processNew: (count) => `Process ${count} new ${count === 1 ? "image" : "images"}`,
  removeFromCount: (count) => `Remove watermarks${count > 1 ? ` from ${count}` : ""}`,
  tryAllAgain: "Try all images again",
  processingCount: (current, total) => `Processing ${current} of ${total}`,
  preparingZip: "Preparing ZIP",
  downloadAsZip: (count) => `Download ${count} as ZIP`,
  downloadResult: "Download result",
  downloadReady: (count) => `Download ${count} ready`,
  downloadReadyResult: "Download ready result",
  processAllAgain: "Process all again",
  retryFailed: (count) => `Retry ${count} failed`,
  clearBatch: "Clear batch",
  batchInProgress: "Batch in progress",
  batchComplete: "Batch complete",
  nowProcessing: (filename) => `Now processing ${filename}`,
  resultsReady: (count) => `${count} ${count === 1 ? "result" : "results"} ready to download`,
  queue: "Queue",
  imageCount: (count) => `${count} ${count === 1 ? "image" : "images"}`,
  noImages: "No images added",
  cleanCount: (count) => `${count} clean`,
  unchangedCount: (count) => `${count} unchanged`,
  failedCount: (count) => `${count} failed`,
  addFirstImage: "Add your first image to start a batch",
  waitingInQueue: "Waiting in queue",
  uploading: (progress) => `Uploading ${progress}%`,
  removingWatermark: "Removing watermark",
  removedCount: (count) => `${count} removed`,
  noMarkReady: "No mark found · ready",
  needsAttention: "Needs attention",
  ready: "Ready",
  comparisonEmptyTitle: "Your comparison appears here",
  comparisonEmptyDescription: "Add an image, run the cleaner, then drag across the result to inspect every repaired edge.",
  waitingPrevious: "Waiting for the previous image",
  uploadingOverlay: (progress) => `Uploading · ${progress}%`,
  rebuildingPixels: "Detecting and rebuilding marked pixels",
  working: "Working",
  queued: "Queued",
  processing: "Processing",
  waiting: "Waiting",
  attention: "Attention",
  inspectionDesk: "Inspection desk",
  noImageSelected: "No image selected",
  awaitingRepair: "Awaiting repair",
  waitingInBatch: "Waiting in this batch",
  readyForProcessing: "Ready for processing",
  watermarkRemoved: "Watermark removed",
  noWatermarkDetected: "No watermark detected",
  processingFailed: "Processing failed",
  originalPreserved: "Original preserved",
  downloadThisImage: "Download this image",
  retryThisImage: "Retry this image",
  progressComplete: (completed, total) => `${completed} of ${total} complete`,
  batchSafeOutput: "Batch-safe output",
  downloadFile: (filename) => `Download ${filename}`,
  downloadThisResult: "Download this result",
  retryFile: (filename) => `Retry ${filename}`,
  removeFile: (filename) => `Remove ${filename}`,
  removeFromBatch: "Remove from batch",
  compareImages: (filename) => `Compare original and cleaned ${filename}`,
  originalImage: (filename) => `Original ${filename}`,
  original: "Original",
  cleaned: "Cleaned",
  footerPipeline: "Unified localization · Calibrated OWLv2 · Validated local LaMa",
  openSource: "Open source · v0.2.0",
  batchFailure: (count) => `${count} ${count === 1 ? "image needs" : "images need"} attention. Retry only the failed items when ready.`,
  zipFailure: (detail) => `Could not prepare the ZIP. ${detail}`,
  errors: {
    connection: "Connection failed while sending this image.",
    serverStatus: (status) => status ? `Server returned ${status}.` : "The server could not process this request.",
    missingResult: "The server response did not include a result.",
    incompleteResult: "The server returned an incomplete result.",
    processing: "This image could not be processed. Try again or use a different image.",
    download: "Download failed.",
    unknown: "Something went wrong. Please try again.",
  },
};

const zhCN: Copy = {
  documentTitle: "OpenNoMark · 本地去水印工具",
  brandHome: "OpenNoMark 首页",
  brandTagline: "本地图像修复",
  privacy: "文件仅在此服务器处理",
  source: "源码",
  language: "语言",
  switchLanguage: "Switch to English",
  eyebrow: "精准修复，而不是模糊涂抹",
  heroTitle: "去掉水印，保留原图质感。",
  heroDescription: "单张或批量上传均可。每张图片都有独立进度、处理结果、重试和下载入口。",
  dropTitle: "拖放图片到这里",
  dropHint: "支持单张或多张 · PNG、JPEG、WebP",
  batchReady: "支持批处理",
  skippedFiles: "部分文件已跳过，请使用 PNG、JPEG 或 WebP 图片。",
  removeWatermarks: "开始移除水印",
  processNew: (count) => `处理 ${count} 张新图片`,
  removeFromCount: (count) => count > 1 ? `移除 ${count} 张图片的水印` : "开始移除水印",
  tryAllAgain: "重新处理全部图片",
  processingCount: (current, total) => `正在处理 ${current}/${total}`,
  preparingZip: "正在生成压缩包",
  downloadAsZip: (count) => `打包下载 ${count} 张`,
  downloadResult: "下载结果",
  downloadReady: (count) => `下载已完成的 ${count} 张`,
  downloadReadyResult: "下载已完成的结果",
  processAllAgain: "重新处理全部",
  retryFailed: (count) => `重试 ${count} 个失败项`,
  clearBatch: "清空列表",
  batchInProgress: "批量处理中",
  batchComplete: "批量处理完成",
  nowProcessing: (filename) => `正在处理 ${filename}`,
  resultsReady: (count) => `${count} 个结果可下载`,
  queue: "图片列表",
  imageCount: (count) => `共 ${count} 张图片`,
  noImages: "尚未添加图片",
  cleanCount: (count) => `${count} 张已去除`,
  unchangedCount: (count) => `${count} 张无需处理`,
  failedCount: (count) => `${count} 张失败`,
  addFirstImage: "添加第一张图片，开始批量处理",
  waitingInQueue: "正在排队",
  uploading: (progress) => `上传中 ${progress}%`,
  removingWatermark: "正在移除水印",
  removedCount: (count) => `已移除 ${count} 处`,
  noMarkReady: "未发现水印 · 可下载",
  needsAttention: "需要处理",
  ready: "就绪",
  comparisonEmptyTitle: "处理前后对比将在这里显示",
  comparisonEmptyDescription: "添加图片并开始处理，完成后拖动分隔线即可检查每一处修复细节。",
  waitingPrevious: "等待上一张图片处理完成",
  uploadingOverlay: (progress) => `上传中 · ${progress}%`,
  rebuildingPixels: "正在检测并重建水印区域",
  working: "处理中",
  queued: "排队中",
  processing: "处理中",
  waiting: "等待中",
  attention: "需处理",
  inspectionDesk: "效果预览",
  noImageSelected: "未选择图片",
  awaitingRepair: "等待处理",
  waitingInBatch: "正在本批次中排队",
  readyForProcessing: "可以开始处理",
  watermarkRemoved: "水印已移除",
  noWatermarkDetected: "未检测到水印",
  processingFailed: "处理失败",
  originalPreserved: "原图已保留",
  downloadThisImage: "下载这张图片",
  retryThisImage: "重试这张图片",
  progressComplete: (completed, total) => `已完成 ${completed}/${total}`,
  batchSafeOutput: "批量结果独立保存",
  downloadFile: (filename) => `下载 ${filename}`,
  downloadThisResult: "下载此结果",
  retryFile: (filename) => `重试 ${filename}`,
  removeFile: (filename) => `移除 ${filename}`,
  removeFromBatch: "从列表中移除",
  compareImages: (filename) => `对比 ${filename} 的原图和处理结果`,
  originalImage: (filename) => `${filename} 原图`,
  original: "原图",
  cleaned: "处理后",
  footerPipeline: "统一水印定位 · 校准 OWLv2 · 本地 LaMa 修复复检",
  openSource: "开源项目 · v0.2.0",
  batchFailure: (count) => `${count} 张图片处理失败，可单独重试失败项。`,
  zipFailure: (detail) => `无法生成压缩包。${detail}`,
  errors: {
    connection: "图片发送失败，请检查网络和服务状态后重试。",
    serverStatus: (status) => status ? `服务器返回错误（${status}），请稍后重试。` : "服务器暂时无法处理该请求。",
    missingResult: "服务器响应中没有处理结果，请重试。",
    incompleteResult: "服务器返回的处理结果不完整，请重试。",
    processing: "这张图片处理失败，请重试或换一张图片。",
    download: "下载失败，请重试。",
    unknown: "出现未知问题，请重试。",
  },
};

export const translations: Record<Locale, Copy> = {
  en,
  "zh-CN": zhCN,
};

function browserLocale(): Locale {
  const preferred = navigator.languages?.[0] || navigator.language;
  return preferred?.toLowerCase().startsWith("zh") ? "zh-CN" : "en";
}

function initialLocale(): Locale {
  try {
    const saved = window.localStorage.getItem(localeStorageKey);
    if (saved === "en" || saved === "zh-CN") return saved;
  } catch {
    // Storage may be unavailable in a private browsing context.
  }
  return browserLocale();
}

export function useLocale() {
  const [locale, setLocaleState] = useState<Locale>(initialLocale);

  useEffect(() => {
    document.documentElement.lang = locale;
    document.title = translations[locale].documentTitle;
  }, [locale]);

  const setLocale = useCallback((next: Locale) => {
    setLocaleState(next);
    try {
      window.localStorage.setItem(localeStorageKey, next);
    } catch {
      // The selected language still applies for this session.
    }
  }, []);

  return {
    locale,
    setLocale,
    t: translations[locale],
  };
}
