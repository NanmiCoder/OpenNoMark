import { useCallback, useRef, useState } from "react";
import { CaretLeftIcon, CaretRightIcon } from "@phosphor-icons/react";

interface CompareSliderProps {
  before: string;
  after: string;
  filename: string;
}

export function CompareSlider({ before, after, filename }: CompareSliderProps) {
  const frameRef = useRef<HTMLDivElement>(null);
  const [split, setSplit] = useState(50);

  const setFromClientX = useCallback((clientX: number) => {
    if (!frameRef.current) return;
    const bounds = frameRef.current.getBoundingClientRect();
    const next = ((clientX - bounds.left) / bounds.width) * 100;
    setSplit(Math.max(3, Math.min(97, next)));
  }, []);

  const handlePointerDown = useCallback(
    (event: React.PointerEvent<HTMLDivElement>) => {
      event.preventDefault();
      event.currentTarget.setPointerCapture(event.pointerId);
      setFromClientX(event.clientX);
    },
    [setFromClientX],
  );

  const handlePointerMove = useCallback(
    (event: React.PointerEvent<HTMLDivElement>) => {
      if (!event.currentTarget.hasPointerCapture(event.pointerId)) return;
      setFromClientX(event.clientX);
    },
    [setFromClientX],
  );

  const handleKeyDown = (event: React.KeyboardEvent<HTMLDivElement>) => {
    if (event.key !== "ArrowLeft" && event.key !== "ArrowRight") return;
    event.preventDefault();
    setSplit((current) =>
      Math.max(3, Math.min(97, current + (event.key === "ArrowRight" ? 4 : -4))),
    );
  };

  return (
    <div
      ref={frameRef}
      className="compare-frame group relative overflow-hidden rounded-[1.75rem] bg-[var(--paper-deep)] select-none"
      style={{ "--split": `${split}%` } as React.CSSProperties}
      onPointerDown={handlePointerDown}
      onPointerMove={handlePointerMove}
      onKeyDown={handleKeyDown}
      role="slider"
      tabIndex={0}
      aria-label={`Compare original and cleaned ${filename}`}
      aria-valuemin={0}
      aria-valuemax={100}
      aria-valuenow={Math.round(split)}
    >
      <img
        src={before}
        alt={`Original ${filename}`}
        className="block max-h-[66vh] w-full object-contain"
        draggable={false}
      />

      <div className="compare-after" aria-hidden="true">
        <img
          src={after}
          alt=""
          className="block max-h-[66vh] w-full object-contain"
          draggable={false}
        />
      </div>

      <div className="compare-divider" aria-hidden="true">
        <div className="compare-handle">
          <CaretLeftIcon size={14} weight="bold" />
          <CaretRightIcon size={14} weight="bold" />
        </div>
      </div>

      <span className="absolute left-4 top-4 rounded-full border border-white/15 bg-[rgba(34,34,31,0.72)] px-3 py-1 font-mono text-[10px] uppercase tracking-[0.16em] text-white backdrop-blur-md">
        Original
      </span>
      <span className="absolute right-4 top-4 rounded-full border border-white/15 bg-[rgba(34,34,31,0.72)] px-3 py-1 font-mono text-[10px] uppercase tracking-[0.16em] text-white backdrop-blur-md">
        Cleaned
      </span>
    </div>
  );
}
