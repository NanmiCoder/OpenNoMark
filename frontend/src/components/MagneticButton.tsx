import type { MouseEventHandler, PointerEvent, ReactNode } from "react";
import {
  motion,
  useMotionValue,
  useReducedMotion,
  useSpring,
  useTransform,
} from "framer-motion";

type ButtonVariant = "primary" | "secondary" | "quiet";

interface MagneticButtonProps {
  children: ReactNode;
  onClick?: MouseEventHandler<HTMLButtonElement>;
  disabled?: boolean;
  variant?: ButtonVariant;
  className?: string;
  type?: "button" | "submit";
  ariaLabel?: string;
}

const variants: Record<ButtonVariant, string> = {
  primary:
    "bg-[var(--accent)] text-white border border-[var(--accent-strong)] shadow-[0_10px_28px_-16px_rgba(15,118,110,0.65)] hover:bg-[var(--accent-strong)]",
  secondary:
    "bg-[var(--paper)] text-[var(--ink)] border border-[var(--line-strong)] hover:bg-[var(--paper-deep)]",
  quiet:
    "bg-transparent text-[var(--ink-muted)] border border-transparent hover:text-[var(--ink)] hover:border-[var(--line)]",
};

export function MagneticButton({
  children,
  onClick,
  disabled = false,
  variant = "primary",
  className = "",
  type = "button",
  ariaLabel,
}: MagneticButtonProps) {
  const reduceMotion = useReducedMotion();
  const pointerX = useMotionValue(0);
  const pointerY = useMotionValue(0);
  const magneticX = useTransform(pointerX, (value) => value * 0.14);
  const magneticY = useTransform(pointerY, (value) => value * 0.14);
  const x = useSpring(magneticX, { stiffness: 180, damping: 18, mass: 0.4 });
  const y = useSpring(magneticY, { stiffness: 180, damping: 18, mass: 0.4 });

  const handlePointerMove = (event: PointerEvent<HTMLButtonElement>) => {
    if (reduceMotion || event.pointerType === "touch") return;
    const bounds = event.currentTarget.getBoundingClientRect();
    pointerX.set(event.clientX - bounds.left - bounds.width / 2);
    pointerY.set(event.clientY - bounds.top - bounds.height / 2);
  };

  const resetPosition = () => {
    pointerX.set(0);
    pointerY.set(0);
  };

  return (
    <motion.button
      type={type}
      onClick={onClick}
      disabled={disabled}
      aria-label={ariaLabel}
      onPointerMove={handlePointerMove}
      onPointerLeave={resetPosition}
      onPointerCancel={resetPosition}
      whileTap={reduceMotion ? undefined : { scale: 0.98 }}
      style={reduceMotion ? undefined : { x, y }}
      className={`inline-flex min-h-11 items-center justify-center gap-2 rounded-full px-5 text-sm font-semibold tracking-[-0.01em] transition-colors duration-300 disabled:cursor-not-allowed disabled:opacity-45 ${variants[variant]} ${className}`}
    >
      {children}
    </motion.button>
  );
}
