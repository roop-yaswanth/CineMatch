"use client";

/**
 * Lightweight global toast/snackbar.
 *
 * Use `toast(message)` or `toast({ message, action: { label, onClick } })`
 * from anywhere in the client tree. The toast renders in a portal so it
 * sits above modals and the bottom nav, and auto-dismisses after 3.2s
 * (or 5s when an action is present so users have time to undo).
 *
 * No context provider is required — the host component subscribes to a
 * tiny event bus on mount; pages just call `toast(...)` and forget.
 */

import { useEffect, useState } from "react";
import { createPortal } from "react-dom";
import { AnimatePresence, motion } from "framer-motion";

export interface ToastOptions {
  message: string;
  /** Optional inline action — shown to the right of the message. */
  action?: { label: string; onClick: () => void };
  /** Override the auto-dismiss delay in ms. */
  durationMs?: number;
  /** Visual tone affects the leading icon color. */
  tone?: "neutral" | "success" | "danger";
}

interface ToastEntry extends ToastOptions {
  id: number;
  durationMs: number;
}

type Listener = (entries: ToastEntry[]) => void;

let entries: ToastEntry[] = [];
let nextId = 1;
const listeners = new Set<Listener>();

function emit() {
  for (const l of listeners) l(entries);
}

/** Show a toast. Accepts a string shorthand for the common "just a message" case. */
export function toast(input: string | ToastOptions): void {
  const opts: ToastOptions = typeof input === "string" ? { message: input } : input;
  const id = nextId++;
  const durationMs = opts.durationMs ?? (opts.action ? 5000 : 3200);
  const entry: ToastEntry = { id, durationMs, ...opts };
  entries = [...entries, entry];
  emit();
  // Auto-dismiss.
  setTimeout(() => {
    entries = entries.filter((e) => e.id !== id);
    emit();
  }, durationMs);
}

function dismiss(id: number) {
  entries = entries.filter((e) => e.id !== id);
  emit();
}

const TONE_COLOR: Record<NonNullable<ToastOptions["tone"]>, string> = {
  neutral: "rgba(255,255,255,0.85)",
  success: "var(--color-like)",
  danger: "var(--color-dislike)",
};

/**
 * Mount once at the app root. Renders a portal listening to the global
 * toast queue.
 */
export default function ToastHost() {
  const [list, setList] = useState<ToastEntry[]>([]);
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    Promise.resolve().then(() => setMounted(true));
    const l: Listener = (next) => setList(next);
    listeners.add(l);
    Promise.resolve().then(() => setList(entries));
    return () => { listeners.delete(l); };
  }, []);

  if (!mounted) return null;

  return createPortal(
    <div
      style={{
        position: "fixed",
        // Sit above the floating bottom nav on mobile.
        bottom: "calc(96px + env(safe-area-inset-bottom))",
        left: 16,
        right: 16,
        zIndex: 1200,
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        gap: 8,
        pointerEvents: "none",
      }}
    >
      <AnimatePresence>
        {list.map((t) => (
          <motion.div
            key={t.id}
            initial={{ y: 16, opacity: 0, scale: 0.96 }}
            animate={{ y: 0, opacity: 1, scale: 1 }}
            exit={{ y: 8, opacity: 0, scale: 0.97 }}
            transition={{ type: "spring", stiffness: 460, damping: 32 }}
            style={{
              pointerEvents: "auto",
              minWidth: "min(320px, 100%)",
              maxWidth: 480,
              padding: "12px 14px",
              borderRadius: 14,
              display: "flex",
              alignItems: "center",
              gap: 12,
              background: "rgba(20, 22, 28, 0.85)",
              border: "1px solid rgba(255,255,255,0.10)",
              backdropFilter: "blur(40px) saturate(1.6)",
              WebkitBackdropFilter: "blur(40px) saturate(1.6)",
              boxShadow: "0 12px 36px rgba(0,0,0,0.55), 0 1px 0 rgba(255,255,255,0.10) inset",
              color: "var(--color-text-primary)",
              fontSize: 13,
              fontWeight: 500,
            }}
          >
            <span
              aria-hidden
              style={{
                width: 8,
                height: 8,
                borderRadius: 999,
                background: TONE_COLOR[t.tone ?? "neutral"],
                flexShrink: 0,
              }}
            />
            <span style={{ flex: 1, minWidth: 0, lineHeight: 1.4 }}>{t.message}</span>
            {t.action && (
              <button
                type="button"
                onClick={() => {
                  t.action!.onClick();
                  dismiss(t.id);
                }}
                style={{
                  background: "rgba(255,255,255,0.10)",
                  border: "1px solid rgba(255,255,255,0.14)",
                  color: "var(--color-text-primary)",
                  borderRadius: 999,
                  padding: "5px 11px",
                  fontSize: 12,
                  fontWeight: 600,
                  cursor: "pointer",
                  flexShrink: 0,
                }}
              >
                {t.action.label}
              </button>
            )}
            <button
              type="button"
              onClick={() => dismiss(t.id)}
              aria-label="Dismiss"
              style={{
                background: "none",
                border: "none",
                color: "var(--color-text-muted)",
                padding: 4,
                cursor: "pointer",
                flexShrink: 0,
                display: "flex",
              }}
            >
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.4" strokeLinecap="round" strokeLinejoin="round">
                <line x1="18" y1="6" x2="6" y2="18" />
                <line x1="6" y1="6" x2="18" y2="18" />
              </svg>
            </button>
          </motion.div>
        ))}
      </AnimatePresence>
    </div>,
    document.body
  );
}
