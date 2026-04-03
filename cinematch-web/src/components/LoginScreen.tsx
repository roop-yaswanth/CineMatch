"use client";

import { useState, useRef } from "react";
import { motion } from "framer-motion";
import { apiLogin, type UserSession } from "@/lib/api";

interface Props {
  onLogin: (session: UserSession) => void;
}

export default function LoginScreen({ onLogin }: Props) {
  const [email, setEmail] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const inputRef = useRef<HTMLInputElement>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    const trimmed = email.trim();
    if (!trimmed) {
      setError("Please enter your email.");
      inputRef.current?.focus();
      return;
    }
    setError("");
    setLoading(true);
    try {
      const session = await apiLogin(trimmed);
      onLogin(session);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Connection failed. Is the backend running?");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex flex-col items-center justify-center min-h-dvh px-6">
      {/* Brand */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.7, delay: 0.1, ease: [0.25, 0.1, 0.25, 1] }}
        className="text-center mb-16"
      >
        <h1 className="text-[2.5rem] md:text-[3.5rem] font-light tracking-[-0.04em] text-[var(--color-text-primary)]">
          CineMatch
        </h1>
        <p className="mt-3 text-sm text-[var(--color-text-muted)] font-light tracking-wide">
          Discover movies you&apos;ll love
        </p>
      </motion.div>

      {/* Email form */}
      <motion.form
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.7, delay: 0.25, ease: [0.25, 0.1, 0.25, 1] }}
        onSubmit={handleSubmit}
        className="w-full max-w-sm"
      >
        <div className="relative">
          <input
            ref={inputRef}
            id="email-input"
            type="email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            placeholder="Enter your email"
            autoComplete="email"
            autoFocus
            disabled={loading}
            className="
              w-full px-0 py-4
              bg-transparent
              border-0 border-b border-[var(--color-border)]
              text-lg font-light text-[var(--color-text-primary)]
              placeholder:text-[var(--color-text-muted)]
              focus:outline-none focus:border-[var(--color-text-secondary)]
              transition-colors duration-300
              disabled:opacity-40
            "
          />
          {/* Animated underline */}
          <motion.div
            className="absolute bottom-0 left-0 h-px bg-[var(--color-text-primary)]"
            initial={{ width: "0%" }}
            whileFocus={{ width: "100%" }}
            transition={{ duration: 0.4 }}
          />
        </div>

        {error && (
          <motion.p
            initial={{ opacity: 0, y: -4 }}
            animate={{ opacity: 1, y: 0 }}
            className="mt-4 text-xs text-[var(--color-danger)] font-light"
          >
            {error}
          </motion.p>
        )}

        <motion.button
          type="submit"
          disabled={loading}
          whileHover={{ scale: 1.01 }}
          whileTap={{ scale: 0.99 }}
          className="
            mt-10 w-full py-3.5
            bg-[var(--color-text-primary)] text-[var(--color-bg)]
            text-sm font-medium tracking-wide
            rounded-full
            hover:bg-[var(--color-accent)]
            disabled:opacity-40 disabled:cursor-not-allowed
            transition-colors duration-200
          "
        >
          {loading ? (
            <span className="inline-flex items-center gap-2">
              <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24" fill="none">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="3" />
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
              </svg>
              Connecting
            </span>
          ) : (
            "Continue"
          )}
        </motion.button>
      </motion.form>

      {/* Footer hint */}
      <motion.p
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.6, duration: 0.6 }}
        className="mt-16 text-xs text-[var(--color-text-muted)] font-light"
      >
        No password needed. Your email is only used to save progress.
      </motion.p>
    </div>
  );
}
