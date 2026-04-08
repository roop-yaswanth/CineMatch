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
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        justifyContent: "center",
        minHeight: "100dvh",
        width: "100%",
        padding: "0 24px",
        fontFamily: "var(--font-sans)",
      }}
    >
      {/* Brand */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.7, delay: 0.1, ease: [0.25, 0.1, 0.25, 1] }}
        style={{
          textAlign: "center",
          marginBottom: "10vh",
        }}
      >
        <h1
          style={{
            fontSize: "clamp(4rem, 12vw, 7.2rem)",
            lineHeight: 0.95,
            fontWeight: 300,
            letterSpacing: "-0.05em",
            color: "var(--color-text-primary)",
            margin: 0,
          }}
        >
          CineMatch
        </h1>
        <p
          style={{
            marginTop: "12px",
            fontSize: "clamp(1.05rem, 2.5vw, 1.5rem)",
            color: "var(--color-text-muted)",
            fontWeight: 300,
            letterSpacing: "0.04em",
          }}
        >
          Discover movies you&apos;ll love
        </p>
      </motion.div>

      {/* Email form */}
      <motion.form
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.7, delay: 0.25, ease: [0.25, 0.1, 0.25, 1] }}
        onSubmit={handleSubmit}
        style={{
          width: "100%",
          maxWidth: "420px",
        }}
      >
        <div style={{ position: "relative" }}>
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
            style={{
              width: "100%",
              padding: "16px 0",
              background: "transparent",
              border: "none",
              borderBottom: "1px solid var(--color-border)",
              fontSize: "clamp(1.1rem, 2.5vw, 1.4rem)",
              fontWeight: 300,
              color: "var(--color-text-primary)",
              outline: "none",
              fontFamily: "inherit",
              opacity: loading ? 0.4 : 1,
              transition: "border-color 0.3s ease",
            }}
            onFocus={(e) => {
              e.currentTarget.style.borderBottomColor = "var(--color-text-secondary)";
            }}
            onBlur={(e) => {
              e.currentTarget.style.borderBottomColor = "var(--color-border)";
            }}
          />
        </div>

        {error && (
          <motion.p
            initial={{ opacity: 0, y: -4 }}
            animate={{ opacity: 1, y: 0 }}
            style={{
              marginTop: "16px",
              fontSize: "12px",
              color: "var(--color-danger)",
              fontWeight: 300,
            }}
          >
            {error}
          </motion.p>
        )}

        <motion.button
          type="submit"
          disabled={loading}
          whileHover={{ scale: 1.01 }}
          whileTap={{ scale: 0.99 }}
          style={{
            marginTop: "40px",
            width: "100%",
            padding: "16px 0",
            backgroundColor: "var(--color-text-primary)",
            color: "var(--color-bg)",
            fontSize: "clamp(1rem, 2vw, 1.25rem)",
            fontWeight: 500,
            letterSpacing: "0.02em",
            borderRadius: "9999px",
            border: "none",
            cursor: loading ? "not-allowed" : "pointer",
            opacity: loading ? 0.4 : 1,
            transition: "background-color 0.2s ease",
            fontFamily: "inherit",
          }}
          onMouseEnter={(e) => {
            if (!loading) e.currentTarget.style.backgroundColor = "var(--color-accent)";
          }}
          onMouseLeave={(e) => {
            e.currentTarget.style.backgroundColor = "var(--color-text-primary)";
          }}
        >
          {loading ? (
            <span style={{ display: "inline-flex", alignItems: "center", gap: "8px" }}>
              <svg
                style={{ animation: "spin 1s linear infinite", height: "16px", width: "16px" }}
                viewBox="0 0 24 24"
                fill="none"
              >
                <circle
                  style={{ opacity: 0.25 }}
                  cx="12"
                  cy="12"
                  r="10"
                  stroke="currentColor"
                  strokeWidth="3"
                />
                <path
                  style={{ opacity: 0.75 }}
                  fill="currentColor"
                  d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"
                />
              </svg>
              Connecting
            </span>
          ) : (
            "Continue"
          )}
        </motion.button>
      </motion.form>

    </div>
  );
}
