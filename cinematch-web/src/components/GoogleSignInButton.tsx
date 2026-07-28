"use client";

import { useEffect, useRef, useState, useCallback } from "react";
import { apiGoogleLogin, type UserSession } from "@/lib/api";

const GOOGLE_CLIENT_ID = process.env.NEXT_PUBLIC_GOOGLE_CLIENT_ID ?? "";

// Minimal shape of the Google Identity Services global we use.
type GoogleIdentity = {
  accounts?: {
    id?: {
      initialize: (config: {
        client_id: string;
        callback: (resp: { credential?: string }) => void;
      }) => void;
      renderButton: (parent: HTMLElement, options: Record<string, unknown>) => void;
      prompt: () => void;
    };
  };
};

interface Props {
  /** Called with the CineMatch session after Google verifies the user. */
  onLogin: (session: UserSession) => void;
  /** Also show Google's One Tap prompt. Default true. */
  oneTap?: boolean;
}

/**
 * Renders the official Google Identity Services sign-in button and exchanges
 * the returned ID token for a CineMatch session via the backend. Self-contained
 * (loads the GIS script once, handles loading/error state); the parent decides
 * what to do with the session through `onLogin`.
 */
export default function GoogleSignInButton({ onLogin, oneTap = true }: Props) {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const btnRef = useRef<HTMLDivElement>(null);

  const handleCredential = useCallback(async (credential: string) => {
    setError("");
    setLoading(true);
    try {
      const session = await apiGoogleLogin(credential);
      onLogin(session);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Sign-in failed. Please try again.");
      setLoading(false);
    }
    // On success we leave `loading` true — the parent navigates away, so
    // flipping it back would only flash the button before unmount.
  }, [onLogin]);

  useEffect(() => {
    if (!GOOGLE_CLIENT_ID) {
      // eslint-disable-next-line react-hooks/set-state-in-effect -- one-time config guard
      setError("Google sign-in isn't configured (missing NEXT_PUBLIC_GOOGLE_CLIENT_ID).");
      return;
    }
    const SCRIPT_SRC = "https://accounts.google.com/gsi/client";

    const init = () => {
      const g = (window as unknown as { google?: GoogleIdentity }).google;
      if (!g?.accounts?.id || !btnRef.current) return;
      g.accounts.id.initialize({
        client_id: GOOGLE_CLIENT_ID,
        callback: (resp: { credential?: string }) => {
          if (resp?.credential) void handleCredential(resp.credential);
        },
      });
      g.accounts.id.renderButton(btnRef.current, {
        theme: "filled_black",
        size: "large",
        shape: "pill",
        text: "continue_with",
        width: 320,
        logo_alignment: "center",
      });
      if (oneTap) g.accounts.id.prompt();
    };

    const existing = document.querySelector<HTMLScriptElement>(`script[src="${SCRIPT_SRC}"]`);
    if (existing) {
      if ((window as unknown as { google?: GoogleIdentity }).google) init();
      else existing.addEventListener("load", init, { once: true });
      return;
    }
    const script = document.createElement("script");
    script.src = SCRIPT_SRC;
    script.async = true;
    script.defer = true;
    script.onload = init;
    script.onerror = () => setError("Couldn't load Google sign-in. Check your connection.");
    document.head.appendChild(script);
  }, [handleCredential, oneTap]);

  return (
    <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: "14px" }}>
      {/* Google renders its official button into this node */}
      <div
        ref={btnRef}
        style={{
          minHeight: "44px",
          opacity: loading ? 0.5 : 1,
          pointerEvents: loading ? "none" : "auto",
          colorScheme: "light",
        }}
      />

      {loading && (
        <span style={{ display: "inline-flex", alignItems: "center", gap: "8px", color: "var(--color-text-secondary)", fontSize: "14px" }}>
          <svg
            style={{ animation: "spin 1s linear infinite", height: "16px", width: "16px" }}
            viewBox="0 0 24 24"
            fill="none"
          >
            <circle style={{ opacity: 0.25 }} cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="3" />
            <path style={{ opacity: 0.75 }} fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
          </svg>
          Signing you in…
        </span>
      )}

      {error && (
        <p style={{ fontSize: "13px", color: "var(--color-danger)", fontWeight: 400, textAlign: "center", maxWidth: "320px" }}>
          {error}
        </p>
      )}
    </div>
  );
}
