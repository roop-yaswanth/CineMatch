"use client";

import { useEffect, useRef, useState, useCallback } from "react";
import { apiGoogleLogin, type UserSession } from "@/lib/api";

const GOOGLE_CLIENT_ID = process.env.NEXT_PUBLIC_GOOGLE_CLIENT_ID ?? "";

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
  /** Button theme. Defaults to "filled_blue" for high contrast on dark UI. */
  theme?: "filled_blue" | "outline" | "filled_black";
  /** Button shape. Defaults to "pill". */
  shape?: "pill" | "rectangular";
  /** Button width in px. Default 320. */
  width?: number;
}

export default function GoogleSignInButton({
  onLogin,
  oneTap = true,
  theme = "filled_blue",
  shape = "pill",
  width = 320,
}: Props) {
  const [loading, setLoading] = useState(false);
  const [ready, setReady] = useState(false);
  const [error, setError] = useState("");
  const btnRef = useRef<HTMLDivElement>(null);

  const handleCredential = useCallback(
    async (credential: string) => {
      setError("");
      setLoading(true);
      try {
        const session = await apiGoogleLogin(credential);
        onLogin(session);
      } catch (err) {
        setError(err instanceof Error ? err.message : "Sign-in failed. Please try again.");
        setLoading(false);
      }
    },
    [onLogin]
  );

  const handleCredentialRef = useRef(handleCredential);
  useEffect(() => {
    handleCredentialRef.current = handleCredential;
  }, [handleCredential]);

  const initializedRef = useRef(false);

  useEffect(() => {
    if (!GOOGLE_CLIENT_ID) {
      // eslint-disable-next-line react-hooks/set-state-in-effect
      setError("Google sign-in isn't configured (missing NEXT_PUBLIC_GOOGLE_CLIENT_ID).");
      return;
    }
    const SCRIPT_SRC = "https://accounts.google.com/gsi/client";

    const init = () => {
      const g = (window as unknown as { google?: GoogleIdentity }).google;
      if (!g?.accounts?.id || !btnRef.current) return;
      if (!initializedRef.current) {
        g.accounts.id.initialize({
          client_id: GOOGLE_CLIENT_ID,
          callback: (resp: { credential?: string }) => {
            if (resp?.credential) void handleCredentialRef.current(resp.credential);
          },
        });
        initializedRef.current = true;
      }
      g.accounts.id.renderButton(btnRef.current, {
        theme,
        size: "large",
        shape,
        text: "continue_with",
        width,
        logo_alignment: "center",
      });
      setReady(true);
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
  }, [oneTap, theme, shape, width]);

  return (
    <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: "12px", width: "100%" }}>
      {/* Container holding the custom high-end button & transparent GIS iframe overlay */}
      <div
        style={{
          position: "relative",
          width: `${width}px`,
          maxWidth: "100%",
          height: "48px",
          display: "flex",
          justifyContent: "center",
          alignItems: "center",
          isolation: "isolate",
        }}
      >
        {/* Custom High-End Visual Button */}
        <button
          type="button"
          tabIndex={-1}
          style={{
            position: "absolute",
            inset: 0,
            width: "100%",
            height: "100%",
            display: "inline-flex",
            alignItems: "center",
            justifyContent: "center",
            gap: "12px",
            borderRadius: "9999px",
            background: "#ffffff",
            color: "#0d0e15",
            fontWeight: 600,
            fontSize: "14.5px",
            letterSpacing: "-0.015em",
            fontFamily: "var(--font-sans), -apple-system, BlinkMacSystemFont, sans-serif",
            boxShadow: "0 4px 20px rgba(255, 255, 255, 0.18), 0 2px 6px rgba(0, 0, 0, 0.4)",
            border: "1px solid rgba(255, 255, 255, 0.8)",
            cursor: "pointer",
            transition: "all 0.2s cubic-bezier(0.16, 1, 0.3, 1)",
            pointerEvents: "none",
          }}
        >
          {/* Precise 4-color Google SVG Logo */}
          <svg width="20" height="20" viewBox="0 0 24 24" style={{ flexShrink: 0 }}>
            <path
              fill="#4285F4"
              d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z"
            />
            <path
              fill="#34A853"
              d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"
            />
            <path
              fill="#FBBC05"
              d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.06H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.94l2.85-2.22.81-.63z"
            />
            <path
              fill="#EA4335"
              d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.06l3.66 2.84c.87-2.6 3.3-4.52 6.16-4.52z"
            />
          </svg>
          <span>Continue with Google</span>
        </button>

        {/* Dedicated target node for Google Identity Services iframe overlaid transparently */}
        <div
          ref={btnRef}
          style={{
            position: "absolute",
            inset: 0,
            width: "100%",
            height: "100%",
            display: "flex",
            justifyContent: "center",
            alignItems: "center",
            opacity: ready ? 0.0001 : 0,
            overflow: "hidden",
            zIndex: 2,
            cursor: loading ? "wait" : "pointer",
            pointerEvents: loading ? "none" : "auto",
          }}
        />
      </div>

      {loading && (
        <span
          style={{
            display: "inline-flex",
            alignItems: "center",
            gap: "8px",
            color: "var(--color-text-secondary)",
            fontSize: "13px",
            fontWeight: 500,
          }}
        >
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
        <p
          style={{
            fontSize: "13px",
            color: "#ff6b6b",
            fontWeight: 500,
            textAlign: "center",
            maxWidth: "320px",
            margin: 0,
            background: "rgba(255, 107, 107, 0.1)",
            padding: "8px 14px",
            borderRadius: "8px",
            border: "1px solid rgba(255, 107, 107, 0.2)",
          }}
        >
          {error}
        </p>
      )}
    </div>
  );
}
