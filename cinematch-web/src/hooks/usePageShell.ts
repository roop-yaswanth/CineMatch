"use client";

import { useEffect, useRef, useState, useCallback } from "react";
export function useScrollProgress() {
  const [progress, setProgress] = useState(0);
  const ticking = useRef(false);
  const lastY = useRef(0);

  useEffect(() => {
    const onScroll = () => {
      if (ticking.current) return;
      ticking.current = true;
      requestAnimationFrame(() => {
        const y = window.scrollY;
        const max = document.documentElement.scrollHeight - window.innerHeight;
        setProgress(max > 40 ? Math.min(y / max, 1) : 0);
        lastY.current = y;
        ticking.current = false;
      });
    };
    window.addEventListener("scroll", onScroll, { passive: true });
    return () => window.removeEventListener("scroll", onScroll);
  }, []);

  return progress;
}

export function useHideOnScroll(threshold = 60) {
  const [hidden, setHidden] = useState(false);
  const lastY = useRef(0);
  const ticking = useRef(false);

  useEffect(() => {
    const onScroll = () => {
      if (ticking.current) return;
      ticking.current = true;
      requestAnimationFrame(() => {
        const y = window.scrollY;
        const dy = y - lastY.current;
        if (Math.abs(dy) > 6) {
          if (y < threshold) setHidden(false);
          else if (dy > 0) setHidden(true);
          else setHidden(false);
          lastY.current = y;
        }
        ticking.current = false;
      });
    };
    window.addEventListener("scroll", onScroll, { passive: true });
    return () => window.removeEventListener("scroll", onScroll);
  }, [threshold]);

  return hidden;
}

export function useDebouncedValue<T>(value: T, delay = 200) {
  const [debounced, setDebounced] = useState(value);
  useEffect(() => {
    const t = setTimeout(() => setDebounced(value), delay);
    return () => clearTimeout(t);
  }, [value, delay]);
  return debounced;
}

export function useLocalStorageFlag(key: string) {
  const get = useCallback(() => {
    try { return sessionStorage.getItem(key) === "1"; } catch { return false; }
  }, [key]);
  const set = useCallback((v: boolean) => {
    try { if (v) sessionStorage.setItem(key, "1"); else sessionStorage.removeItem(key); } catch { }
  }, [key]);
  const clear = useCallback(() => { try { sessionStorage.removeItem(key); } catch { } }, [key]);
  return { get, set, clear };
}
