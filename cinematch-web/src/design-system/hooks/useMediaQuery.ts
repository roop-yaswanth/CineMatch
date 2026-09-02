"use client";

import { useState, useEffect, useCallback, useRef } from "react";

export function useMediaQuery(query: string): boolean {
  const [matches, setMatches] = useState(() =>
    typeof window !== "undefined" ? window.matchMedia(query).matches : false
  );

  useEffect(() => {
    if (typeof window === "undefined") return;
    const media = window.matchMedia(query);
    const listener = (e: MediaQueryListEvent) => setMatches(e.matches);
    // Sync on query change — intentional setState in effect to reflect media query.
    // eslint-disable-next-line react-hooks/set-state-in-effect
    setMatches(media.matches);
    media.addEventListener("change", listener);
    return () => media.removeEventListener("change", listener);
  }, [query]);

  return matches;
}

export function useBreakpoint() {
  const isMobile = useMediaQuery("(max-width: 639px)");
  const isTablet = useMediaQuery("(min-width: 640px) and (max-width: 1023px)");
  const isDesktop = useMediaQuery("(min-width: 1024px)");
  const isWide = useMediaQuery("(min-width: 1440px)");

  return { isMobile, isTablet, isDesktop, isWide };
}

export function useReducedMotion(): boolean {
  return useMediaQuery("(prefers-reduced-motion: reduce)");
}

export function useHover(): boolean {
  return useMediaQuery("(hover: hover)");
}

export function useTouch(): boolean {
  return useMediaQuery("(pointer: coarse)");
}

export function useLocalStorage<T>(key: string, initialValue: T | (() => T)): [T, (value: T | ((val: T) => T)) => void] {
  const [storedValue, setStoredValue] = useState<T>(() => {
    if (typeof window === "undefined") return initialValue instanceof Function ? initialValue() : initialValue;
    try {
      const item = window.localStorage.getItem(key);
      return item ? JSON.parse(item) : (initialValue instanceof Function ? initialValue() : initialValue);
    } catch {
      return initialValue instanceof Function ? initialValue() : initialValue;
    }
  });

  const setValue = useCallback((value: T | ((val: T) => T)) => {
    try {
      const valueToStore = value instanceof Function ? value(storedValue) : value;
      setStoredValue(valueToStore);
      if (typeof window !== "undefined") {
        window.localStorage.setItem(key, JSON.stringify(valueToStore));
      }
    } catch (error) {
      console.error(`Error setting localStorage key "${key}":`, error);
    }
  }, [key, storedValue]);

  return [storedValue, setValue];
}

export function useSessionStorage<T>(key: string, initialValue: T | (() => T)): [T, (value: T | ((val: T) => T)) => void] {
  const [storedValue, setStoredValue] = useState<T>(() => {
    if (typeof window === "undefined") return initialValue instanceof Function ? initialValue() : initialValue;
    try {
      const item = window.sessionStorage.getItem(key);
      return item ? JSON.parse(item) : (initialValue instanceof Function ? initialValue() : initialValue);
    } catch {
      return initialValue instanceof Function ? initialValue() : initialValue;
    }
  });

  const setValue = useCallback((value: T | ((val: T) => T)) => {
    try {
      const valueToStore = value instanceof Function ? value(storedValue) : value;
      setStoredValue(valueToStore);
      if (typeof window !== "undefined") {
        window.sessionStorage.setItem(key, JSON.stringify(valueToStore));
      }
    } catch (error) {
      console.error(`Error setting sessionStorage key "${key}":`, error);
    }
  }, [key, storedValue]);

  return [storedValue, setValue];
}

export function useDebounce<T>(value: T, delay: number): T {
  const [debouncedValue, setDebouncedValue] = useState(value);

  useEffect(() => {
    const handler = setTimeout(() => setDebouncedValue(value), delay);
    return () => clearTimeout(handler);
  }, [value, delay]);

  return debouncedValue;
}

export function useThrottle<T>(value: T, limit: number): T {
  const [throttledValue, setThrottledValue] = useState(value);
  const lastRan = useRef<number | null>(null);

  useEffect(() => {
    if (lastRan.current === null) {
      lastRan.current = Date.now();
    }
    const handler = setTimeout(() => {
      if (lastRan.current !== null && Date.now() - lastRan.current >= limit) {
        setThrottledValue(value);
        lastRan.current = Date.now();
      }
    }, limit - (lastRan.current !== null ? Date.now() - lastRan.current : limit));

    return () => clearTimeout(handler);
  }, [value, limit]);

  return throttledValue;
}

export function useClickOutside<T extends HTMLElement = HTMLDivElement>(
  handler: (event: MouseEvent | TouchEvent) => void,
  options?: { capture?: boolean; ignoreRefs?: React.RefObject<HTMLElement>[] }
): React.RefObject<T | null> {
  const ref = useRef<T>(null);
  const { capture = false, ignoreRefs = [] } = options || {};

  useEffect(() => {
    const listener = (event: MouseEvent | TouchEvent) => {
      const target = event.target as Node;
      if (!ref.current || ref.current.contains(target)) return;
      if (ignoreRefs.some((r) => r.current && r.current.contains(target))) return;
      handler(event);
    };

    document.addEventListener("mousedown", listener, capture);
    document.addEventListener("touchstart", listener, capture);
    return () => {
      document.removeEventListener("mousedown", listener, capture);
      document.removeEventListener("touchstart", listener, capture);
    };
  }, [handler, capture, ignoreRefs]);

  return ref;
}

export function useOnScreen(ref: React.RefObject<HTMLElement>, rootMargin = "0px"): boolean {
  const [isVisible, setIsVisible] = useState(false);

  useEffect(() => {
    if (!ref.current || typeof IntersectionObserver === "undefined") return;

    const observer = new IntersectionObserver(
      ([entry]) => setIsVisible(entry.isIntersecting),
      { rootMargin }
    );

    observer.observe(ref.current);
    return () => observer.disconnect();
  }, [ref, rootMargin]);

  return isVisible;
}

export function useIntersectionObserver(
  ref: React.RefObject<HTMLElement>,
  options?: IntersectionObserverInit
): IntersectionObserverEntry | undefined {
  const [entry, setEntry] = useState<IntersectionObserverEntry>();

  useEffect(() => {
    if (!ref.current || typeof IntersectionObserver === "undefined") return;

    const observer = new IntersectionObserver(([entry]) => setEntry(entry), options);
    observer.observe(ref.current);
    return () => observer.disconnect();
  }, [ref, options]);

  return entry;
}

export function useWindowSize(): { width: number; height: number } {
  const [size, setSize] = useState({ width: 0, height: 0 });

  useEffect(() => {
    if (typeof window === "undefined") return;

    const handleResize = () => setSize({ width: window.innerWidth, height: window.innerHeight });
    handleResize();
    window.addEventListener("resize", handleResize);
    return () => window.removeEventListener("resize", handleResize);
  }, []);

  return size;
}

export function useScrollPosition(): { x: number; y: number } {
  const [position, setPosition] = useState({ x: 0, y: 0 });

  useEffect(() => {
    if (typeof window === "undefined") return;

    const handleScroll = () => setPosition({ x: window.scrollX, y: window.scrollY });
    handleScroll();
    window.addEventListener("scroll", handleScroll, { passive: true });
    return () => window.removeEventListener("scroll", handleScroll);
  }, []);

  return position;
}

export function useTimeout(callback: () => void, delay: number | null): { reset: () => void; clear: () => void } {
  const callbackRef = useRef(callback);
  const timeoutRef = useRef<NodeJS.Timeout | null>(null);

  useEffect(() => {
    callbackRef.current = callback;
  }, [callback]);

  const set = useCallback(() => {
    if (delay === null) return;
    timeoutRef.current = setTimeout(() => callbackRef.current(), delay);
  }, [delay]);

  const clear = useCallback(() => {
    if (timeoutRef.current) clearTimeout(timeoutRef.current);
    timeoutRef.current = null;
  }, []);

  useEffect(() => {
    set();
    return clear;
  }, [delay, set, clear]);

  const reset = useCallback(() => {
    clear();
    set();
  }, [clear, set]);

  return { reset, clear };
}

export function useInterval(callback: () => void, delay: number | null): { clear: () => void } {
  const callbackRef = useRef(callback);
  const intervalRef = useRef<NodeJS.Timeout | null>(null);

  useEffect(() => {
    callbackRef.current = callback;
  }, [callback]);

  useEffect(() => {
    if (delay === null) return;
    intervalRef.current = setInterval(() => callbackRef.current(), delay);
    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
    };
  }, [delay]);

  const clear = useCallback(() => {
    if (intervalRef.current) clearInterval(intervalRef.current);
    intervalRef.current = null;
  }, []);

  return { clear };
}

export function usePrevious<T>(value: T): T | undefined {
  const ref = useRef<T | null>(null);
  useEffect(() => {
    ref.current = value;
  }, [value]);
  // Accessing ref during render is intentional for usePrevious — tracks previous render value.
  // eslint-disable-next-line react-hooks/refs
  return ref.current ?? undefined;
}

export function useUpdateEffect(effect: React.EffectCallback, deps?: React.DependencyList) {
  const isFirstRender = useRef(true);

  useEffect(() => {
    if (isFirstRender.current) {
      isFirstRender.current = false;
      return;
    }
    return effect();
    // effect is intentionally not in deps — caller controls when to run.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);
}

export function useMounted(): boolean {
  const [mounted, setMounted] = useState(false);
  // Mounted flag must be set in effect — intentional to avoid hydration mismatch.
  // eslint-disable-next-line react-hooks/set-state-in-effect
  useEffect(() => setMounted(true), []);
  return mounted;
}

export function useCopyToClipboard(): [boolean, (text: string) => Promise<void>] {
  const [copied, setCopied] = useState(false);

  const copy = useCallback(async (text: string) => {
    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (error) {
      console.error("Failed to copy:", error);
      setCopied(false);
    }
  }, []);

  return [copied, copy];
}

export function useAsync<T, E = Error>(
  asyncFn: () => Promise<T>,
  immediate = true
): { data: T | null; error: E | null; loading: boolean; execute: () => Promise<T | null> } {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<E | null>(null);
  const [loading, setLoading] = useState(false);

  const execute = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const result = await asyncFn();
      setData(result);
      return result;
    } catch (err) {
      setError(err as E);
      return null;
    } finally {
      setLoading(false);
    }
  }, [asyncFn]);

  useEffect(() => {
    if (immediate) {
      // Immediate execute on mount — intentional eager fetch.
      // eslint-disable-next-line react-hooks/set-state-in-effect
      void execute();
    }
  }, [execute, immediate]);

  return { data, error, loading, execute };
}