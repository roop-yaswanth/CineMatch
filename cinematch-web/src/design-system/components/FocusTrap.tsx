"use client";

import { type ReactNode, useEffect, useRef, useCallback } from "react";

export interface FocusTrapProps {
  children: ReactNode;
  enabled?: boolean;
  onDeactivate?: () => void;
  autoFocus?: boolean;
}

export function FocusTrap({ children, enabled = true, onDeactivate, autoFocus = true }: FocusTrapProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const previousActiveElement = useRef<HTMLElement | null>(null);

  const getFocusableElements = useCallback(() => {
    if (!containerRef.current) return [];
    return Array.from(
      containerRef.current.querySelectorAll<HTMLElement>(
        'button:not([disabled]), [href], input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"]):not([disabled]), summary'
      )
    ).filter((el) => el.offsetWidth > 0 || el.offsetHeight > 0 || el.getClientRects().length > 0);
  }, []);

  const handleKeyDown = useCallback(
    (e: KeyboardEvent) => {
      if (!enabled || e.key !== "Tab") return;

      const focusableElements = getFocusableElements();
      if (focusableElements.length === 0) return;

      const firstElement = focusableElements[0];
      const lastElement = focusableElements[focusableElements.length - 1];

      if (e.shiftKey) {
        if (document.activeElement === firstElement) {
          e.preventDefault();
          lastElement.focus();
        }
      } else {
        if (document.activeElement === lastElement) {
          e.preventDefault();
          firstElement.focus();
        }
      }
    },
    [enabled, getFocusableElements]
  );

  useEffect(() => {
    if (!enabled) return;

    previousActiveElement.current = document.activeElement as HTMLElement;

    const focusableElements = getFocusableElements();
    if (autoFocus && focusableElements.length > 0) {
      setTimeout(() => focusableElements[0].focus(), 0);
    }

    document.addEventListener("keydown", handleKeyDown);

    return () => {
      document.removeEventListener("keydown", handleKeyDown);
      if (onDeactivate) onDeactivate();
      previousActiveElement.current?.focus();
    };
  }, [enabled, autoFocus, getFocusableElements, handleKeyDown, onDeactivate]);

  return <div ref={containerRef} tabIndex={-1}>{children}</div>;
}

export interface FocusLockProps {
  children: ReactNode;
  enabled?: boolean;
  returnFocus?: boolean;
}

export function FocusLock({ children, enabled = true, returnFocus = true }: FocusLockProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const previousActiveElement = useRef<HTMLElement | null>(null);

  useEffect(() => {
    if (!enabled) return;

    previousActiveElement.current = document.activeElement as HTMLElement;

    const focusableElements = containerRef.current?.querySelectorAll<HTMLElement>(
      'button:not([disabled]), [href], input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"]):not([disabled])'
    );

    if (focusableElements?.length) {
      focusableElements[0].focus();
    }

    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key !== "Tab") return;

      const elements = Array.from(focusableElements || []);
      if (elements.length === 0) return;

      const firstElement = elements[0];
      const lastElement = elements[elements.length - 1];

      if (e.shiftKey && document.activeElement === firstElement) {
        e.preventDefault();
        lastElement.focus();
      } else if (!e.shiftKey && document.activeElement === lastElement) {
        e.preventDefault();
        firstElement.focus();
      }
    };

    document.addEventListener("keydown", handleKeyDown);

    return () => {
      document.removeEventListener("keydown", handleKeyDown);
      if (returnFocus) {
        previousActiveElement.current?.focus();
      }
    };
  }, [enabled, returnFocus]);

  return <div ref={containerRef}>{children}</div>;
}