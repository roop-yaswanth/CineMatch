"use client";

import * as React from "react";
import { type ReactNode, type HTMLAttributes, useEffect, useRef, useCallback, useState } from "react";
import { createPortal } from "react-dom";
import { motion, AnimatePresence } from "framer-motion";
import { glassStyles, typographyStyles } from "../utils/styles";
import { Button } from "./Button";
import { Flex } from "./Layout";
import { IconClose } from "@/components/shared/icons";

// ISP: small, focused interfaces — clients depend only on what they use
export interface ModalCoreProps {
  isOpen: boolean;
  onClose: () => void;
  children: ReactNode;
}
export interface ModalHeaderProps {
  title?: ReactNode;
  description?: string;
  showCloseButton?: boolean;
  hideHeader?: boolean;
}
export interface ModalBehaviorProps {
  closeOnOverlayClick?: boolean;
  closeOnEscape?: boolean;
}
export interface ModalStyleProps {
  size?: "sm" | "md" | "lg" | "xl" | "full";
  variant?: "default" | "confirmation" | "bottom-sheet";
  footer?: ReactNode;
}
export interface ModalProps extends Omit<HTMLAttributes<HTMLDivElement>, "children" | "title">, ModalCoreProps, ModalHeaderProps, ModalBehaviorProps, ModalStyleProps {}

const sizeStyles: Record<Exclude<ModalProps["size"], undefined>, React.CSSProperties> = {
  sm: { maxWidth: "360px" },
  md: { maxWidth: "480px" },
  lg: { maxWidth: "640px" },
  xl: { maxWidth: "800px" },
  full: { maxWidth: "calc(100vw - 32px)", width: "calc(100vw - 32px)" },
};

const variantStyles: Record<Exclude<ModalProps["variant"], undefined>, React.CSSProperties> = {
  default: { borderRadius: "var(--radius-modal)" },
  confirmation: { borderRadius: "var(--radius-modal)", maxWidth: "400px" },
  "bottom-sheet": { borderRadius: "var(--radius-2xl) var(--radius-2xl) 0 0", maxWidth: "100%", width: "100%", margin: 0, maxHeight: "85vh" },
};

export function Modal({
  isOpen,
  onClose,
  title,
  description,
  children,
  size = "md",
  variant = "default",
  closeOnOverlayClick = true,
  closeOnEscape = true,
  showCloseButton = true,
  footer,
  hideHeader = false,
  className,
}: ModalProps) {
  const contentRef = useRef<HTMLDivElement>(null);
  const previousActiveElement = useRef<HTMLElement | null>(null);

  const handleKeyDown = useCallback(
    (e: KeyboardEvent) => {
      if (!closeOnEscape) return;
      if (e.key === "Escape") {
        onClose();
      }
      if (e.key === "Tab") {
        const focusableElements = contentRef.current?.querySelectorAll<HTMLElement>(
          'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
        );
        if (focusableElements?.length) {
          const firstElement = focusableElements[0];
          const lastElement = focusableElements[focusableElements.length - 1];
          if (e.shiftKey && document.activeElement === firstElement) {
            e.preventDefault();
            lastElement.focus();
          } else if (!e.shiftKey && document.activeElement === lastElement) {
            e.preventDefault();
            firstElement.focus();
          }
        }
      }
    },
    [closeOnEscape, onClose]
  );

  useEffect(() => {
    if (isOpen) {
      previousActiveElement.current = document.activeElement as HTMLElement;
      document.addEventListener("keydown", handleKeyDown);
      document.body.style.overflow = "hidden";
      setTimeout(() => contentRef.current?.focus(), 0);
    }
    return () => {
      document.removeEventListener("keydown", handleKeyDown);
      document.body.style.overflow = "";
      previousActiveElement.current?.focus();
    };
  }, [isOpen, handleKeyDown]);

  if (!isOpen) return null;

  const modalContent = (
    <AnimatePresence>
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        transition={{ duration: 0.2, ease: "easeOut" }}
        style={{
          position: "fixed",
          inset: 0,
          zIndex: 100,
          background: "rgba(6, 6, 10, 0.55)",
          backdropFilter: "blur(16px) saturate(1.2)",
          WebkitBackdropFilter: "blur(16px) saturate(1.2)",
          display: "flex",
          alignItems: variant === "bottom-sheet" ? "flex-end" : "center",
          justifyContent: "center",
          padding: "24px",
          overflowY: "auto",
        }}
        onClick={closeOnOverlayClick ? onClose : undefined}
        role="presentation"
      >
        <motion.div
          ref={contentRef}
          initial={{ opacity: 0, scale: 0.96, y: variant === "bottom-sheet" ? 40 : 20 }}
          animate={{ opacity: 1, scale: 1, y: 0 }}
          exit={{ opacity: 0, scale: 0.96, y: variant === "bottom-sheet" ? 40 : 20 }}
          transition={{ duration: 0.25, ease: [0.22, 1, 0.36, 1] }}
          style={{
            ...glassStyles.modal,
            ...sizeStyles[size],
            ...variantStyles[variant],
            width: "100%",
            maxHeight: variant === "bottom-sheet" ? "85vh" : "90vh",
            overflow: "hidden",
            display: "flex",
            flexDirection: "column",
          }}
          tabIndex={-1}
          onClick={(e) => e.stopPropagation()}
          role="dialog"
          aria-modal="true"
          aria-labelledby={title ? "modal-title" : undefined}
          aria-describedby={description ? "modal-description" : undefined}
          className={className}
        >
          {!hideHeader && (title || showCloseButton) && (
            <div
              style={{
                display: "flex",
                alignItems: description ? "flex-start" : "center",
                justifyContent: "space-between",
                padding: "18px 24px",
                borderBottom: "1px solid var(--hairline)",
                flexShrink: 0,
              }}
            >
              <div style={{ flex: 1, paddingRight: "var(--s-4)" }}>
                {title && (
                  <h2
                    id="modal-title"
                    style={{
                      fontSize: "20px",
                      fontWeight: 700,
                      letterSpacing: "-0.025em",
                      color: "var(--color-text-primary)",
                      margin: 0,
                      lineHeight: 1.25,
                    }}
                  >
                    {title}
                  </h2>
                )}
                {description && <p id="modal-description" style={{ ...typographyStyles.meta, margin: "var(--s-1) 0 0" }}>{description}</p>}
              </div>
              {showCloseButton && (
                <button
                  type="button"
                  onClick={onClose}
                  aria-label="Close"
                  style={{
                    width: 32,
                    height: 32,
                    borderRadius: "50%",
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    color: "var(--color-text-secondary)",
                    cursor: "pointer",
                    padding: 0,
                    flexShrink: 0,
                    background: "rgba(255, 255, 255, 0.08)",
                    border: "1px solid rgba(255, 255, 255, 0.14)",
                    backdropFilter: "blur(20px) saturate(1.4)",
                    WebkitBackdropFilter: "blur(20px) saturate(1.4)",
                    transition: "all var(--dur-fast) var(--ease-out)",
                  }}
                  onMouseEnter={(e) => {
                    e.currentTarget.style.background = "rgba(255, 255, 255, 0.16)";
                    e.currentTarget.style.color = "#ffffff";
                    e.currentTarget.style.borderColor = "rgba(255, 255, 255, 0.28)";
                  }}
                  onMouseLeave={(e) => {
                    e.currentTarget.style.background = "rgba(255, 255, 255, 0.08)";
                    e.currentTarget.style.color = "var(--color-text-secondary)";
                    e.currentTarget.style.borderColor = "rgba(255, 255, 255, 0.14)";
                  }}
                >
                  <IconClose size={14} />
                </button>
              )}
            </div>
          )}

          <div
            style={{
              flex: 1,
              overflowY: "auto",
              padding: hideHeader ? "var(--s-6)" : "var(--s-5) var(--s-6)",
              WebkitOverflowScrolling: "touch",
            }}
          >
            {children}
          </div>

          {footer && (
            <div
              style={{
                display: "flex",
                alignItems: "center",
                justifyContent: "flex-end",
                gap: "var(--s-3)",
                padding: "var(--s-4) var(--s-6)",
                borderTop: "1px solid var(--hairline)",
                flexShrink: 0,
              }}
            >
              {footer}
            </div>
          )}
        </motion.div>
      </motion.div>
    </AnimatePresence>
  );

  if (typeof window === "undefined") return null;
  return createPortal(modalContent, document.body);
}

export interface ConfirmDialogProps extends Omit<ModalProps, "variant" | "footer" | "children"> {
  message: string;
  confirmLabel?: string;
  cancelLabel?: string;
  confirmVariant?: "primary" | "danger";
  onConfirm: () => void;
  loading?: boolean;
}

export function ConfirmDialog({
  isOpen,
  onClose,
  title = "Confirm",
  message,
  confirmLabel = "Confirm",
  cancelLabel = "Cancel",
  confirmVariant = "primary",
  onConfirm,
  loading = false,
  ...props
}: ConfirmDialogProps) {
  return (
    <Modal
      isOpen={isOpen}
      onClose={onClose}
      title={title}
      variant="confirmation"
      size="sm"
      footer={
        <Flex gap="3" justify="end">
          <Button variant="ghost" onClick={onClose} disabled={loading}>
            {cancelLabel}
          </Button>
          <Button variant={confirmVariant} onClick={onConfirm} loading={loading}>
            {confirmLabel}
          </Button>
        </Flex>
      }
      {...props}
    >
      <p style={{ ...typographyStyles.body, margin: 0 }}>{message}</p>
    </Modal>
  );
}

export interface BottomSheetProps extends Omit<ModalProps, "variant" | "size"> {
  handle?: boolean;
  snapPoints?: number[];
  defaultSnap?: number;
}

export function BottomSheet({
  isOpen,
  onClose,
  title,
  description,
  children,
  handle = true,
  className,
  style,
  ...props
}: BottomSheetProps) {
  return (
    <Modal
      isOpen={isOpen}
      onClose={onClose}
      title={title}
      description={description}
      variant="bottom-sheet"
      size="full"
      showCloseButton={false}
      className={className}
      style={style}
      {...props}
    >
      {handle && (
        <div
          style={{
            width: "36px",
            height: "5px",
            borderRadius: "999px",
            background: "var(--hairline)",
            margin: "0 auto var(--s-4)",
            cursor: "grab",
          }}
        />
      )}
      {children}
    </Modal>
  );
}

export interface ToastProps extends HTMLAttributes<HTMLDivElement> {
  type?: "info" | "success" | "warning" | "error" | "default";
  title: string;
  message?: string;
  action?: ReactNode;
  duration?: number;
  onClose?: () => void;
}

export function Toast({ type = "default", title, message, action, duration = 4000, className, style }: ToastProps) {
  const [visible, setVisible] = useState(true);

  useEffect(() => {
    if (duration > 0) {
      const timer = setTimeout(() => setVisible(false), duration);
      return () => clearTimeout(timer);
    }
  }, [duration]);

  if (!visible) return null;

  const typeStyles: Record<string, React.CSSProperties> = {
    info: { borderLeft: "3px solid var(--color-blue)" },
    success: { borderLeft: "3px solid var(--color-success)" },
    warning: { borderLeft: "3px solid var(--color-yellow)" },
    error: { borderLeft: "3px solid var(--color-danger)" },
    default: { borderLeft: "3px solid var(--color-accent)" },
  };

  const icons: Record<string, ReactNode> = {
    info: <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="var(--color-blue)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/></svg>,
    success: <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="var(--color-success)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>,
    warning: <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="var(--color-yellow)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>,
    error: <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="var(--color-danger)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><circle cx="12" cy="12" r="10"/><line x1="15" y1="9" x2="9" y2="15"/><line x1="9" y1="9" x2="15" y2="15"/></svg>,
    default: <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="var(--color-accent)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/></svg>,
  };

  return (
    <motion.div
      initial={{ opacity: 0, x: 300, y: 20 }}
      animate={{ opacity: 1, x: 0, y: 0 }}
      exit={{ opacity: 0, x: 300, y: 20 }}
      transition={{ duration: 0.3, ease: [0.22, 1, 0.36, 1] }}
      className={className}
      style={{
        display: "flex",
        alignItems: "flex-start",
        gap: "var(--s-3)",
        padding: "var(--s-4) var(--s-5)",
        background: "rgba(20, 22, 28, 0.96)",
        backdropFilter: "blur(40px) saturate(1.6)",
        WebkitBackdropFilter: "blur(40px) saturate(1.6)",
        border: "1px solid var(--hairline)",
        borderRadius: "var(--radius-lg)",
        boxShadow: "var(--shadow-lg)",
        minWidth: "300px",
        maxWidth: "420px",
        ...typeStyles[type],
        ...style,
      }}
      role="alert"
      aria-live="polite"
    >
      <div style={{ flexShrink: 0, marginTop: 2 }}>{icons[type]}</div>
      <div style={{ flex: 1, minWidth: 0 }}>
        <p style={{ ...typographyStyles.h3, margin: 0 }}>{title}</p>
        {message && <p style={{ ...typographyStyles.meta, margin: "var(--s-1) 0 0" }}>{message}</p>}
      </div>
      <Flex>
        {action}
        <Button variant="ghost" size="sm" onClick={() => setVisible(false)} style={{ padding: "4px 8px", flexShrink: 0 }}>
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <line x1="18" y1="6" x2="6" y2="18" />
            <line x1="6" y1="6" x2="18" y2="18" />
          </svg>
        </Button>
      </Flex>
    </motion.div>
  );
}

export function ToastContainer({ children, ...props }: { children: ReactNode } & HTMLAttributes<HTMLDivElement>) {
  return (
    <div
      style={{
        position: "fixed",
        bottom: "calc(24px + env(safe-area-inset-bottom))",
        right: "24px",
        left: "24px",
        maxWidth: "480px",
        margin: "0 auto",
        zIndex: 200,
        display: "flex",
        flexDirection: "column",
        gap: "var(--s-3)",
        pointerEvents: "none",
        ...props,
      }}
    >
      <AnimatePresence>{React.Children.map(children, (child) => React.isValidElement(child) ? React.cloneElement(child as React.ReactElement<{ style?: React.CSSProperties }>, { style: { ...((child as React.ReactElement<{ style?: React.CSSProperties }>).props.style as React.CSSProperties), pointerEvents: "auto" } }) : child)}</AnimatePresence>
    </div>
  );
}