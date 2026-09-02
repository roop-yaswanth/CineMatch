"use client";

import { type ReactNode, type HTMLAttributes, useId } from "react";
import { mergeStyles, glassStyles, typographyStyles } from "../utils/styles";
import { BackButton } from "./BackButton";

export interface PageHeaderProps extends Omit<HTMLAttributes<HTMLElement>, "title"> {
  title: ReactNode;
  subtitle?: ReactNode;
  backHref?: string;
  backLabel?: string;
  hideBackButton?: boolean;
  onBack?: () => void;
  actions?: ReactNode;
  variant?: "default" | "transparent" | "minimal";
  sticky?: boolean;
  showSearchButton?: boolean;
  onSearch?: () => void;
}

export function PageHeader({
  title,
  subtitle,
  backHref,
  backLabel = "Back",
  hideBackButton = false,
  onBack,
  actions,
  variant = "default",
  sticky = true,
  showSearchButton = false,
  onSearch,
  className,
  style,
  ...props
}: PageHeaderProps) {
  const headerVariants: Record<string, React.CSSProperties> = {
    default: glassStyles.header,
    transparent: {
      background: "transparent",
      backdropFilter: "none",
      WebkitBackdropFilter: "none",
      borderBottom: "none",
      boxShadow: "none",
    },
    minimal: {
      background: "rgba(5, 5, 7, 0.8)",
      backdropFilter: "blur(20px) saturate(1.5)",
      WebkitBackdropFilter: "blur(20px) saturate(1.5)",
      borderBottom: "1px solid var(--hairline)",
      boxShadow: "none",
    },
  };

  const combinedStyle = mergeStyles(
    {
      position: sticky ? "sticky" : "relative",
      top: 0,
      zIndex: 40,
      paddingTop: "env(safe-area-inset-top, 0px)",
      width: "100%",
    },
    headerVariants[variant],
    style
  );

  return (
    <header className={className} style={combinedStyle} {...props}>
      <div
        style={{
          width: "100%",
          padding: "10px 16px",
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          position: "relative",
          minHeight: "48px",
        }}
      >
        <div style={{ display: "flex", alignItems: "center", minWidth: "44px", zIndex: 10 }}>
          {!hideBackButton ? (
            <BackButton href={backHref} onClick={onBack} ariaLabel={backLabel} />
          ) : (
            <div style={{ width: "44px" }} />
          )}
        </div>

        <div style={{ flex: 1, display: "flex", flexDirection: "column", alignItems: "center", gap: 2, textAlign: "center", padding: "0 12px" }}>
          <h1 style={{ ...typographyStyles.h1, margin: 0 }}>{title}</h1>
          {subtitle && <p style={{ ...typographyStyles.meta, margin: 0 }}>{subtitle}</p>}
        </div>

        <div style={{ display: "flex", alignItems: "center", justifyContent: "flex-end", gap: "8px", minWidth: "44px", zIndex: 10 }}>
          {showSearchButton && (
            <button
              type="button"
              onClick={onSearch}
              style={{
                display: "inline-flex",
                alignItems: "center",
                gap: "8px",
                padding: "8px 14px",
                borderRadius: "var(--radius-pill)",
                background: "var(--glass-chrome)",
                backdropFilter: "blur(var(--blur-thin)) saturate(1.4)",
                WebkitBackdropFilter: "blur(var(--blur-thin)) saturate(1.4)",
                border: "1px solid var(--hairline)",
                color: "var(--color-text-secondary)",
                fontSize: "var(--fs-sm)",
                fontWeight: 500,
                cursor: "pointer",
                transition: "all var(--dur-base) var(--ease-out)",
              }}
              onMouseEnter={(e) => {
                e.currentTarget.style.background = "var(--glass-chrome-strong)";
                e.currentTarget.style.borderColor = "var(--hairline-strong)";
                e.currentTarget.style.color = "var(--color-text-primary)";
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.background = "var(--glass-chrome)";
                e.currentTarget.style.borderColor = "var(--hairline)";
                e.currentTarget.style.color = "var(--color-text-secondary)";
              }}
            >
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
                <circle cx="11" cy="11" r="8" />
                <line x1="21" y1="21" x2="16.65" y2="16.65" />
              </svg>
              Search
            </button>
          )}
          {actions}
        </div>
      </div>
    </header>
  );
}

export interface PageLayoutProps extends HTMLAttributes<HTMLDivElement> {
  header?: PageHeaderProps;
  children: ReactNode;
  footer?: ReactNode;
  bottomNav?: ReactNode;
  className?: string;
  style?: React.CSSProperties;
  background?: string;
}

export function PageLayout({
  header,
  children,
  footer,
  bottomNav,
  className,
  style,
  background = "var(--color-bg)",
  ...props
}: PageLayoutProps) {
  return (
    <div
      className={className}
      style={{
        minHeight: "100dvh",
        display: "flex",
        flexDirection: "column",
        background,
        fontFamily: "var(--font-sans)",
        ...style,
      }}
      {...props}
    >
      {header && <PageHeader {...header} />}
      <main style={{ flex: 1, width: "100%", position: "relative", zIndex: 1 }}>{children}</main>
      {footer && <footer style={{ padding: "var(--s-6) var(--s-header-x)", borderTop: "1px solid var(--hairline)" }}>{footer}</footer>}
      {bottomNav && (
        <div style={{ position: "fixed", bottom: 0, left: 0, right: 0, zIndex: 50, pointerEvents: "none" }}>
          <div style={{ pointerEvents: "auto" }}>{bottomNav}</div>
        </div>
      )}
    </div>
  );
}

export interface AppShellProps extends HTMLAttributes<HTMLDivElement> {
  header?: PageHeaderProps;
  children: ReactNode;
  sidebar?: ReactNode;
  sidebarOpen?: boolean;
  onSidebarToggle?: () => void;
}

export function AppShell({ header, children, sidebar, sidebarOpen, onSidebarToggle, className, style, ...props }: AppShellProps) {
  const sidebarId = useId();

  return (
    <div
      className={className}
      style={{
        minHeight: "100dvh",
        display: "flex",
        background: "var(--color-bg)",
        fontFamily: "var(--font-sans)",
        ...style,
      }}
      {...props}
    >
      {sidebar && (
        <aside
          id={sidebarId}
          style={{
            position: "fixed",
            top: 0,
            left: 0,
            bottom: 0,
            width: "280px",
            maxWidth: "85vw",
            zIndex: 60,
            background: "rgba(10, 10, 15, 0.98)",
            backdropFilter: "blur(40px) saturate(1.5)",
            WebkitBackdropFilter: "blur(40px) saturate(1.5)",
            borderRight: "1px solid var(--hairline)",
            boxShadow: "var(--shadow-xl)",
            transform: sidebarOpen ? "translateX(0)" : "translateX(-100%)",
            transition: "transform var(--dur-base) var(--ease-out)",
            overflowY: "auto",
            pointerEvents: sidebarOpen ? "auto" : "none",
          }}
          aria-label="Sidebar"
        >
          {sidebar}
        </aside>
      )}

      <div style={{ flex: 1, width: "100%", minWidth: 0, marginLeft: sidebar ? (sidebarOpen ? "280px" : 0) : 0, transition: "margin-left var(--dur-base) var(--ease-out)" }}>
        {header && <PageHeader {...header} />}
        <main style={{ flex: 1, width: "100%", position: "relative", zIndex: 1 }}>{children}</main>
      </div>

      {!sidebarOpen && sidebar && (
        <div
          style={{
            position: "fixed",
            inset: 0,
            zIndex: 50,
            background: "rgba(0,0,0,0.5)",
            backdropFilter: "blur(4px)",
            WebkitBackdropFilter: "blur(4px)",
            opacity: sidebarOpen ? 1 : 0,
            pointerEvents: sidebarOpen ? "auto" : "none",
            transition: "opacity var(--dur-base) var(--ease-out)",
          }}
          onClick={onSidebarToggle}
          aria-hidden="true"
        />
      )}
    </div>
  );
}

export interface EmptyStateProps extends HTMLAttributes<HTMLDivElement> {
  title: string;
  description?: string;
  icon?: ReactNode;
  action?: ReactNode;
  illustration?: ReactNode;
}

export function EmptyState({ title, description, icon, action, illustration, className, style, ...props }: EmptyStateProps) {
  return (
    <div
      className={className}
      style={{
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        justifyContent: "center",
        textAlign: "center",
        padding: "var(--s-10) var(--s-5)",
        gap: "var(--s-4)",
        ...style,
      }}
      {...props}
    >
      {illustration || icon || (
        <div style={{ fontSize: "64px", opacity: 0.3 }}>🎬</div>
      )}
      <h2 style={{ ...typographyStyles.h2, margin: 0 }}>{title}</h2>
      {description && <p style={{ ...typographyStyles.body, margin: 0, maxWidth: "320px" }}>{description}</p>}
      {action && <div style={{ marginTop: "var(--s-3)" }}>{action}</div>}
    </div>
  );
}

export interface LoadingStateProps extends HTMLAttributes<HTMLDivElement> {
  message?: string;
  size?: "sm" | "md" | "lg";
  inline?: boolean;
}

export function LoadingState({ message, size = "md", inline = false, className, style, ...props }: LoadingStateProps) {
  const spinnerSize = size === "sm" ? 20 : size === "md" ? 32 : 48;
  const messageStyle: React.CSSProperties = {
    marginTop: "var(--s-3)",
    fontSize: size === "sm" ? "var(--fs-sm)" : "var(--fs-md)",
    color: "var(--color-text-muted)",
    textAlign: "center",
  };

  return (
    <div
      className={className}
      style={{
        display: inline ? "inline-flex" : "flex",
        flexDirection: inline ? "row" : "column",
        alignItems: "center",
        justifyContent: "center",
        gap: inline ? "var(--s-3)" : "var(--s-3)",
        padding: inline ? "var(--s-2) var(--s-4)" : "var(--s-8)",
        ...style,
      }}
      {...props}
      aria-busy="true"
      aria-label={message || "Loading"}
    >
      <svg
        width={spinnerSize}
        height={spinnerSize}
        viewBox="0 0 24 24"
        fill="none"
        stroke="var(--color-accent)"
        strokeWidth={2.5}
        strokeLinecap="round"
        strokeLinejoin="round"
        style={{ animation: "spin 0.7s linear infinite" }}
        aria-hidden="true"
      >
        <circle cx="12" cy="12" r="10" strokeOpacity="0.25" />
        <path d="M12 2a10 10 0 0 1 10 10" strokeOpacity="1" />
      </svg>
      {message && !inline && <p style={messageStyle}>{message}</p>}
      {message && inline && <span style={messageStyle}>{message}</span>}
    </div>
  );
}

export interface ErrorStateProps extends HTMLAttributes<HTMLDivElement> {
  title: string;
  message?: string;
  action?: ReactNode;
  dismissible?: boolean;
  onDismiss?: () => void;
}

export function ErrorState({ title, message, action, dismissible = false, onDismiss, className, style, ...props }: ErrorStateProps) {
  return (
    <div
      className={className}
      style={{
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        justifyContent: "center",
        textAlign: "center",
        padding: "var(--s-8) var(--s-5)",
        gap: "var(--s-4)",
        background: "rgba(255, 69, 58, 0.08)",
        border: "1px solid rgba(255, 69, 58, 0.2)",
        borderRadius: "var(--radius-lg)",
        ...style,
      }}
      role="alert"
      {...props}
    >
      <div style={{ fontSize: "48px" }}>⚠️</div>
      <h2 style={{ ...typographyStyles.h2, color: "var(--color-danger)", margin: 0 }}>{title}</h2>
      {message && <p style={{ ...typographyStyles.body, margin: 0, maxWidth: "320px" }}>{message}</p>}
      {dismissible && onDismiss && (
        <button
          type="button"
          onClick={onDismiss}
          style={{
            position: "absolute",
            top: "var(--s-3)",
            right: "var(--s-3)",
            background: "none",
            border: "none",
            color: "var(--color-text-muted)",
            cursor: "pointer",
            padding: "var(--s-1)",
            borderRadius: "var(--radius-sm)",
          }}
          aria-label="Dismiss"
        >
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <line x1="18" y1="6" x2="6" y2="18" />
            <line x1="6" y1="6" x2="18" y2="18" />
          </svg>
        </button>
      )}
      {action && <div style={{ marginTop: "var(--s-2)" }}>{action}</div>}
    </div>
  );
}