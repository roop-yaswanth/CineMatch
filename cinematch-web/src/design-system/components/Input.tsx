"use client";

import { forwardRef, type InputHTMLAttributes, type ReactNode, useId } from "react";
import { mergeStyles } from "../utils/styles";

export type InputSize = "sm" | "md" | "lg";

export interface InputProps extends Omit<InputHTMLAttributes<HTMLInputElement>, "size"> {
  label?: string;
  error?: string;
  hint?: string;
  leftIcon?: ReactNode;
  rightIcon?: ReactNode;
  leftElement?: ReactNode;
  rightElement?: ReactNode;
  size?: InputSize;
  fullWidth?: boolean;
}

const sizeStyles: Record<InputSize, React.CSSProperties> = {
  sm: { padding: "9px 36px 9px 40px", fontSize: "var(--fs-sm)", borderRadius: "10px" },
  md: { padding: "13px 40px 13px 44px", fontSize: "16px", borderRadius: "14px" },
  lg: { padding: "15px 44px 15px 48px", fontSize: "var(--fs-lg)", borderRadius: "16px" },
};

const baseStyle: React.CSSProperties = {
  width: "100%",
  border: "1px solid rgba(255, 255, 255, 0.10)",
  background: "rgba(28, 30, 36, 0.82)",
  color: "var(--color-text-primary)",
  fontWeight: 400,
  letterSpacing: "-0.005em",
  outline: "none",
  transition: "border-color var(--dur-base) var(--ease-out), background var(--dur-base) var(--ease-out), box-shadow var(--dur-base) var(--ease-out)",
  WebkitAppearance: "none",
  appearance: "none",
};

const focusStyle: React.CSSProperties = {
  borderColor: "rgba(255, 255, 255, 0.30)",
  background: "rgba(36, 38, 46, 0.92)",
  boxShadow: "0 0 0 3px rgba(255, 255, 255, 0.08), 0 1px 0 0 rgba(255, 255, 255, 0.06) inset",
};

const errorStyle: React.CSSProperties = {
  borderColor: "var(--color-danger)",
  boxShadow: "0 0 0 3px rgba(255, 69, 58, 0.15)",
};

export const Input = forwardRef<HTMLInputElement, InputProps>(
  ({
    label,
    error,
    hint,
    leftIcon,
    rightIcon,
    leftElement,
    rightElement,
    size = "md",
    fullWidth = true,
    className,
    style,
    id: providedId,
    disabled,
    required,
    ...props
  }, ref) => {
    const generatedId = useId();
    const id = providedId || generatedId;
    const errorId = `${id}-error`;
    const hintId = `${id}-hint`;
    const hasError = Boolean(error);

    const combinedStyle = mergeStyles(
      baseStyle,
      sizeStyles[size],
      fullWidth && { width: "100%" },
      hasError && errorStyle,
      disabled && { opacity: 0.5, cursor: "not-allowed" },
      style
    );

    const wrapperStyle: React.CSSProperties = {
      position: "relative",
      display: "inline-flex",
      alignItems: "center",
      width: fullWidth ? "100%" : "auto",
    };

    const labelStyle: React.CSSProperties = {
      display: "block",
      marginBottom: "var(--s-1)",
      fontSize: "var(--fs-sm)",
      fontWeight: 500,
      color: "var(--color-text-secondary)",
    };

    const messageStyle: React.CSSProperties = {
      marginTop: "var(--s-1)",
      fontSize: "var(--fs-xs)",
      color: hasError ? "var(--color-danger)" : "var(--color-text-muted)",
    };

    return (
      <div style={wrapperStyle} className={className}>
        {label && <label htmlFor={id} style={labelStyle}>{label}{required && <span style={{ color: "var(--color-danger)", marginLeft: 4 }}> *</span>}</label>}
        <div style={{ position: "relative", width: "100%" }}>
          {(leftIcon || leftElement) && (
            <div
              style={{
                position: "absolute",
                left: size === "sm" ? 10 : 14,
                top: "50%",
                transform: "translateY(-50%)",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                color: "var(--color-text-muted)",
                pointerEvents: "none",
                zIndex: 1,
              }}
            >
              {leftElement || leftIcon}
            </div>
          )}
          <input
            ref={ref}
            id={id}
            {...props}
            style={combinedStyle}
            disabled={disabled}
            required={required}
            aria-invalid={hasError}
            aria-describedby={`${hasError ? errorId : ""} ${hint ? hintId : ""}`.trim() || undefined}
            onFocus={(e) => {
              e.currentTarget.style.cssText += `;${Object.entries(focusStyle).map(([k, v]) => `${k}:${v}`).join(";")}`;
              props.onFocus?.(e);
            }}
            onBlur={(e) => {
              const errorStyles = hasError ? Object.entries(errorStyle).map(([k, v]) => `${k}:${v}`).join(";") : "";
              e.currentTarget.style.cssText = Object.entries(combinedStyle).map(([k, v]) => `${k}:${v}`).join(";") + (errorStyles ? `;${errorStyles}` : "");
              props.onBlur?.(e);
            }}
          />
          {(rightIcon || rightElement) && (
            <div
              style={{
                position: "absolute",
                right: size === "sm" ? 10 : 14,
                top: "50%",
                transform: "translateY(-50%)",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                color: "var(--color-text-muted)",
                pointerEvents: "none",
                zIndex: 1,
              }}
            >
              {rightElement || rightIcon}
            </div>
          )}
        </div>
        {hasError && <p id={errorId} style={messageStyle} role="alert">{error}</p>}
        {hint && !hasError && <p id={hintId} style={messageStyle}>{hint}</p>}
      </div>
    );
  }
);

Input.displayName = "Input";

export interface TextareaProps extends Omit<React.TextareaHTMLAttributes<HTMLTextAreaElement>, "size"> {
  label?: string;
  error?: string;
  hint?: string;
  size?: InputSize;
  fullWidth?: boolean;
  minRows?: number;
}

export const Textarea = forwardRef<HTMLTextAreaElement, TextareaProps>(
  ({ label, error, hint, size = "md", fullWidth = true, minRows = 3, className, style, id: providedId, disabled, required, ...props }, ref) => {
    const generatedId = useId();
    const id = providedId || generatedId;
    const errorId = `${id}-error`;
    const hintId = `${id}-hint`;
    const hasError = Boolean(error);

    const baseTextareaStyle: React.CSSProperties = {
      ...baseStyle,
      minHeight: `${minRows * (size === "sm" ? 28 : size === "md" ? 32 : 40)}px`,
      resize: "vertical",
      paddingTop: size === "sm" ? 9 : 13,
      paddingBottom: size === "sm" ? 9 : 13,
      lineHeight: "var(--lh-base)",
    };

    const combinedStyle = mergeStyles(
      baseTextareaStyle,
      fullWidth && { width: "100%" },
      hasError && errorStyle,
      disabled && { opacity: 0.5, cursor: "not-allowed" },
      style
    );

    return (
      <div style={{ display: "inline-flex", flexDirection: "column", width: fullWidth ? "100%" : "auto" }} className={className}>
        {label && <label htmlFor={id} style={{ marginBottom: "var(--s-1)", fontSize: "var(--fs-sm)", fontWeight: 500, color: "var(--color-text-secondary)" }}>{label}{required && <span style={{ color: "var(--color-danger)", marginLeft: 4 }}> *</span>}</label>}
        <textarea
          ref={ref}
          id={id}
          {...props}
          style={combinedStyle}
          disabled={disabled}
          required={required}
          aria-invalid={hasError}
          aria-describedby={`${hasError ? errorId : ""} ${hint ? hintId : ""}`.trim() || undefined}
          onFocus={(e) => {
            e.currentTarget.style.cssText += `;${Object.entries(focusStyle).map(([k, v]) => `${k}:${v}`).join(";")}`;
            props.onFocus?.(e);
          }}
          onBlur={(e) => {
            const errorStyles = hasError ? Object.entries(errorStyle).map(([k, v]) => `${k}:${v}`).join(";") : "";
            e.currentTarget.style.cssText = Object.entries(combinedStyle).map(([k, v]) => `${k}:${v}`).join(";") + (errorStyles ? `;${errorStyles}` : "");
            props.onBlur?.(e);
          }}
        />
        {hasError && <p id={errorId} style={{ marginTop: "var(--s-1)", fontSize: "var(--fs-xs)", color: "var(--color-danger)" }} role="alert">{error}</p>}
        {hint && !hasError && <p id={hintId} style={{ marginTop: "var(--s-1)", fontSize: "var(--fs-xs)", color: "var(--color-text-muted)" }}>{hint}</p>}
      </div>
    );
  }
);

Textarea.displayName = "Textarea";

export interface SelectProps extends Omit<React.SelectHTMLAttributes<HTMLSelectElement>, "size"> {
  label?: string;
  error?: string;
  hint?: string;
  options: Array<{ value: string; label: string; disabled?: boolean }>;
  placeholder?: string;
  size?: InputSize;
  fullWidth?: boolean;
}

export const Select = forwardRef<HTMLSelectElement, SelectProps>(
  ({ label, error, hint, options, placeholder, size = "md", fullWidth = true, className, style, id: providedId, disabled, required, ...props }, ref) => {
    const generatedId = useId();
    const id = providedId || generatedId;
    const errorId = `${id}-error`;
    const hintId = `${id}-hint`;
    const hasError = Boolean(error);

    const selectBaseStyle: React.CSSProperties = {
      ...baseStyle,
      paddingRight: size === "sm" ? 36 : 44,
      backgroundImage: "url(\"data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='16' height='16' viewBox='0 0 24 24' fill='none' stroke='%237c7c85' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpolyline points='6 9 12 15 18 9'%3E%3C/polyline%3E%3C/svg%3E\")",
      backgroundRepeat: "no-repeat",
      backgroundPosition: `right ${size === "sm" ? 10 : 14}px center`,
      backgroundSize: "16px 16px",
      appearance: "none",
    };

    const combinedStyle = mergeStyles(
      selectBaseStyle,
      sizeStyles[size],
      fullWidth && { width: "100%" },
      hasError && errorStyle,
      disabled && { opacity: 0.5, cursor: "not-allowed" },
      style
    );

    return (
      <div style={{ display: "inline-flex", flexDirection: "column", width: fullWidth ? "100%" : "auto" }} className={className}>
        {label && <label htmlFor={id} style={{ marginBottom: "var(--s-1)", fontSize: "var(--fs-sm)", fontWeight: 500, color: "var(--color-text-secondary)" }}>{label}{required && <span style={{ color: "var(--color-danger)", marginLeft: 4 }}> *</span>}</label>}
        <select
          ref={ref}
          id={id}
          {...props}
          style={combinedStyle}
          disabled={disabled}
          required={required}
          aria-invalid={hasError}
          aria-describedby={`${hasError ? errorId : ""} ${hint ? hintId : ""}`.trim() || undefined}
          onFocus={(e) => {
            e.currentTarget.style.cssText += `;${Object.entries(focusStyle).map(([k, v]) => `${k}:${v}`).join(";")}`;
            props.onFocus?.(e);
          }}
          onBlur={(e) => {
            const errorStyles = hasError ? Object.entries(errorStyle).map(([k, v]) => `${k}:${v}`).join(";") : "";
            e.currentTarget.style.cssText = Object.entries(combinedStyle).map(([k, v]) => `${k}:${v}`).join(";") + (errorStyles ? `;${errorStyles}` : "");
            props.onBlur?.(e);
          }}
        >
          {placeholder && <option value="" disabled>{placeholder}</option>}
          {options.map((opt) => (
            <option key={opt.value} value={opt.value} disabled={opt.disabled}>
              {opt.label}
            </option>
          ))}
        </select>
        {hasError && <p id={errorId} style={{ marginTop: "var(--s-1)", fontSize: "var(--fs-xs)", color: "var(--color-danger)" }} role="alert">{error}</p>}
        {hint && !hasError && <p id={hintId} style={{ marginTop: "var(--s-1)", fontSize: "var(--fs-xs)", color: "var(--color-text-muted)" }}>{hint}</p>}
      </div>
    );
  }
);

Select.displayName = "Select";