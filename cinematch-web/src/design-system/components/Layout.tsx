"use client";

import * as React from "react";
import { type ReactNode, type HTMLAttributes, forwardRef } from "react";
import { mergeStyles, layoutStyles } from "../utils/styles";

export interface ContainerProps extends HTMLAttributes<HTMLDivElement> {
  size?: "sm" | "md" | "lg" | "xl" | "full";
  padding?: boolean;
}

const sizeStyles: Record<Exclude<ContainerProps["size"], undefined>, React.CSSProperties> = {
  sm: { maxWidth: "640px" },
  md: { maxWidth: "960px" },
  lg: { maxWidth: "1200px" },
  xl: { maxWidth: "1400px" },
  full: { maxWidth: "100%" },
};

export const Container = forwardRef<HTMLDivElement, ContainerProps>(
  ({ size = "xl", padding = true, children, className, style, ...props }, ref) => {
    const combinedStyle = mergeStyles(
      layoutStyles.container,
      sizeStyles[size],
      padding && { paddingLeft: "var(--s-header-x)", paddingRight: "var(--s-header-x)" },
      style
    );

    return <div ref={ref} className={className} style={combinedStyle} {...props}>{children}</div>;
  }
);

Container.displayName = "Container";

export interface SectionProps extends HTMLAttributes<HTMLElement> {
  variant?: "default" | "narrow" | "wide";
  spacing?: "none" | "sm" | "md" | "lg" | "xl";
}

const spacingStyles: Record<Exclude<SectionProps["spacing"], undefined>, React.CSSProperties> = {
  none: {},
  sm: { marginBottom: "var(--s-4)" },
  md: { marginBottom: "var(--s-section-gap)" },
  lg: { marginBottom: "var(--s-9)" },
  xl: { marginBottom: "var(--s-11)" },
};

const variantStyles: Record<Exclude<SectionProps["variant"], undefined>, React.CSSProperties> = {
  default: {},
  narrow: { maxWidth: "720px", marginLeft: "auto", marginRight: "auto" },
  wide: { maxWidth: "1200px", marginLeft: "auto", marginRight: "auto" },
};

export function Section({ variant = "default", spacing = "md", children, className, style, ...props }: SectionProps) {
  return (
    <section className={className} style={mergeStyles(variantStyles[variant], spacingStyles[spacing], style)} {...props}>
      {children}
    </section>
  );
}

export interface FlexProps extends HTMLAttributes<HTMLDivElement> {
  direction?: "row" | "col" | "row-reverse" | "col-reverse";
  align?: "start" | "center" | "end" | "stretch" | "baseline";
  justify?: "start" | "center" | "end" | "between" | "around" | "evenly";
  gap?: keyof typeof import("../tokens").spacing.scale | number | string;
  wrap?: boolean;
  fullWidth?: boolean;
  fullHeight?: boolean;
}

const alignMap = { start: "flex-start", center: "center", end: "flex-end", stretch: "stretch", baseline: "baseline" };
const justifyMap = { start: "flex-start", center: "center", end: "flex-end", between: "space-between", around: "space-around", evenly: "space-evenly" };

export const Flex = forwardRef<HTMLDivElement, FlexProps>(
  ({ direction = "row", align = "stretch", justify = "start", gap, wrap = false, fullWidth = false, fullHeight = false, children, className, style, ...props }, ref) => {
    const gapValue = typeof gap === "number" ? `${gap}px` : gap ? `var(--s-${gap})` : undefined;

    const combinedStyle = mergeStyles(
      {
        display: "flex",
        flexDirection: direction as React.CSSProperties["flexDirection"],
        alignItems: alignMap[align] as React.CSSProperties["alignItems"],
        justifyContent: justifyMap[justify] as React.CSSProperties["justifyContent"],
        gap: gapValue,
        flexWrap: wrap ? "wrap" : "nowrap",
        width: fullWidth ? "100%" : undefined,
        height: fullHeight ? "100%" : undefined,
      },
      style
    );

    return <div ref={ref} className={className} style={combinedStyle} {...props}>{children}</div>;
  }
);

Flex.displayName = "Flex";

export interface GridProps extends HTMLAttributes<HTMLDivElement> {
  columns?: number | string;
  rows?: number | string;
  gap?: keyof typeof import("../tokens").spacing.scale | number | string;
  minColumnWidth?: string;
  autoFit?: boolean;
  autoFill?: boolean;
}

export const Grid = forwardRef<HTMLDivElement, GridProps>(
  ({ columns, rows, gap, minColumnWidth = "140px", autoFit = false, autoFill = false, children, className, style, ...props }, ref) => {
    const gapValue = typeof gap === "number" ? `${gap}px` : gap ? `var(--s-${gap})` : "var(--s-card-gap)";

    let gridTemplateColumns: string;
    if (autoFit) {
      gridTemplateColumns = `repeat(auto-fit, minmax(${minColumnWidth}, 1fr))`;
    } else if (autoFill) {
      gridTemplateColumns = `repeat(auto-fill, minmax(${minColumnWidth}, 1fr))`;
    } else if (typeof columns === "number") {
      gridTemplateColumns = `repeat(${columns}, 1fr)`;
    } else {
      gridTemplateColumns = columns || "repeat(auto-fill, minmax(140px, 1fr))";
    }

    const combinedStyle = mergeStyles(
      {
        display: "grid",
        gridTemplateColumns,
        gridTemplateRows: rows,
        gap: gapValue,
      },
      style
    );

    return <div ref={ref} className={className} style={combinedStyle} {...props}>{children}</div>;
  }
);

Grid.displayName = "Grid";

export interface StackProps extends HTMLAttributes<HTMLDivElement> {
  direction?: "vertical" | "horizontal";
  gap?: keyof typeof import("../tokens").spacing.scale | number | string;
  align?: "start" | "center" | "end" | "stretch";
  divider?: boolean;
  dividerColor?: string;
}

export const Stack = forwardRef<HTMLDivElement, StackProps>(
  ({ direction = "vertical", gap = "4", align = "stretch", divider = false, dividerColor, children, className, style, ...props }, ref) => {
    const gapValue = typeof gap === "number" ? `${gap}px` : `var(--s-${gap})`;
    const isHorizontal = direction === "horizontal";

    const childArray = React.Children.toArray(children);
    const childrenWithDividers = divider
      ? childArray.flatMap((child, index) => [
          child,
          index < childArray.length - 1 && (
            <div
              key={`divider-${index}`}
              style={{
                width: isHorizontal ? 1 : "100%",
                height: isHorizontal ? "100%" : 1,
                background: dividerColor || "var(--hairline)",
              }}
            />
          ),
        ])
      : childArray;

    const combinedStyle = mergeStyles(
      {
        display: "flex",
        flexDirection: isHorizontal ? "row" : "column",
        alignItems: align,
        gap: divider ? 0 : gapValue,
      },
      style
    );

    return <div ref={ref} className={className} style={combinedStyle} {...props}>{childrenWithDividers}</div>;
  }
);

Stack.displayName = "Stack";

export interface RailProps extends HTMLAttributes<HTMLDivElement> {
  gap?: keyof typeof import("../tokens").spacing.scale | number | string;
  paddingX?: boolean;
  paddingBottom?: boolean;
  scrollable?: boolean;
}

export function Rail({ gap = "cardGap", paddingX = true, paddingBottom = true, scrollable = true, children, className, style, ...props }: RailProps) {
  const gapValue = typeof gap === "number" ? `${gap}px` : `var(--s-${gap})`;

  return (
    <div
      className={className}
      style={{
        display: "flex",
        gap: gapValue,
        overflowX: scrollable ? "auto" : "visible",
        padding: `6px ${paddingX ? "var(--s-header-x)" : 0} ${paddingBottom ? "16px" : 0}`,
        scrollbarWidth: "none",
        WebkitOverflowScrolling: "touch",
        ...style,
      }}
      {...props}
    >
      {children}
    </div>
  );
}

export interface InlineProps extends HTMLAttributes<HTMLSpanElement> {
  gap?: keyof typeof import("../tokens").spacing.scale | number | string;
  align?: "start" | "center" | "end" | "baseline";
  wrap?: boolean;
}

export const Inline = forwardRef<HTMLSpanElement, InlineProps>(
  ({ gap = "2", align = "center", wrap = false, children, className, style, ...props }, ref) => {
    const gapValue = typeof gap === "number" ? `${gap}px` : `var(--s-${gap})`;

    return (
      <span
        ref={ref}
        className={className}
        style={{
          display: "inline-flex",
          flexWrap: wrap ? "wrap" : "nowrap",
          alignItems: align,
          gap: gapValue,
          ...style,
        }}
        {...props}
      >
        {children}
      </span>
    );
  }
);

Inline.displayName = "Inline";

export interface CenterProps extends HTMLAttributes<HTMLDivElement> {
  fullWidth?: boolean;
  fullHeight?: boolean;
}

export function Center({ fullWidth = false, fullHeight = false, children, className, style, ...props }: CenterProps) {
  return (
    <div
      className={className}
      style={{
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        width: fullWidth ? "100%" : undefined,
        height: fullHeight ? "100%" : undefined,
        ...style,
      }}
      {...props}
    >
      {children}
    </div>
  );
}

export interface AbsoluteCenterProps extends HTMLAttributes<HTMLDivElement> {
  inset?: number | string;
}

export function AbsoluteCenter({ inset = 0, children, className, style, ...props }: AbsoluteCenterProps) {
  return (
    <div
      className={className}
      style={{
        position: "absolute",
        inset: typeof inset === "number" ? `${inset}px` : inset,
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        ...style,
      }}
      {...props}
    >
      {children}
    </div>
  );
}

export interface VisuallyHiddenProps extends HTMLAttributes<HTMLSpanElement> {
  children: ReactNode;
}

export function VisuallyHidden({ children, ...props }: VisuallyHiddenProps) {
  return (
    <span
      {...props}
      style={{
        position: "absolute",
        width: 1,
        height: 1,
        padding: 0,
        margin: -1,
        overflow: "hidden",
        clip: "rect(0, 0, 0, 0)",
        whiteSpace: "nowrap",
        border: 0,
      }}
    >
      {children}
    </span>
  );
}