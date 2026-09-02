"use client";

import { type ReactNode } from "react";

export interface FilterBarProps {
  children: ReactNode;
  sticky?: boolean;
}

/**
 * Horizontally scrollable filter bar
 */
export function FilterBar({ children, sticky = false }: FilterBarProps) {
  return (
    <div
      className="filter-bar"
      style={{
        display: "flex",
        alignItems: "center",
        gap: 8,
        overflowX: "auto",
        scrollbarWidth: "none",
        padding: "8px var(--s-header-x) 12px",
        borderBottom: "1px solid rgba(255,255,255,0.06)",
        background: sticky ? "var(--color-bg)" : "transparent",
        position: sticky ? "sticky" : "relative",
        top: sticky ? 0 : undefined,
        zIndex: sticky ? 5 : undefined,
      }}
    >
      {children}
      <style>{`.filter-bar::-webkit-scrollbar{display:none}`}</style>
    </div>
  );
}

export function FilterPillSelect({
  value,
  onChange,
  options,
  active,
}: {
  value: string;
  onChange: (v: string) => void;
  options: Array<{ value: string; label: string }>;
  active?: boolean;
}) {
  const isActive = active ?? (value !== "" && value !== options[0]?.value);
  return (
    <select
      value={value}
      onChange={(e) => onChange(e.target.value)}
      className="filter-select"
      data-active={isActive ? "true" : undefined}
    >
      {options.map((o) => (
        <option key={o.value} value={o.value}>
          {o.label}
        </option>
      ))}
    </select>
  );
}
