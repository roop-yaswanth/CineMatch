"use client";

import { type ReactNode, type HTMLAttributes, useState, useCallback, useRef, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { mergeStyles } from "../utils/styles";
import { Button } from "./Button";
import { Flex } from "./Layout";

export interface TabItem {
  id: string;
  label: ReactNode;
  disabled?: boolean;
  badge?: ReactNode;
  icon?: ReactNode;
}

export interface TabsProps extends Omit<HTMLAttributes<HTMLDivElement>, "onChange"> {
  tabs: TabItem[];
  activeTab: string;
  onChange: (tabId: string) => void;
  variant?: "default" | "pills" | "underline" | "segmented";
  size?: "sm" | "md" | "lg";
  fullWidth?: boolean;
  scrollable?: boolean;
  indicator?: boolean;
}

const variantStyles = {
  default: {
    tab: (active: boolean) => ({
      padding: "8px 16px",
      fontSize: "var(--fs-sm)",
      fontWeight: active ? 600 : 500,
      color: active ? "var(--color-text-primary)" : "var(--color-text-secondary)",
      background: active ? "rgba(255, 255, 255, 0.06)" : "transparent",
      borderRadius: "var(--radius-pill)",
      border: "1px solid transparent",
      transition: "all var(--dur-base) var(--ease-out)",
    }),
    container: { gap: "4px", background: "transparent" },
  },
  pills: {
    tab: (active: boolean) => ({
      padding: "8px 20px",
      fontSize: "var(--fs-sm)",
      fontWeight: active ? 600 : 500,
      color: active ? "#0a0a0f" : "var(--color-text-secondary)",
      background: active ? "linear-gradient(180deg, rgba(250, 250, 250, 0.95) 0%, rgba(245, 245, 245, 0.95) 100%)" : "rgba(255, 255, 255, 0.04)",
      border: active ? "1px solid rgba(255, 255, 255, 0.3)" : "1px solid var(--hairline)",
      borderRadius: "var(--radius-pill)",
      backdropFilter: "blur(12px) saturate(1.5)",
      WebkitBackdropFilter: "blur(12px) saturate(1.5)",
      boxShadow: active ? "0 4px 14px rgba(0, 0, 0, 0.3), 0 1px 0 rgba(255, 255, 255, 0.15) inset" : "0 1px 0 rgba(255, 255, 255, 0.05) inset",
      transition: "all var(--dur-base) var(--ease-out)",
    }),
    container: { gap: "8px", background: "transparent" },
  },
  underline: {
    tab: (active: boolean) => ({
      padding: "12px 16px",
      fontSize: "var(--fs-sm)",
      fontWeight: active ? 600 : 500,
      color: active ? "var(--color-text-primary)" : "var(--color-text-muted)",
      background: "transparent",
      borderRadius: 0,
      borderBottom: active ? "2px solid var(--color-accent)" : "2px solid transparent",
      marginBottom: -2,
      transition: "all var(--dur-base) var(--ease-out)",
    }),
    container: { gap: 0, background: "transparent", borderBottom: "1px solid var(--hairline)" },
  },
  segmented: {
    tab: (active: boolean) => ({
      padding: "10px 20px",
      fontSize: "var(--fs-sm)",
      fontWeight: active ? 600 : 500,
      color: active ? "var(--color-text-primary)" : "var(--color-text-secondary)",
      background: active ? "rgba(255, 255, 255, 0.1)" : "transparent",
      borderRadius: "var(--radius-md)",
      border: active ? "1px solid var(--hairline-strong)" : "1px solid transparent",
      transition: "all var(--dur-base) var(--ease-out)",
    }),
    container: { gap: "4px", background: "var(--glass-chrome)", borderRadius: "var(--radius-lg)", padding: "4px", border: "1px solid var(--hairline)" },
  },
};

const sizeStyles = {
  sm: { fontSize: "var(--fs-xs)", paddingScale: 0.8 },
  md: { fontSize: "var(--fs-sm)", paddingScale: 1 },
  lg: { fontSize: "var(--fs-md)", paddingScale: 1.2 },
};

export function Tabs({
  tabs,
  activeTab,
  onChange,
  variant = "default",
  size = "md",
  fullWidth = false,
  scrollable = true,
  indicator = false,
  className,
  style,
  children,
  ...props
}: TabsProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const indicatorRef = useRef<HTMLDivElement>(null);
  const [indicatorStyle, setIndicatorStyle] = useState<React.CSSProperties>({});

  const updateIndicator = useCallback(() => {
    if (!indicator || variant !== "underline" || !containerRef.current || !indicatorRef.current) return;

    const activeButton = containerRef.current.querySelector(`[data-tab-id="${activeTab}"]`) as HTMLElement;
    if (activeButton) {
      const containerRect = containerRef.current.getBoundingClientRect();
      const buttonRect = activeButton.getBoundingClientRect();
      setIndicatorStyle({
        width: buttonRect.width,
        left: buttonRect.left - containerRect.left,
        opacity: 1,
      });
    }
  }, [activeTab, indicator, variant]);

  useEffect(() => {
    updateIndicator();
    window.addEventListener("resize", updateIndicator);
    return () => window.removeEventListener("resize", updateIndicator);
  }, [updateIndicator]);

  const vStyles = variantStyles[variant];
  const sStyles = sizeStyles[size];

  const tabButtonStyle = (tab: TabItem) => {
    const base = vStyles.tab(tab.id === activeTab);
    return mergeStyles(base, { fontSize: sStyles.fontSize }, tab.disabled && { opacity: 0.4, cursor: "not-allowed", pointerEvents: "none" });
  };

  const indicatorStyleValue = {
    position: "absolute",
    bottom: 0,
    height: 2,
    background: "var(--color-accent)",
    borderRadius: "1px 1px 0 0",
    transition: "all var(--dur-base) var(--ease-out)",
    ...indicatorStyle,
  } as React.CSSProperties;

  return (
    <div className={className} style={{ width: fullWidth ? "100%" : "auto", ...style }} {...props}>
      <div
        ref={containerRef}
        style={{
          display: "flex",
          alignItems: "center",
          overflowX: scrollable ? "auto" : "visible",
          scrollbarWidth: "none",
          WebkitOverflowScrolling: "touch",
          ...vStyles.container,
        }}
        role="tablist"
        aria-label="Tabs"
      >
        {variant === "underline" && indicator && (
          <div
            ref={indicatorRef}
            style={indicatorStyleValue}
          />
        )}
        {tabs.map((tab) => (
          <motion.button
            key={tab.id}
            type="button"
            role="tab"
            aria-selected={tab.id === activeTab}
            aria-disabled={tab.disabled}
            id={`tab-${tab.id}`}
            data-tab-id={tab.id}
            onClick={() => !tab.disabled && onChange(tab.id)}
            style={tabButtonStyle(tab)}
            whileTap={{ scale: 0.98 }}
            whileHover={{ scale: tab.disabled ? 1 : 1.02 }}
          >
            <Flex gap="2" align="center">
              {tab.icon && <span style={{ display: "inline-flex" }}>{tab.icon}</span>}
              <span>{tab.label}</span>
              {tab.badge && <span style={{ fontSize: "var(--fs-2xs)", padding: "1px 6px", borderRadius: "999px", background: "var(--color-accent)", color: "#0a0a0f" }}>{tab.badge}</span>}
            </Flex>
          </motion.button>
        ))}
      </div>
      {children}
    </div>
  );
}

export interface TabPanelProps extends Omit<HTMLAttributes<HTMLDivElement>, "onAnimationStart" | "onAnimationEnd" | "onAnimationIteration"> {
  tabId: string;
  activeTab: string;
  animate?: boolean;
}

export function TabPanel({ tabId, activeTab, animate = true, children, className, style }: TabPanelProps) {
  if (tabId !== activeTab) return null;

  if (animate) {
    return (
      <AnimatePresence mode="wait">
        <motion.div
          key={tabId}
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -10 }}
          transition={{ duration: 0.25, ease: [0.22, 1, 0.36, 1] }}
          role="tabpanel"
          aria-labelledby={`tab-${tabId}`}
          className={className}
          style={style}
        >
          {children}
        </motion.div>
      </AnimatePresence>
    );
  }

  return (
    <div role="tabpanel" aria-labelledby={`tab-${tabId}`} className={className} style={style}>
      {children}
    </div>
  );
}

export interface TabViewProps extends Omit<HTMLAttributes<HTMLDivElement>, "onChange"> {
  tabs: TabItem[];
  defaultTab?: string;
  variant?: "default" | "pills" | "underline" | "segmented";
  size?: "sm" | "md" | "lg";
  fullWidth?: boolean;
  renderPanel: (tab: TabItem) => ReactNode;
  onChange?: (tabId: string) => void;
}

export function TabView({ tabs, defaultTab, variant = "default", size = "md", fullWidth = false, renderPanel, onChange, className, style, ...props }: TabViewProps) {
  const [activeTab, setActiveTab] = useState(defaultTab || tabs[0]?.id || "");

  const handleChange = useCallback((tabId: string) => {
    setActiveTab(tabId);
    onChange?.(tabId);
  }, [onChange]);

  return (
    <div className={className} style={style} {...props}>
      <Tabs tabs={tabs} activeTab={activeTab} onChange={handleChange} variant={variant} size={size} fullWidth={fullWidth} />
      <div style={{ marginTop: "var(--s-5)" }}>
        {tabs.map((tab) => (
          <TabPanel key={tab.id} tabId={tab.id} activeTab={activeTab}>
            {renderPanel(tab)}
          </TabPanel>
        ))}
      </div>
    </div>
  );
}

export interface DropdownProps extends Omit<HTMLAttributes<HTMLDivElement>, "onSelect"> {
  trigger: ReactNode;
  items: Array<{ id: string; label: ReactNode; disabled?: boolean; danger?: boolean; icon?: ReactNode; shortcut?: string }>;
  onSelect: (itemId: string) => void;
  align?: "left" | "right";
  width?: string | number;
  closeOnSelect?: boolean;
}

export function Dropdown({
  trigger,
  items,
  onSelect,
  align = "left",
  width = 240,
  closeOnSelect = true,
  className,
  style,
  ...props
}: DropdownProps) {
  const [open, setOpen] = useState(false);
  const dropdownRef = useRef<HTMLDivElement>(null);
  const triggerRef = useRef<HTMLButtonElement>(null);

  useEffect(() => {
    const handleClickOutside = (e: MouseEvent) => {
      if (dropdownRef.current && !dropdownRef.current.contains(e.target as Node) && triggerRef.current && !triggerRef.current.contains(e.target as Node)) {
        setOpen(false);
      }
    };

    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  const handleItemClick = (itemId: string) => {
    onSelect(itemId);
    if (closeOnSelect) setOpen(false);
  };

  return (
    <div ref={dropdownRef} className={className} style={{ position: "relative", display: "inline-flex", ...style }} {...props}>
      <Button
        ref={triggerRef}
        variant="glass"
        onClick={() => setOpen(!open)}
        rightIcon={<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><polyline points="6 9 12 15 18 9" /></svg>}
        aria-haspopup="true"
        aria-expanded={open}
      >
        {trigger}
      </Button>

      <AnimatePresence>
        {open && (
          <motion.div
            initial={{ opacity: 0, y: -8, scale: 0.98 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: -8, scale: 0.98 }}
            transition={{ duration: 0.15, ease: "easeOut" }}
            style={{
              position: "absolute",
              top: "calc(100% + 8px)",
              [align]: 0,
              zIndex: 50,
              minWidth: typeof width === "number" ? `${width}px` : width,
              background: "rgba(20, 22, 28, 0.96)",
              backdropFilter: "blur(40px) saturate(1.6)",
              WebkitBackdropFilter: "blur(40px) saturate(1.6)",
              border: "1px solid var(--hairline)",
              borderRadius: "var(--radius-lg)",
              boxShadow: "var(--shadow-lg)",
              overflow: "hidden",
              padding: "4px",
            }}
            role="menu"
          >
            {items.map((item) => (
              <motion.button
                key={item.id}
                type="button"
                role="menuitem"
                disabled={item.disabled}
                onClick={() => !item.disabled && handleItemClick(item.id)}
                style={{
                  width: "100%",
                  display: "flex",
                  alignItems: "center",
                  gap: "var(--s-3)",
                  padding: "10px 12px",
                  borderRadius: "var(--radius-md)",
                  background: "transparent",
                  border: "none",
                  color: item.danger ? "var(--color-danger)" : "var(--color-text-primary)",
                  fontSize: "var(--fs-sm)",
                  fontWeight: 500,
                  textAlign: "left",
                  cursor: item.disabled ? "not-allowed" : "pointer",
                  opacity: item.disabled ? 0.4 : 1,
                  transition: "background-color var(--dur-fast) var(--ease-out)",
                }}
                whileTap={{ scale: 0.98 }}
                whileHover={{ background: item.danger ? "rgba(255, 69, 58, 0.1)" : "var(--glass-chrome)" }}
              >
                {item.icon && <span style={{ display: "inline-flex", flexShrink: 0 }}>{item.icon}</span>}
                <span style={{ flex: 1 }}>{item.label}</span>
                {item.shortcut && <span style={{ fontSize: "var(--fs-2xs)", color: "var(--color-text-muted)", fontVariantNumeric: "tabular-nums" }}>{item.shortcut}</span>}
              </motion.button>
            ))}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

export interface DesignSystemSelectProps extends Omit<HTMLAttributes<HTMLDivElement>, "onChange"> {
  value: string;
  placeholder?: string;
  options: Array<{ value: string; label: ReactNode; disabled?: boolean }>;
  onChange: (value: string) => void;
  width?: string | number;
  size?: "sm" | "md" | "lg";
}

export function DesignSystemSelect({ value, placeholder, options, onChange, width = 200, size = "md", className, style, ...props }: DesignSystemSelectProps) {
  const [open, setOpen] = useState(false);
  const selectRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const handleClickOutside = (e: MouseEvent) => {
      if (selectRef.current && !selectRef.current.contains(e.target as Node)) {
        setOpen(false);
      }
    };
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  const sizeStyles = {
    sm: { padding: "8px 32px 8px 12px", fontSize: "var(--fs-sm)", height: 36 },
    md: { padding: "10px 36px 10px 14px", fontSize: "var(--fs-sm)", height: 44 },
    lg: { padding: "12px 40px 12px 16px", fontSize: "var(--fs-md)", height: 52 },
  };

  const selectedOption = options.find((o) => o.value === value);

  return (
    <div ref={selectRef} className={className} style={{ position: "relative", display: "inline-flex", width: typeof width === "number" ? `${width}px` : width, ...style }} {...props}>
      <button
        type="button"
        onClick={() => setOpen(!open)}
        aria-haspopup="listbox"
        aria-expanded={open}
        style={{
          width: "100%",
          height: sizeStyles[size].height,
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          padding: sizeStyles[size].padding,
          background: "rgba(28, 30, 36, 0.82)",
          border: "1px solid var(--hairline)",
          borderRadius: "var(--radius-pill)",
          color: value ? "var(--color-text-primary)" : "var(--color-text-muted)",
          fontSize: sizeStyles[size].fontSize,
          fontWeight: 500,
          cursor: "pointer",
          outline: "none",
          transition: "border-color var(--dur-base) var(--ease-out), background var(--dur-base) var(--ease-out), box-shadow var(--dur-base) var(--ease-out)",
        }}
        onFocus={(e) => {
          e.currentTarget.style.borderColor = "rgba(255, 255, 255, 0.30)";
          e.currentTarget.style.background = "rgba(36, 38, 46, 0.92)";
          e.currentTarget.style.boxShadow = "0 0 0 3px rgba(255, 255, 255, 0.08)";
        }}
        onBlur={(e) => {
          e.currentTarget.style.borderColor = "var(--hairline)";
          e.currentTarget.style.background = "rgba(28, 30, 36, 0.82)";
          e.currentTarget.style.boxShadow = "none";
        }}
      >
        <span style={{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{selectedOption?.label || placeholder || "Select"}</span>
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={{ flexShrink: 0, marginLeft: 8, color: "var(--color-text-muted)" }}>
          <polyline points="6 9 12 15 18 9" />
        </svg>
      </button>

      <AnimatePresence>
        {open && (
          <motion.div
            initial={{ opacity: 0, y: -8, scale: 0.98 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: -8, scale: 0.98 }}
            transition={{ duration: 0.15, ease: "easeOut" }}
            style={{
              position: "absolute",
              top: "calc(100% + 6px)",
              left: 0,
              right: 0,
              zIndex: 50,
              background: "rgba(20, 22, 28, 0.96)",
              backdropFilter: "blur(40px) saturate(1.6)",
              WebkitBackdropFilter: "blur(40px) saturate(1.6)",
              border: "1px solid var(--hairline)",
              borderRadius: "var(--radius-lg)",
              boxShadow: "var(--shadow-lg)",
              overflow: "hidden",
              maxHeight: "280px",
              overflowY: "auto",
              padding: "4px",
            }}
            role="listbox"
          >
            {options.map((option) => (
              <button
                key={option.value}
                type="button"
                role="option"
                aria-selected={option.value === value}
                disabled={option.disabled}
                onClick={() => { if (!option.disabled) { onChange(option.value); setOpen(false); } }}
                style={{
                  width: "100%",
                  display: "flex",
                  alignItems: "center",
                  padding: "10px 12px",
                  borderRadius: "var(--radius-md)",
                  background: option.value === value ? "rgba(var(--rgb-accent), 0.12)" : "transparent",
                  border: "none",
                  color: "var(--color-text-primary)",
                  fontSize: "var(--fs-sm)",
                  fontWeight: option.value === value ? 600 : 500,
                  textAlign: "left",
                  cursor: option.disabled ? "not-allowed" : "pointer",
                  opacity: option.disabled ? 0.4 : 1,
                }}
              >
                {option.label}
              </button>
            ))}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}