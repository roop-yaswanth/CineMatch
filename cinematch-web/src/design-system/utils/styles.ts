import type { CSSProperties } from 'react';

export type TokenValue = string | number;
export type StyleObject = CSSProperties & {
  [key: `--${string}`]: TokenValue;
};

export const cssVar = (name: string, value: TokenValue): StyleObject => ({
  [name.startsWith('--') ? name : `--${name}`]: value,
});

export const token = {
  color: (path: string) => `var(--color-${path.replace(/\./g, '-')})`,
  spacing: (path: string) => `var(--s-${path})`,
  radius: (path: string) => `var(--radius-${path})`,
  shadow: (path: string) => `var(--shadow-${path})`,
  motion: {
    easing: (path: string) => `var(--ease-${path})`,
    duration: (path: string) => `var(--dur-${path})`,
  },
  blur: (path: string) => `var(--blur-${path})`,
  typography: {
    size: (path: string) => `var(--fs-${path})`,
    lineHeight: (path: string) => `var(--lh-${path})`,
    tracking: (path: string) => `var(--tracking-${path})`,
  },
  zIndex: (path: string) => `var(--z-${path})`,
} as const;

export function createStyleObject(styles: Record<string, TokenValue>): StyleObject {
  const result: StyleObject = {};
  for (const [key, value] of Object.entries(styles)) {
    if (key.startsWith('--')) {
      result[key as `--${string}`] = value;
    } else {
      // TokenValue is string | number — safe for CSSProperties values.
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      result[key as keyof CSSProperties] = value as any;
    }
  }
  return result;
}

export function mergeStyles(...styles: (StyleObject | CSSProperties | boolean | undefined)[]): React.CSSProperties {
  return Object.assign({}, ...styles.filter((s): s is StyleObject | CSSProperties => Boolean(s))) as React.CSSProperties;
}

export const glassStyles = {
  thin: {
    background: 'rgba(22, 22, 32, 0.35)',
    backdropFilter: 'blur(12px) saturate(1.4)',
    WebkitBackdropFilter: 'blur(12px) saturate(1.4)',
    border: '1px solid rgba(255, 255, 255, 0.08)',
    borderRadius: 'var(--radius-md)',
  },
  medium: {
    background: 'rgba(20, 22, 28, 0.72)',
    backdropFilter: 'blur(40px) saturate(1.6)',
    WebkitBackdropFilter: 'blur(40px) saturate(1.6)',
    border: '1px solid rgba(255, 255, 255, 0.10)',
    boxShadow: '0 12px 36px rgba(0, 0, 0, 0.55), 0 1px 0 rgba(255, 255, 255, 0.10) inset',
  },
  strong: {
    background: 'linear-gradient(145deg, rgba(24, 26, 32, 0.96) 0%, rgba(14, 16, 20, 0.98) 50%, rgba(18, 20, 24, 0.97) 100%)',
    backdropFilter: 'blur(60px) saturate(2.2) brightness(1.08)',
    WebkitBackdropFilter: 'blur(60px) saturate(2.2) brightness(1.08)',
    boxShadow: '0 40px 80px -20px rgba(0, 0, 0, 0.7), 0 18px 40px -10px rgba(0, 0, 0, 0.45), 0 0 0 0.5px rgba(255, 255, 255, 0.08) inset',
  },
  card: {
    position: 'relative',
    background: 'linear-gradient(145deg, rgba(38, 40, 46, 0.55) 0%, rgba(22, 24, 28, 0.65) 100%)',
    backdropFilter: 'blur(40px) saturate(1.9) brightness(1.05)',
    WebkitBackdropFilter: 'blur(40px) saturate(1.9) brightness(1.05)',
    borderRadius: 'var(--radius-card)',
    boxShadow: '0 4px 14px rgba(0, 0, 0, 0.35), 0 1px 2px rgba(0, 0, 0, 0.4), 0 0 0 0.5px rgba(255, 255, 255, 0.1) inset, 0 1px 0 0 rgba(255, 255, 255, 0.14) inset',
    transition: 'transform var(--dur-base) var(--ease-spring), box-shadow var(--dur-base) var(--ease-out), filter var(--dur-base) var(--ease-out)',
    overflow: 'hidden',
  },
  modal: {
    background: 'linear-gradient(145deg, rgba(24, 26, 32, 0.96) 0%, rgba(14, 16, 20, 0.98) 50%, rgba(18, 20, 24, 0.97) 100%)',
    backdropFilter: 'blur(60px) saturate(2.2) brightness(1.08)',
    WebkitBackdropFilter: 'blur(60px) saturate(2.2) brightness(1.08)',
    borderRadius: 'var(--radius-modal)',
    boxShadow: '0 40px 80px -20px rgba(0, 0, 0, 0.7), 0 18px 40px -10px rgba(0, 0, 0, 0.45)',
    position: 'relative',
    overflow: 'hidden',
  },
  header: {
    background: 'linear-gradient(180deg, rgba(34, 36, 44, 0.5) 0%, rgba(16, 18, 22, 0.66) 100%)',
    backdropFilter: 'blur(30px) saturate(200%) brightness(1.12)',
    WebkitBackdropFilter: 'blur(30px) saturate(200%) brightness(1.12)',
    borderBottom: '1px solid rgba(255, 255, 255, 0.12)',
    boxShadow: '0 1px 0 0 rgba(255, 255, 255, 0.18) inset, 0 8px 32px rgba(0, 0, 0, 0.32)',
    position: 'relative',
  },
} as const;

export const buttonStyles = {
  base: {
    display: 'inline-flex',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 'var(--s-2)',
    padding: '11px 22px',
    borderRadius: 'var(--radius-pill)',
    fontSize: 'var(--fs-md)',
    fontWeight: 600,
    letterSpacing: 'var(--tracking-snug)',
    lineHeight: 1,
    cursor: 'pointer',
    border: 'none',
    textDecoration: 'none',
    transition: 'background var(--dur-base) var(--ease-out), color var(--dur-base) var(--ease-out), transform var(--dur-fast) var(--ease-out), opacity var(--dur-base) var(--ease-out)',
    WebkitTapHighlightColor: 'transparent',
  },
  primary: {
    background: '#fff',
    color: '#0a0a0f',
    boxShadow: '0 6px 22px rgba(0, 0, 0, 0.45)',
  },
  secondary: {
    background: 'rgba(28, 30, 36, 0.66)',
    color: 'var(--color-text-primary)',
    border: '1px solid rgba(255, 255, 255, 0.18)',
    backdropFilter: 'blur(20px) saturate(1.4)',
    WebkitBackdropFilter: 'blur(20px) saturate(1.4)',
  },
  ghost: {
    background: 'transparent',
    color: 'var(--color-text-secondary)',
    padding: '8px 14px',
  },
  sm: {
    padding: '7px 14px',
    fontSize: 'var(--fs-sm)',
  },
} as const;

export const inputStyles = {
  base: {
    width: '100%',
    padding: '13px 40px 13px 44px',
    borderRadius: '14px',
    border: '1px solid rgba(255, 255, 255, 0.10)',
    background: 'rgba(28, 30, 36, 0.82)',
    color: 'var(--color-text-primary)',
    fontSize: '16px',
    fontWeight: 400,
    letterSpacing: '-0.005em',
    outline: 'none',
    transition: 'border-color var(--dur-base) var(--ease-out), background var(--dur-base) var(--ease-out), box-shadow var(--dur-base) var(--ease-out)',
    WebkitAppearance: 'none',
    appearance: 'none',
  },
  focus: {
    borderColor: 'rgba(255, 255, 255, 0.30)',
    background: 'rgba(36, 38, 46, 0.92)',
    boxShadow: '0 0 0 3px rgba(255, 255, 255, 0.08), 0 1px 0 0 rgba(255, 255, 255, 0.06) inset',
  },
} as const;

export const cardStyles = {
  poster: {
    position: 'relative',
    borderRadius: 'var(--radius-poster)',
    overflow: 'hidden',
    background: 'rgba(20, 20, 28, 0.4)',
    isolation: 'isolate',
  },
  posterHover: {
    transform: 'translateY(-4px) scale(1.02)',
  },
} as const;

export const typographyStyles = {
  display: {
    fontWeight: 700,
    letterSpacing: '-0.035em',
    lineHeight: 1.05,
  },
  h1: {
    fontSize: 'clamp(20px, 4.5vw, 28px)',
    fontWeight: 700,
    letterSpacing: 'var(--tracking-tight)',
    lineHeight: 'var(--lh-tight)',
    margin: 0,
    color: 'var(--color-text-primary)',
  },
  h2: {
    fontSize: 'var(--fs-lg)',
    fontWeight: 600,
    letterSpacing: '-0.02em',
    lineHeight: 'var(--lh-tight)',
    color: 'var(--color-text-primary)',
    margin: 0,
  },
  h3: {
    fontSize: 'var(--fs-md)',
    fontWeight: 600,
    letterSpacing: '-0.01em',
    lineHeight: 'var(--lh-tight)',
    color: 'var(--color-text-primary)',
    margin: 0,
  },
  eyebrow: {
    fontSize: 'var(--fs-xs)',
    fontWeight: 600,
    letterSpacing: 'var(--tracking-wider)',
    textTransform: 'uppercase',
    color: 'var(--color-text-muted)',
    margin: 0,
  },
  body: {
    fontSize: 'var(--fs-md)',
    lineHeight: 'var(--lh-base)',
    color: 'var(--color-text-secondary)',
  },
  meta: {
    fontSize: 'var(--fs-sm)',
    color: 'var(--color-text-muted)',
  },
  title: {
    margin: 0,
    fontSize: '13px',
    fontWeight: 600,
    letterSpacing: '-0.01em',
    lineHeight: 1.35,
    color: 'var(--color-text-primary)',
    display: '-webkit-box',
    WebkitLineClamp: 2,
    lineClamp: 2,
    WebkitBoxOrient: 'vertical',
    overflow: 'hidden',
    minHeight: '2.7em',
  },
  metaText: {
    minWidth: 0,
    overflow: 'hidden',
    textOverflow: 'ellipsis',
    whiteSpace: 'nowrap',
    fontSize: '10.5px',
    fontWeight: 500,
    lineHeight: 1.2,
    color: 'var(--color-text-muted)',
  },
} as const;

export const layoutStyles = {
  container: {
    width: '100%',
    maxWidth: '1400px',
    margin: '0 auto',
    padding: '0 var(--s-header-x)',
  },
  section: {
    marginBottom: 'var(--s-section-gap)',
  },
  rail: {
    display: 'flex',
    gap: 'var(--s-card-gap)',
    overflowX: 'auto',
    padding: '6px var(--s-header-x) 16px',
    scrollbarWidth: 'none',
    WebkitOverflowScrolling: 'touch',
  },
  grid: {
    display: 'grid',
    gridTemplateColumns: 'repeat(auto-fill, minmax(140px, 1fr))',
    gap: '20px 14px',
  },
} as const;

export const badgeStyles = {
  base: {
    display: 'inline-flex',
    alignItems: 'center',
    gap: '4px',
    padding: '2px 8px',
    borderRadius: '999px',
    fontSize: '9px',
    fontWeight: 700,
    letterSpacing: '0.02em',
    textTransform: 'uppercase',
  },
  status: {
    upcoming: {
      background: 'rgba(99, 102, 241, 0.88)',
      color: '#ffffff',
    },
    theatres: {
      background: 'rgba(16, 185, 129, 0.88)',
      color: '#ffffff',
    },
  },
  rating: {
    position: 'absolute',
    top: '8px',
    right: '8px',
    zIndex: 3,
    display: 'inline-flex',
    alignItems: 'center',
    gap: '3px',
    padding: '3px 7px',
    borderRadius: '7px',
    background: 'rgba(0, 0, 0, 0.55)',
    backdropFilter: 'blur(8px)',
    WebkitBackdropFilter: 'blur(8px)',
    color: 'var(--color-rating)',
    fontSize: '10px',
    fontWeight: 700,
    lineHeight: 1.4,
    whiteSpace: 'nowrap',
    letterSpacing: '0.01em',
    pointerEvents: 'none',
  },
} as const;

export const skeletonStyles = {
  base: {
    position: 'relative',
    overflow: 'hidden',
    background: 'rgba(255, 255, 255, 0.04)',
    borderRadius: '12px',
  },
  shimmer: {
    position: 'absolute',
    inset: 0,
    background: 'linear-gradient(90deg, transparent 0%, rgba(255, 255, 255, 0.06) 50%, transparent 100%)',
    backgroundSize: '200% 100%',
    animation: 'shimmer 1.6s linear infinite',
  },
  grain: {
    position: 'relative',
    overflow: 'hidden',
  },
} as const;