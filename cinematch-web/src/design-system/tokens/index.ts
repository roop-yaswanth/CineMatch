export const colors = {
  bg: '#050507',
  bgRaised: '#0c0c11',

  glass: {
    thin: 'rgba(22, 22, 32, 0.35)',
    regular: 'rgba(22, 22, 32, 0.55)',
    thick: 'rgba(16, 16, 24, 0.72)',
    chrome: 'rgba(255, 255, 255, 0.04)',
    chromeStrong: 'rgba(255, 255, 255, 0.08)',
  },

  surface: 'rgba(22, 22, 28, 0.72)',
  surfaceHover: 'rgba(32, 32, 40, 0.78)',
  surfaceElevated: 'rgba(38, 38, 48, 0.82)',

  border: {
    hairline: 'rgba(255, 255, 255, 0.08)',
    hairlineStrong: 'rgba(255, 255, 255, 0.14)',
    hairlineAccent: 'rgba(255, 255, 255, 0.22)',
    default: 'rgba(255, 255, 255, 0.1)',
    subtle: 'rgba(255, 255, 255, 0.06)',
  },

  text: {
    primary: '#f5f5f7',
    secondary: '#b8b8bf',
    muted: '#7c7c85',
    faint: '#55555c',
  },

  accent: {
    default: '#ffffff',
    strong: '#ffffff',
    warm: '#ffb84d',
    rgb: '255, 255, 255',
    gradient: 'linear-gradient(135deg, #ffffff 0%, #e5e5ea 50%, #ffffff 100%)',
  },

  system: {
    blue: '#0a84ff',
    green: '#30d158',
    red: '#ff453a',
    yellow: '#ffd60a',
    orange: '#ff9f0a',
    purple: '#8e8e93',
    pink: '#ff375f',
  },

  rating: {
    success: '#30d158',
    danger: '#ff453a',
    love: '#30d158',
    like: '#facc15',
    dislike: '#ff453a',
    skip: '#8e8e93',
    gold: '#e8c84a',
    goldRgb: '232, 200, 74',
    loveRgb: '48, 209, 88',
    likeRgb: '250, 204, 21',
    dislikeRgb: '255, 69, 58',
    skipRgb: '142, 142, 147',
  },
} as const;

export const spacing = {
  headerY: '12px',
  headerX: '20px',
  sectionGap: '36px',
  cardGap: '16px',
  bottomClearance: 'calc(120px + env(safe-area-inset-bottom))',
  posterW: 'min(38vw, 150px)',

  scale: {
    1: '4px',
    2: '8px',
    3: '12px',
    4: '16px',
    5: '20px',
    6: '24px',
    7: '32px',
    8: '40px',
    9: '48px',
    10: '64px',
    11: '80px',
    12: '96px',
  },
} as const;

export const radius = {
  xs: '6px',
  sm: '10px',
  md: '14px',
  lg: '20px',
  xl: '28px',
  '2xl': '36px',
  card: '18px',
  poster: '14px',
  modal: '28px',
  pill: '9999px',
} as const;

export const shadows = {
  sm: '0 1px 2px rgba(0, 0, 0, 0.25), 0 0 0 0.5px rgba(255, 255, 255, 0.04) inset',
  md: '0 4px 14px rgba(0, 0, 0, 0.35), 0 1px 2px rgba(0, 0, 0, 0.4), 0 0 0 0.5px rgba(255, 255, 255, 0.06) inset',
  lg: '0 20px 50px -12px rgba(0, 0, 0, 0.55), 0 6px 18px rgba(0, 0, 0, 0.35), 0 0 0 0.5px rgba(255, 255, 255, 0.06) inset',
  xl: '0 40px 80px -20px rgba(0, 0, 0, 0.7), 0 18px 40px -10px rgba(0, 0, 0, 0.45), 0 0 0 0.5px rgba(255, 255, 255, 0.08) inset',
  glow: {
    blue: '0 0 40px rgba(10, 132, 255, 0.35)',
    like: '0 0 32px rgba(255, 45, 85, 0.40)',
    dislike: '0 0 32px rgba(255, 69, 58, 0.35)',
  },
} as const;

export const motion = {
  easing: {
    out: 'cubic-bezier(0.22, 1, 0.36, 1)',
    inOut: 'cubic-bezier(0.65, 0, 0.35, 1)',
    spring: 'cubic-bezier(0.34, 1.56, 0.64, 1)',
  },
  duration: {
    fast: '140ms',
    base: '220ms',
    slow: '360ms',
  },
} as const;

export const blur = {
  thin: '12px',
  regular: '24px',
  thick: '40px',
  chrome: '60px',
} as const;

export const typography = {
  fontFamily: 'var(--font-inter), -apple-system, "SF Pro Display", "SF Pro Text", BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif',

  scale: {
    '2xs': '10px',
    xs: '11px',
    sm: '12px',
    base: '13px',
    md: '14px',
    lg: '16px',
    xl: '19px',
    '2xl': '24px',
    '3xl': '32px',
    '4xl': '44px',
  },

  lineHeight: {
    tight: '1.15',
    snug: '1.35',
    base: '1.55',
    loose: '1.7',
  },

  tracking: {
    tight: '-0.025em',
    snug: '-0.01em',
    wide: '0.04em',
    wider: '0.08em',
  },

  fontWeight: {
    normal: 400,
    medium: 500,
    semibold: 600,
    bold: 700,
  },
} as const;

export const breakpoints = {
  sm: '640px',
  md: '768px',
  lg: '1024px',
  xl: '1280px',
  '2xl': '1440px',
} as const;

export const zIndex = {
  dropdown: 50,
  sticky: 40,
  modal: 60,
  popover: 70,
  toast: 80,
  tooltip: 90,
  loading: 100,
} as const;

export const transitions = {
  default: 'background-color var(--dur-base) var(--ease-out), border-color var(--dur-base) var(--ease-out), color var(--dur-base) var(--ease-out), transform var(--dur-base) var(--ease-spring), box-shadow var(--dur-base) var(--ease-out), opacity var(--dur-base) var(--ease-out)',
  fast: 'background-color var(--dur-fast) var(--ease-out), border-color var(--dur-fast) var(--ease-out), color var(--dur-fast) var(--ease-out), transform var(--dur-fast) var(--ease-spring), box-shadow var(--dur-fast) var(--ease-out), opacity var(--dur-fast) var(--ease-out)',
} as const;

export const focusRing = '0 0 0 2px rgba(var(--rgb-accent), 0.6)';

export type ColorScale = keyof typeof colors;
export type SpacingScale = keyof typeof spacing.scale;
export type RadiusScale = keyof typeof radius;
export type ShadowScale = keyof typeof shadows;
export type MotionDuration = keyof typeof motion.duration;
export type TypographyScale = keyof typeof typography.scale;