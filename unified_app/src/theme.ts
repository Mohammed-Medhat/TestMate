import type { Mode } from './types'

export interface ModeTheme {
  /** Tailwind text class, e.g. 'text-emerald-400' */
  text:       string
  /** Tailwind bg-opacity class for active tab fill */
  bg:         string
  /** Tailwind border class for active tab ring */
  border:     string
  /** Full tailwind compound for rail icon active state */
  railActive: string
  /** Hex colour used for inline styles (ripple, SVG, etc.) */
  hex:        string
  /** Tailwind class for badge bg */
  badgeBg:    string
  /** Tailwind class for badge text */
  badgeText:  string
}

export const MODE_COLORS: Record<Mode, ModeTheme> = {
  combined: {
    text:       'text-emerald-400',
    bg:         'bg-emerald-500/10',
    border:     'border-emerald-500/30',
    railActive: 'bg-emerald-500/15 text-emerald-400 ring-1 ring-emerald-500/30',
    hex:        '#10b981',
    badgeBg:    'bg-emerald-500/20',
    badgeText:  'text-emerald-400',
  },
  parta: {
    text:       'text-blue-400',
    bg:         'bg-blue-500/10',
    border:     'border-blue-500/30',
    railActive: 'bg-blue-500/15 text-blue-400 ring-1 ring-blue-500/30',
    hex:        '#60a5fa',
    badgeBg:    'bg-blue-500/20',
    badgeText:  'text-blue-400',
  },
  partb: {
    text:       'text-violet-400',
    bg:         'bg-violet-500/10',
    border:     'border-violet-500/30',
    railActive: 'bg-violet-500/15 text-violet-400 ring-1 ring-violet-500/30',
    hex:        '#a78bfa',
    badgeBg:    'bg-violet-500/20',
    badgeText:  'text-violet-400',
  },
  partc: {
    text:       'text-amber-400',
    bg:         'bg-amber-500/10',
    border:     'border-amber-500/30',
    railActive: 'bg-amber-500/15 text-amber-400 ring-1 ring-amber-500/30',
    hex:        '#fbbf24',
    badgeBg:    'bg-amber-500/20',
    badgeText:  'text-amber-400',
  },
}
