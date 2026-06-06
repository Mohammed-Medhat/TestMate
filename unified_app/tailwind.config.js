/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  // Safelist all dynamically-composed mode-color classes from theme.ts
  // (Tailwind's scanner cannot follow runtime object lookups)
  safelist: [
    // text
    'text-emerald-400','text-blue-400','text-violet-400','text-amber-400',
    // bg — tab fill + rail active
    'bg-emerald-500/10','bg-blue-500/10','bg-violet-500/10','bg-amber-500/10',
    'bg-emerald-500/15','bg-blue-500/15','bg-violet-500/15','bg-amber-500/15',
    // bg — count badges
    'bg-emerald-500/20','bg-blue-500/20','bg-violet-500/20','bg-amber-500/20',
    // border — tab active ring
    'border-emerald-500/30','border-blue-500/30','border-violet-500/30','border-amber-500/30',
    // ring — rail icon active
    'ring-1',
    'ring-emerald-500/30','ring-blue-500/30','ring-violet-500/30','ring-amber-500/30',
    // action button + toggle backgrounds (solid)
    'bg-blue-500','bg-blue-600','hover:bg-blue-600',
    'bg-violet-500','bg-violet-600','hover:bg-violet-600',
    'bg-amber-500','bg-amber-600',
    'bg-emerald-500',
    // focus rings on inputs
    'focus:border-blue-500','focus:border-violet-500',
    // checkbox / range accents
    'accent-blue-500','accent-violet-500',
    // select-all text
    'text-violet-400','text-violet-300','hover:text-violet-300',
    'text-blue-400',
  ],
  theme: {
    extend: {
      colors: {
        zinc: { 950: '#09090b' }
      },
      fontFamily: {
        mono: ['ui-monospace', 'SFMono-Regular', 'Menlo', 'Monaco', 'Consolas', 'monospace']
      }
    }
  },
  plugins: []
}
