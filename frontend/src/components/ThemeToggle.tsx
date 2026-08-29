import { Monitor, Moon, Sun } from 'lucide-react'
import { useThemeStore, type ThemeMode } from '../lib/theme'

const OPTIONS: { mode: ThemeMode; icon: typeof Sun; label: string }[] = [
  { mode: 'light', icon: Sun, label: 'Light' },
  { mode: 'dark', icon: Moon, label: 'Dark' },
  { mode: 'system', icon: Monitor, label: 'System' },
]

export function ThemeToggle() {
  const { mode, setMode } = useThemeStore()

  return (
    <div className="flex items-center gap-1 rounded-full border border-(--color-border) bg-(--color-bg-elevated) p-1">
      {OPTIONS.map(({ mode: m, icon: Icon, label }) => (
        <button
          key={m}
          type="button"
          aria-label={label}
          onClick={() => setMode(m)}
          className={`rounded-full p-1.5 transition-colors ${
            mode === m
              ? 'brand-gradient text-(--color-brand-fg)'
              : 'text-(--color-fg-muted) hover:text-(--color-fg)'
          }`}
        >
          <Icon size={14} />
        </button>
      ))}
    </div>
  )
}
