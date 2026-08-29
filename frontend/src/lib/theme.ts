import { create } from 'zustand'

export type ThemeMode = 'light' | 'dark' | 'system'

function resolveIsDark(mode: ThemeMode): boolean {
  if (mode === 'system') {
    return window.matchMedia('(prefers-color-scheme: dark)').matches
  }
  return mode === 'dark'
}

function applyTheme(mode: ThemeMode) {
  document.documentElement.classList.toggle('dark', resolveIsDark(mode))
}

interface ThemeState {
  mode: ThemeMode
  setMode: (mode: ThemeMode) => void
}

const STORAGE_KEY = 'sensex-theme'

const initialMode = (localStorage.getItem(STORAGE_KEY) as ThemeMode | null) ?? 'system'
applyTheme(initialMode)

export const useThemeStore = create<ThemeState>((set) => ({
  mode: initialMode,
  setMode: (mode) => {
    localStorage.setItem(STORAGE_KEY, mode)
    applyTheme(mode)
    set({ mode })
  },
}))

window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => {
  if (useThemeStore.getState().mode === 'system') applyTheme('system')
})
