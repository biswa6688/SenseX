import { NavLink, Outlet } from 'react-router-dom'
import { Logo } from './Logo'
import { ThemeToggle } from './ThemeToggle'

const NAV = [
  { to: '/', label: 'Home', end: true },
  { to: '/playground', label: 'Playground' },
  { to: '/models', label: 'Models' },
  { to: '/opencode-cli', label: 'OpenCode CLI' },
]

export function Layout() {
  return (
    <div className="min-h-screen bg-(--color-bg) text-(--color-fg)">
      <header className="sticky top-0 z-40 border-b border-(--color-border) bg-(--color-bg)/80 backdrop-blur">
        <div className="mx-auto flex max-w-6xl items-center justify-between px-6 py-3">
          <NavLink to="/" className="flex items-center gap-2">
            <Logo size={28} />
            <span className="text-lg font-semibold tracking-tight">SenseX</span>
          </NavLink>
          <nav className="flex items-center gap-1">
            {NAV.map(({ to, label, end }) => (
              <NavLink
                key={to}
                to={to}
                end={end}
                className={({ isActive }) =>
                  `rounded-full px-3 py-1.5 text-sm font-medium transition-colors ${
                    isActive
                      ? 'bg-(--color-brand)/10 text-(--color-brand)'
                      : 'text-(--color-fg-muted) hover:text-(--color-fg)'
                  }`
                }
              >
                {label}
              </NavLink>
            ))}
          </nav>
          <ThemeToggle />
        </div>
      </header>
      <main>
        <Outlet />
      </main>
    </div>
  )
}
